"""
NEAT Hopper Controller: drop-in replacement for HopperController using an evolved network.

Loads a NEAT-evolved genome (from TensorNEAT) and runs it through the same
sensor/actuator interface as RLHopperController. The PD attitude inner loop
is identical to the PPO GuidanceWrapper.

NEAT networks are typically very small (5-50 connections), producing smooth
control without the high-frequency oscillations of large MLPs.

Usage:
    from simulacra.onboard.neat_hopper_controller import NEATHopperController
    controller = NEATHopperController(
        throttle=actuators._throttle,
        gimbal=actuators._gimbal,
        waypoints=waypoints,
        genome_path="runs/neat_20260322/best_genome.npz",
    )
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..phenomena.actuators import (
    ThrottleActuator,
    ThrottleCommand,
    GimbalActuator,
    GimbalCommand,
)
from ..phenomena.sensors import Sensors
from .controller_legacy import Controller
from .hopper_controller import Waypoint, HopperConfig


class NEATHopperController(Controller):
    """
    NEAT-evolved policy wrapped with waypoint sequencing and PD attitude control.

    Same external interface as RLHopperController:
      - .current_target, .phase_name, .finished for telemetry
      - .run() reads sensors and applies actuator commands
      - .with_sensors() attaches sensor suite
    """

    # Inner PD attitude gains (identical to RLHopperController / GuidanceWrapper)
    ATT_KP = 500.0
    ATT_KD = 200.0
    MAX_THRUST = 2000.0
    MOMENT_ARM = 2.0
    MAX_GIMBAL = 0.26

    def __init__(
        self,
        throttle: ThrottleActuator,
        gimbal: GimbalActuator,
        waypoints: list[Waypoint],
        genome_path: str | Path,
        config: Optional[HopperConfig] = None,
        max_tilt: float = 0.25,
    ):
        self.throttle = throttle
        self.gimbal = gimbal
        self.waypoints = waypoints
        self.config = config or HopperConfig()
        self.max_tilt = max_tilt
        self.sensors: Optional[Sensors] = None

        # Load genome
        self._load_genome(Path(genome_path))

        # Waypoint state (same as HopperController)
        self._wp_idx = 0
        self._wp_state = "TRACKING"
        self._hold_timer = 0.0

        # Telemetry
        self.current_target = np.zeros(3)
        self.phase_name = ""

        # Last forward pass activations (for visualization)
        self._last_activations: dict[int, float] = {}

    def _load_genome(self, genome_path: Path):
        """Load NEAT genome from TensorNEAT checkpoint.

        Supports two formats:
          1. NPZ file with 'nodes' and 'conns' arrays (TensorNEAT export)
          2. Directory with genome state (TensorNEAT save_dir)
        """
        if genome_path.suffix == ".npz":
            data = np.load(genome_path, allow_pickle=True)
            self._nodes = data["nodes"]
            self._conns = data["conns"]
        elif genome_path.is_dir():
            # TensorNEAT save directory — load best genome
            data = np.load(genome_path / "best_genome.npz", allow_pickle=True)
            self._nodes = data["nodes"]
            self._conns = data["conns"]
        else:
            raise ValueError(f"Unknown genome format: {genome_path}")

        # Build a fast numpy forward pass from the topology
        self._build_forward_pass()

    def _build_forward_pass(self):
        """Pre-compute the network topology for fast numpy inference.

        TensorNEAT genome format:
          nodes: (max_nodes, 5) — [key, bias, response, aggregation_idx, activation_idx]
          conns: (max_conns, 3) — [src_key, dst_key, weight]
        NaN rows = unused slots. Node keys: 0..21=inputs, 22..24=outputs, 25+=hidden.
        """
        nodes = self._nodes
        conns = self._conns

        # Activation lookup (default NEAT uses sigmoid for hidden nodes)
        def _sigmoid(x):
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

        self._activations = {
            0: _sigmoid,       # ACT.sigmoid (default)
            -1: lambda x: x,   # identity
        }

        # Parse nodes: {key: (bias, response, agg_idx, act_idx)}
        self._node_info = {}
        for n in nodes:
            if np.isnan(n[0]):
                continue
            key = int(n[0])
            bias = float(n[1])
            response = float(n[2])
            agg_idx = int(n[3])
            act_idx = int(n[4])
            self._node_info[key] = (bias, response, agg_idx, act_idx)

        # Parse connections: [(src, dst, weight)]
        self._active_conns = []
        for c in conns:
            if np.isnan(c[0]):
                continue
            src, dst, weight = int(c[0]), int(c[1]), float(c[2])
            self._active_conns.append((src, dst, weight))

        # Auto-detect num_inputs from topology.
        # In TensorNEAT: output keys are the smallest destination keys.
        # Key layout: inputs=[0..N-1], outputs=[N..N+2], hidden=[N+3..].
        # So min(destination keys) = num_inputs.
        dst_keys = set(dst for _, dst, _ in self._active_conns)
        if dst_keys:
            self._num_inputs = min(dst_keys)
        else:
            self._num_inputs = 22  # fallback
        self._num_outputs = 3
        self._input_keys = set(range(self._num_inputs))
        self._output_keys = set(range(self._num_inputs, self._num_inputs + self._num_outputs))

        # Topological sort (Kahn's algorithm)
        all_nodes = set()
        for src, dst, _ in self._active_conns:
            all_nodes.add(src)
            all_nodes.add(dst)
        # Include output nodes even if no connections reach them
        all_nodes.update(self._output_keys)

        in_degree = {n: 0 for n in all_nodes}
        adj = {n: [] for n in all_nodes}
        for src, dst, _ in self._active_conns:
            adj[src].append(dst)
            in_degree[dst] = in_degree.get(dst, 0) + 1

        queue = sorted([n for n in all_nodes if in_degree[n] == 0])
        self._eval_order = []
        visited = set()
        while queue:
            node = queue.pop(0)
            if node not in visited:
                self._eval_order.append(node)
                visited.add(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in visited:
                    queue.append(neighbor)

        # Build per-node incoming connections for fast lookup
        self._incoming = {}
        for src, dst, weight in self._active_conns:
            if dst not in self._incoming:
                self._incoming[dst] = []
            self._incoming[dst].append((src, weight))

        n_conns = len(self._active_conns)
        n_hidden = len(self._node_info) - self._num_inputs - self._num_outputs
        print(f"NEAT genome: {n_conns} connections, {max(0, n_hidden)} hidden nodes, {self._num_inputs} inputs")

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        """Evaluate NEAT network on observation. Returns 3-dim output.

        Supports 22-dim (standard) or 26-dim (with next-waypoint info).
        Node forward: z = activation(bias + response * sum(weight_i * input_i))
        Output nodes skip activation; output_transform (tanh) is applied instead.
        """
        values = {}

        # Set input values
        for i in range(self._num_inputs):
            values[i] = float(obs[i])

        # Evaluate in topological order
        for key in self._eval_order:
            if key in self._input_keys:
                continue

            info = self._node_info.get(key)
            if info is None:
                continue
            bias, response, agg_idx, act_idx = info

            # Aggregate: sum of weighted inputs
            agg = 0.0
            for src, weight in self._incoming.get(key, []):
                if src in values:
                    agg += values[src] * weight

            z = bias + response * agg

            # Hidden nodes: apply activation; output nodes: skip (output_transform later)
            if key not in self._output_keys:
                act_fn = self._activations.get(act_idx, self._activations[0])
                z = act_fn(z)

            values[key] = z

        # Read outputs and apply output_transform (tanh)
        output = np.zeros(3)
        for i, key in enumerate(sorted(self._output_keys)):
            output[i] = math.tanh(values.get(key, 0.0))

        # Store activations for visualization
        self._last_activations = {k: float(v) for k, v in values.items()}

        return output

    def get_activations(self) -> dict[str, float]:
        """Return last forward pass activations as {str(key): value}."""
        return {str(k): v for k, v in self._last_activations.items()}

    def get_topology(self) -> dict:
        """Export network topology for visualization.

        Returns dict with: nodes, connections, input_labels, output_labels.
        """
        input_labels_22 = [
            "gps_x", "alt", "gps_z",
            "vx", "vy", "vz",
            "qw", "qx", "qy", "qz",
            "ωx", "ωy", "ωz",
            "Δx", "Δy", "Δz",
            "dist", "v_toward", "brake_m",
            "track", "land", "liftoff",
        ]
        input_labels_26 = input_labels_22 + ["nx_dx", "nx_dy", "nx_dz", "is_final"]
        input_labels = input_labels_26 if self._num_inputs > 22 else input_labels_22
        output_labels = ["throttle", "tilt_x", "tilt_z"]

        # Assign layers for layout
        # Build layer assignment via BFS from inputs
        layers = {}
        for k in self._input_keys:
            layers[k] = 0
        for k in self._output_keys:
            layers[k] = -1  # placeholder

        # Forward pass through eval order to assign layers
        for key in self._eval_order:
            if key in self._input_keys:
                continue
            max_src_layer = 0
            for src, _ in self._incoming.get(key, []):
                if src in layers and layers[src] >= 0:
                    max_src_layer = max(max_src_layer, layers[src])
            layers[key] = max_src_layer + 1

        # Set output layer to max + 1
        max_layer = max((l for l in layers.values() if l >= 0), default=0)
        output_layer = max_layer + 1
        for k in self._output_keys:
            layers[k] = output_layer

        nodes = []
        for key in sorted(self._node_info.keys()):
            if key in self._input_keys:
                ntype = "input"
                label = input_labels[key] if key < len(input_labels) else f"in{key}"
            elif key in self._output_keys:
                ntype = "output"
                oidx = key - self._num_inputs
                label = output_labels[oidx] if oidx < len(output_labels) else f"out{oidx}"
            else:
                ntype = "hidden"
                label = f"h{key}"
            nodes.append({
                "key": key,
                "label": label,
                "type": ntype,
                "layer": layers.get(key, 0),
            })

        connections = [
            {"src": src, "dst": dst, "weight": round(weight, 4)}
            for src, dst, weight in self._active_conns
        ]

        return {
            "nodes": nodes,
            "connections": connections,
            "input_labels": input_labels,
            "output_labels": output_labels,
        }

    def with_sensors(self, sensors: Sensors) -> "NEATHopperController":
        self.sensors = sensors
        return self

    @property
    def finished(self) -> bool:
        return self._wp_idx >= len(self.waypoints)

    def run(self) -> None:
        if self.sensors is None:
            raise ValueError("Sensors not attached to controller")
        if self.finished:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            return

        dt = 0.02

        # ── Read sensors ──
        gps = self.sensors.GPS.read()
        alt = self.sensors.Barometer.read()
        vel = self.sensors.VelocitySensor.read()
        orientation = self.sensors.OrientationSensor.read()
        omega_body = self.sensors.GyroSensor.read()
        fuel_frac = self.sensors.FuelSensor.read()

        pos = np.array([gps.x, alt, gps.z])
        vel_vec = np.array([vel.x, vel.y, vel.z])
        quat = orientation.as_array()  # [w, x, y, z]
        omega = np.array([omega_body.x, omega_body.y, omega_body.z])

        # ── Waypoint management ──
        target = self._get_target(pos, dt)
        self.current_target = target.copy()

        if self.finished:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            return

        # Landing detection
        speed = np.linalg.norm(vel_vec)
        cfg = self.config
        is_last_wp = (self._wp_idx == len(self.waypoints) - 1)
        if is_last_wp and alt < cfg.landing_altitude and speed < cfg.landing_speed:
            self.throttle.apply(ThrottleCommand(thrust_N=0.0))
            self.gimbal.apply(GimbalCommand(desired_outer_angle=0.0, desired_inner_angle=0.0))
            self._wp_idx = len(self.waypoints)
            return

        # ── Build observation (22-dim) ──
        state = np.array([
            gps.x, alt, gps.z,
            vel.x, vel.y, vel.z,
            quat[0], quat[1], quat[2], quat[3],
            omega[0], omega[1], omega[2],
        ], dtype=np.float32)

        goal_rel = (target - pos).astype(np.float32)

        dist_to_wp = float(np.linalg.norm(goal_rel))
        if dist_to_wp > 0.01:
            dir_to_wp = goal_rel / dist_to_wp
            vel_toward = float(np.dot(vel_vec, dir_to_wp))
        else:
            vel_toward = 0.0

        g = cfg.gravity
        total_mass = cfg.dry_mass + fuel_frac * cfg.full_fuel_mass
        a_max = max(cfg.max_thrust / total_mass - g, 1.0)
        brake_margin = dist_to_wp - speed ** 2 / (2 * a_max)

        features = np.array([dist_to_wp, vel_toward, brake_margin], dtype=np.float32)

        is_land = target[1] < 1.0
        on_ground = alt < 0.5 and speed < 0.5
        is_liftoff = on_ground and not is_land
        is_track = not is_land and not is_liftoff
        mode = np.array([float(is_track), float(is_land), float(is_liftoff)], dtype=np.float32)

        obs = np.concatenate([state, goal_rel, features, mode])

        # ── Append next-waypoint info for 26-dim genomes ──
        if self._num_inputs > 22:
            if self._wp_idx + 1 < len(self.waypoints):
                next_target = self.waypoints[self._wp_idx + 1].position
            else:
                next_target = target  # Final waypoint: next = current
            next_goal_rel = (next_target - pos).astype(np.float32)
            is_final = float(self._wp_idx >= len(self.waypoints) - 1)
            obs = np.concatenate([obs, next_goal_rel, np.array([is_final], dtype=np.float32)])

        # ── Run NEAT network ──
        action = self._forward(obs)

        # Map from tanh output: throttle [-1,1]→[0,1], tilts stay [-1,1]
        throttle_cmd = float((action[0] + 1.0) / 2.0)
        tilt_x = float(action[1]) * self.max_tilt
        tilt_z = float(action[2]) * self.max_tilt

        # ── Inner PD attitude controller (identical to RLHopperController) ──
        desired_dir = np.array([tilt_x, 1.0, tilt_z])
        desired_dir = desired_dir / np.linalg.norm(desired_dir)

        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        nose_world = _rotate_body_to_world(qw, qx, qy, qz, [1, 0, 0])
        error_world = np.cross(nose_world, desired_dir)
        error_body = _rotate_world_to_body(qw, qx, qy, qz, error_world)

        tau = self.ATT_KP * error_body - self.ATT_KD * omega
        thrust_n = throttle_cmd * self.MAX_THRUST

        if thrust_n > 100:
            scale = 1.0 / (thrust_n * self.MOMENT_ARM)
            outer = float(np.clip(-tau[2] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
            inner = float(np.clip(-tau[1] * scale, -self.MAX_GIMBAL, self.MAX_GIMBAL))
        else:
            outer = 0.0
            inner = 0.0

        # ── Apply commands ──
        self.throttle.apply(ThrottleCommand(thrust_N=thrust_n))
        self.gimbal.apply(GimbalCommand(
            desired_outer_angle=outer,
            desired_inner_angle=inner,
        ))

    def _get_target(self, pos: np.ndarray, dt: float) -> np.ndarray:
        """Same waypoint sequencing as HopperController."""
        wp = self.waypoints[self._wp_idx]

        if self._wp_state == "TRACKING":
            dist = np.linalg.norm(wp.position - pos)
            if dist < wp.acceptance_radius:
                self._wp_state = "HOLDING"
                self._hold_timer = 0.0
                self.phase_name = f"HOLD wp{self._wp_idx} ({wp.hold_time:.0f}s)"
            else:
                self.phase_name = f"TRACK wp{self._wp_idx} ({dist:.1f}m)"

        if self._wp_state == "HOLDING":
            self._hold_timer += dt
            if self._hold_timer >= wp.hold_time:
                self._wp_idx += 1
                self._wp_state = "TRACKING"
                if self._wp_idx < len(self.waypoints):
                    self.phase_name = f"TRACK wp{self._wp_idx}"
                else:
                    self.phase_name = "DONE"

        if self._wp_idx < len(self.waypoints):
            return self.waypoints[self._wp_idx].position.copy()
        else:
            return wp.position.copy()


# ──────────────────────────────────────────────────────────────
# Rotation helpers (same as rl_hopper_controller.py)
# ──────────────────────────────────────────────────────────────

def _rotate_body_to_world(qw, qx, qy, qz, vec_body):
    qv = np.array([qx, qy, qz])
    v = np.asarray(vec_body, dtype=np.float64)
    c1 = np.cross(qv, v)
    c2 = np.cross(qv, c1)
    return v + 2.0 * qw * c1 + 2.0 * c2


def _rotate_world_to_body(qw, qx, qy, qz, vec_world):
    qv = np.array([-qx, -qy, -qz])
    v = np.asarray(vec_world, dtype=np.float64)
    c1 = np.cross(qv, v)
    c2 = np.cross(qv, c1)
    return v + 2.0 * qw * c1 + 2.0 * c2
