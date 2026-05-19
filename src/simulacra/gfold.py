"""
G-FOLD: Guidance for Fuel-Optimal Large Diverts — adapted to our hopper.

Solves a convex (SOCP) optimization problem for the fuel-optimal 3-DoF
landing trajectory. Returns a time-indexed thrust profile that a tracking
controller can follow.

Based on:
  - Açıkmeşe & Ploen, "Convex Programming Approach to Powered Descent
    Guidance for Mars Landing" (JGCD, 2007)
  - JPL flight-tested on the Xombie rocket (G-FOLD, 2013)

Reference implementations:
  - references/gfold/ (jonnyhyman/G-FOLD-Python)
  - references/scvx/ (EmbersArc/SCvx)

Key formulation details (following the paper faithfully):
  - Control variable: u = T/m (thrust acceleration), NOT raw thrust force.
    This avoids the bilinear term T/m in the dynamics.
  - Mass variable: z = ln(m). Mass dynamics become dz/dt = -α·||u||,
    which is convex. This is the paper's core insight.
  - Slack variable: s ≥ ||u||. Fuel cost = Σ s[n] (convex).
  - Thrust bounds: convexified via first-order Taylor of T_max·exp(-z),
    giving linear constraints in (s, z). See eqs 34-36 of the paper.
  - No minimum thrust (engine throttles to zero) → no lower bound needed.
  - No planetary rotation (ω = 0).
  - 3-DoF only — attitude tracked by inner PD controller.

Usage:
    from simulacra.gfold import GFOLDSolver, HopperParams
    solver = GFOLDSolver(HopperParams())
    result = solver.solve(r0=[5, 20, -3], v0=[0, -2, 0], rf=[0, 0, 0])
    # result.actions: (N, 3) array of [throttle, tilt_x, tilt_z]
    # result.fuel_used: total fuel consumed (kg)
    # result.flight_time: optimal flight time (s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np


@dataclass
class HopperParams:
    """Physical parameters matching jax_hopper_env defaults."""

    dry_mass: float = 60.0       # kg
    fuel_mass: float = 80.0      # kg — initial fuel load
    max_thrust: float = 2000.0   # N
    burn_rate: float = 0.68      # kg/s at max thrust
    gravity: float = 9.81        # m/s² (positive; applied as -g along Y)
    max_tilt: float = 0.25       # rad — max tilt for RL action conversion
    dt: float = 0.02             # s — env timestep

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def isp(self) -> float:
        """Specific impulse (s)."""
        return self.max_thrust / (self.burn_rate * self.gravity)

    @property
    def alpha(self) -> float:
        """Fuel consumption parameter: 1 / (Isp * g0)."""
        return 1.0 / (self.isp * self.gravity)

    @property
    def max_accel(self) -> float:
        """Max acceleration at dry mass (m/s²)."""
        return self.max_thrust / self.dry_mass


@dataclass
class GFOLDResult:
    """Result of a G-FOLD trajectory optimization."""

    positions: np.ndarray      # (N, 3) — [x, y, z] trajectory
    velocities: np.ndarray     # (N, 3) — [vx, vy, vz]
    thrust_accel: np.ndarray   # (N, 3) — thrust acceleration in world frame (u = T/m)
    mass: np.ndarray           # (N,) — mass at each timestep
    times: np.ndarray          # (N,) — time at each node
    actions: np.ndarray        # (N, 3) — [throttle, tilt_x, tilt_z] in RL format
    flight_time: float         # total flight time (s)
    fuel_used: float           # fuel consumed (kg)
    solve_time: float          # solver wall time (s)
    status: str                # 'optimal', 'infeasible', etc.
    smoothness: dict           # thrust reversals, rms_throttle_rate, etc.


class GFOLDSolver:
    """Solves the fuel-optimal 3-DoF powered descent problem.

    Formulation (Açıkmeşe & Ploen 2007, eqs 1-36):

    Variables:
        u(t) ∈ R³  — thrust acceleration (= T/m) in world frame
        z(t) ∈ R   — log mass (= ln m)
        s(t) ∈ R   — slack for ||u||

    minimize    Σ s[n]                        (≈ fuel consumption)
    subject to:
        ṙ = v                                (kinematics)
        v̇ = u + g                            (Newton, with u = T/m)
        ż = -α s                             (mass depletion in log space)
        r(0) = r₀, v(0) = v₀                (initial conditions)
        r(tf) = rf, v(tf) ≈ 0               (terminal conditions)
        ||u|| ≤ s                            (SOC relaxation)
        s ≤ μ₂(1 - (z - z₀))               (convexified upper thrust bound)
        z₀ ≤ z ≤ z₁                         (mass physical bounds)
        r_y(t) ≥ 0                           (no underground)
        u_y ≥ cos(max_tilt) · s              (thrust pointing)
    """

    def __init__(self, params: Optional[HopperParams] = None):
        self.p = params or HopperParams()

    def solve(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        rf: np.ndarray = None,
        vf: np.ndarray = None,
        fuel_fraction: float = 1.0,
        N: int = 80,
        tf: Optional[float] = None,
        glideslope_deg: float = 30.0,
        verbose: bool = False,
    ) -> GFOLDResult:
        """Solve for the fuel-optimal trajectory.

        Uses the proper G-FOLD formulation with log-mass variables and
        convexified thrust bounds. Flight time is fixed (estimated or
        provided) — free-final-time requires successive convexification.

        Args:
            r0: Initial position [x, y, z] (m). Y is altitude.
            v0: Initial velocity [vx, vy, vz] (m/s).
            rf: Target position (default [0, 0, 0]).
            vf: Target velocity (default [0, 0, 0]).
            fuel_fraction: Initial fuel fraction (0-1).
            N: Number of discretization nodes.
            tf: Fixed flight time (s). Auto-estimated if None.
            glideslope_deg: Glideslope cone half-angle (degrees).
            verbose: Print solver output.

        Returns:
            GFOLDResult with optimal trajectory and RL-format actions.
        """
        import time as _time

        p = self.p
        r0 = np.asarray(r0, dtype=np.float64)
        v0 = np.asarray(v0, dtype=np.float64)
        rf = np.asarray(rf, dtype=np.float64) if rf is not None else np.zeros(3)
        vf = np.asarray(vf, dtype=np.float64) if vf is not None else np.zeros(3)

        m0 = p.dry_mass + fuel_fraction * p.fuel_mass
        g_vec = np.array([0.0, -p.gravity, 0.0])

        # ── Estimate flight time ──
        if tf is None:
            tf = self._estimate_flight_time(r0, v0, rf, m0)

        dt = tf / (N - 1)

        # ── Pre-compute mass bounds (eqs 34-36 of the paper) ──
        # r1 = min thrust = 0 (our engine throttles to zero)
        # r2 = max thrust
        r2 = p.max_thrust
        alpha = p.alpha

        # z0[n] = ln(m_wet - α·r2·n·dt) — minimum possible mass at node n
        # z1[n] = ln(m_wet - α·r1·n·dt) = ln(m_wet) — max mass (no fuel burned)
        # mu_2[n] = r2 / (m_wet - α·r2·n·dt) — linearization slope
        z0_vals = np.zeros(N)
        z1_vals = np.zeros(N)
        mu_2_vals = np.zeros(N)

        for n in range(N):
            z0_term = m0 - alpha * r2 * n * dt
            z1_term = m0  # r1 = 0, so z1_term = m0 always
            # Safety: z0_term must stay above dry mass
            z0_term = max(z0_term, p.dry_mass * 0.99)
            z0_vals[n] = np.log(z0_term)
            z1_vals[n] = np.log(z1_term)
            mu_2_vals[n] = r2 / z0_term

        # ── CVXPY Variables ──
        # u = T/m : thrust acceleration in world frame (3 x N)
        u = cp.Variable((3, N))
        # z = ln(m) : log-mass (1 x N, but we use (N,) shape)
        z = cp.Variable(N)
        # s = slack for ||u|| (N,)
        s = cp.Variable(N, nonneg=True)
        # Position and velocity
        r = cp.Variable((3, N))
        v = cp.Variable((3, N))

        # ── Constraints ──
        constraints = []

        # Initial conditions
        constraints += [r[:, 0] == r0, v[:, 0] == v0]
        constraints += [z[0] == np.log(m0)]

        # Terminal conditions
        constraints += [r[:, -1] == rf]
        constraints += [cp.norm(v[:, -1] - vf) <= 0.3]  # small velocity margin
        constraints += [s[-1] == 0]  # zero thrust at landing

        # Dynamics (leapfrog integration, same as reference)
        for n in range(N - 1):
            # v̇ = u + g → v[n+1] = v[n] + dt/2 * ((u[n]+g) + (u[n+1]+g))
            constraints += [
                v[:, n + 1] == v[:, n] + (dt / 2) * ((u[:, n] + g_vec) + (u[:, n + 1] + g_vec))
            ]
            # ṙ = v → r[n+1] = r[n] + dt/2 * (v[n+1] + v[n])
            constraints += [
                r[:, n + 1] == r[:, n] + (dt / 2) * (v[:, n + 1] + v[:, n])
            ]
            # ż = -α·s → z[n+1] = z[n] - α·dt/2·(s[n] + s[n+1])
            constraints += [
                z[n + 1] == z[n] - (alpha * dt / 2) * (s[n] + s[n + 1])
            ]

        # SOC constraint: ||u[n]|| ≤ s[n]
        for n in range(N):
            constraints += [cp.norm(u[:, n]) <= s[n]]

        # Convexified thrust bounds (eqs 34-36)
        # Upper: s[n] ≤ μ₂[n] · (1 - (z[n] - z₀[n]))
        # Mass bounds: z₀[n] ≤ z[n] ≤ z₁[n]
        for n in range(1, N):  # skip n=0 (z[0] is fixed)
            constraints += [s[n] <= mu_2_vals[n] * (1 - (z[n] - z0_vals[n]))]
            constraints += [z[n] >= z0_vals[n]]
            constraints += [z[n] <= z1_vals[n]]

        # No underground
        for n in range(N):
            constraints += [r[1, n] >= 0.0]

        # Thrust pointing: u_y ≥ cos(max_tilt) · s
        # (thrust must point mostly upward)
        cos_tilt = np.cos(p.max_tilt)
        for n in range(N):
            constraints += [u[1, n] >= cos_tilt * s[n]]

        # Glideslope: horizontal distance from target ≤ altitude / tan(angle)
        if glideslope_deg > 0:
            tan_gs = np.tan(np.radians(glideslope_deg))
            for n in range(N):
                constraints += [
                    cp.norm(cp.hstack([r[0, n] - rf[0], r[2, n] - rf[2]])) <= (r[1, n] - rf[1]) / tan_gs
                ]

        # ── Objective: maximize final mass = maximize z[N-1] ──
        # Equivalent to minimizing fuel (Paper Problem 4)
        objective = cp.Maximize(z[-1])

        # ── Solve ──
        problem = cp.Problem(objective, constraints)

        t_start = _time.time()
        try:
            problem.solve(solver=cp.CLARABEL, verbose=verbose)
        except cp.SolverError:
            try:
                problem.solve(solver=cp.SCS, verbose=verbose, max_iters=5000)
            except cp.SolverError:
                return self._failure_result(m0, _time.time() - t_start, 'solver_error')
        solve_time = _time.time() - t_start

        if problem.status not in ('optimal', 'optimal_inaccurate'):
            return self._failure_result(m0, solve_time, problem.status)

        # ── Extract solution ──
        u_val = u.value.T   # (N, 3) — thrust acceleration
        r_val = r.value.T   # (N, 3) — position
        v_val = v.value.T   # (N, 3) — velocity
        z_val = z.value      # (N,) — log-mass
        s_val = s.value      # (N,) — slack
        t_val = np.linspace(0, tf, N)

        # Recover mass from log-mass
        m_val = np.exp(z_val)
        fuel_used = m0 - m_val[-1]

        # Convert to RL actions
        actions = self._accel_to_rl_actions(u_val, m_val)

        # Smoothness metrics
        smoothness = self._compute_smoothness(actions, tf, N)

        return GFOLDResult(
            positions=r_val,
            velocities=v_val,
            thrust_accel=u_val,
            mass=m_val,
            times=t_val,
            actions=actions,
            flight_time=tf,
            fuel_used=fuel_used,
            solve_time=solve_time,
            status=problem.status,
            smoothness=smoothness,
        )

    def _estimate_flight_time(
        self, r0: np.ndarray, v0: np.ndarray, rf: np.ndarray, m0: float
    ) -> float:
        """Estimate minimum feasible flight time from initial conditions.

        For fuel-optimal powered descent, shorter flight time = less fuel
        (extra time means hovering against gravity). So we estimate the
        MINIMUM feasible time, then add a small margin.

        Key physics: solve for time to arrest velocity and reach target
        using available net acceleration a_net = T_max/m - g.
        """
        p = self.p
        altitude = max(r0[1] - rf[1], 0.1)
        horiz_dist = np.sqrt((r0[0] - rf[0])**2 + (r0[2] - rf[2])**2)
        v_down = max(-v0[1], 0.0)  # downward speed (positive)
        v_horiz = np.sqrt(v0[0]**2 + v0[2]**2)

        # Net vertical acceleration (thrust - gravity) at initial mass
        a_net = p.max_thrust / m0 - p.gravity
        a_net = max(a_net, 2.0)

        # Vertical: time to soft-land from altitude h with initial downward speed
        # Using kinematic equation: h = v_down*t - 0.5*a_net*t² = 0
        # → t = (v_down + sqrt(v_down² + 2*a_net*h)) / a_net
        discriminant = v_down**2 + 2 * a_net * altitude
        t_vert = (v_down + np.sqrt(discriminant)) / a_net

        # Horizontal: time to traverse lateral distance with available accel
        # Optimal bang-bang: t = 2*sqrt(d / a_lat) where a_lat = a_net * sin(max_tilt)
        a_lat = a_net * np.sin(p.max_tilt)
        a_lat = max(a_lat, 1.0)
        if horiz_dist > 0.1:
            t_horiz = 2 * np.sqrt(horiz_dist / a_lat) + v_horiz / a_lat
        else:
            t_horiz = v_horiz / a_lat

        # Liftoff needs extra time to go up and come back
        if altitude < 1.0 and v_down < 0.5 and horiz_dist > 0.5:
            t_liftoff = 2 * np.sqrt(2 * max(horiz_dist, 5.0) / a_net)
            t_vert = max(t_vert, t_liftoff)

        # Take the larger of vertical and horizontal, add 30% margin
        tf = max(t_vert, t_horiz) * 1.3
        tf = np.clip(tf, 3.0, 20.0)

        return tf

    def _failure_result(self, m0: float, solve_time: float, status: str) -> GFOLDResult:
        return GFOLDResult(
            positions=np.zeros((1, 3)), velocities=np.zeros((1, 3)),
            thrust_accel=np.zeros((1, 3)), mass=np.array([m0]),
            times=np.array([0.0]), actions=np.zeros((1, 3)),
            flight_time=0.0, fuel_used=0.0,
            solve_time=solve_time, status=status, smoothness={},
        )

    def _accel_to_rl_actions(self, u: np.ndarray, mass: np.ndarray) -> np.ndarray:
        """Convert thrust acceleration u = T/m to RL action format [throttle, tilt_x, tilt_z].

        T_world = u * m (force vector)
        throttle = ||T|| / max_thrust = ||u|| * m / max_thrust
        tilt: desired_dir = [tilt_x, 1.0, tilt_z] normalized
            → tilt_x = u_x / u_y, tilt_z = u_z / u_y (when u_y > 0)
            → action = tilt / max_tilt, clipped to [-1, 1]
        """
        p = self.p
        N = u.shape[0]
        actions = np.zeros((N, 3))

        for k in range(N):
            T_world = u[k] * mass[k]  # force vector (N)
            thrust_mag = np.linalg.norm(T_world)
            throttle = np.clip(thrust_mag / p.max_thrust, 0.0, 1.0)

            if thrust_mag > 1.0 and u[k, 1] > 0.1:
                raw_tilt_x = u[k, 0] / u[k, 1]
                raw_tilt_z = u[k, 2] / u[k, 1]
                tilt_x = np.clip(raw_tilt_x / p.max_tilt, -1.0, 1.0)
                tilt_z = np.clip(raw_tilt_z / p.max_tilt, -1.0, 1.0)
            else:
                tilt_x = 0.0
                tilt_z = 0.0

            actions[k] = [throttle, tilt_x, tilt_z]

        return actions

    def _compute_smoothness(self, actions: np.ndarray, tf: float, N: int) -> dict:
        """Compute smoothness metrics from the action trajectory."""
        throttle = actions[:, 0]
        tilt_x = actions[:, 1]
        tilt_z = actions[:, 2]

        dthrottle = np.diff(throttle)

        # Thrust reversals (sign changes in dthrottle)
        if len(dthrottle) > 1:
            signs = np.sign(dthrottle)
            nonzero = signs[signs != 0]
            reversals = int(np.sum(np.diff(nonzero) != 0)) if len(nonzero) > 1 else 0
        else:
            reversals = 0

        # RMS throttle rate
        rms_throttle_rate = float(np.sqrt(np.mean(dthrottle ** 2)))

        # Mean gimbal rate
        dtilt_x = np.diff(tilt_x)
        dtilt_z = np.diff(tilt_z)
        mean_gimbal_rate = float(np.mean(np.abs(dtilt_x) + np.abs(dtilt_z)))

        return {
            'thrust_reversals': reversals,
            'rms_throttle_rate': rms_throttle_rate,
            'mean_gimbal_rate': mean_gimbal_rate,
        }


def solve_scenario(
    scenario: str,
    params: Optional[HopperParams] = None,
    seed: int = 0,
    verbose: bool = False,
) -> GFOLDResult:
    """Solve G-FOLD for a named scenario matching the RL training set.

    Scenarios match jax_hopper_env scenario types:
        approach, hover, landing, diagonal, liftoff
    """
    rng = np.random.default_rng(seed)
    p = params or HopperParams()

    if scenario == 'approach':
        alt = rng.uniform(10.0, 30.0)
        r0 = np.array([rng.uniform(-5, 5), alt, rng.uniform(-5, 5)])
        v0 = np.array([rng.uniform(-1, 1), rng.uniform(-1.5, 0), rng.uniform(-1, 1)])
    elif scenario == 'hover':
        alt = rng.uniform(5.0, 15.0)
        r0 = np.array([rng.uniform(-3, 3), alt, rng.uniform(-3, 3)])
        v0 = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
    elif scenario == 'landing':
        alt = rng.uniform(2.0, 8.0)
        r0 = np.array([rng.uniform(-2, 2), alt, rng.uniform(-2, 2)])
        v0 = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 0), rng.uniform(-0.5, 0.5)])
    elif scenario == 'diagonal':
        alt = rng.uniform(15.0, 30.0)
        offset = rng.uniform(5, 15)
        angle = rng.uniform(0, 2 * np.pi)
        r0 = np.array([offset * np.cos(angle), alt, offset * np.sin(angle)])
        v0 = np.array([rng.uniform(-1, 1), rng.uniform(-2, 0), rng.uniform(-1, 1)])
    elif scenario == 'liftoff':
        r0 = np.array([rng.uniform(-1, 1), 0.1, rng.uniform(-1, 1)])
        v0 = np.array([0.0, 0.0, 0.0])
        solver = GFOLDSolver(p)
        return solver.solve(r0=r0, v0=v0, glideslope_deg=0.0, tf=8.0, verbose=verbose)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    solver = GFOLDSolver(p)
    return solver.solve(r0=r0, v0=v0, verbose=verbose)


def run_benchmark(
    n_episodes: int = 200,
    scenarios: Optional[list[str]] = None,
    params: Optional[HopperParams] = None,
    verbose: bool = False,
) -> dict:
    """Run G-FOLD across multiple scenarios, collecting performance metrics.

    Returns dict with per-scenario and overall statistics matching
    the RL eval format for direct comparison.
    """
    if scenarios is None:
        scenarios = ['approach', 'hover', 'landing', 'diagonal', 'liftoff']

    p = params or HopperParams()
    results_by_scenario = {}

    for scenario in scenarios:
        scenario_results = []
        n_success = 0
        n_infeasible = 0

        for ep in range(n_episodes):
            result = solve_scenario(scenario, params=p, seed=ep * 1000 + hash(scenario) % 1000)

            if result.status in ('optimal', 'optimal_inaccurate'):
                n_success += 1
                scenario_results.append({
                    'fuel_used': result.fuel_used,
                    'flight_time': result.flight_time,
                    'solve_time': result.solve_time,
                    **result.smoothness,
                })
            else:
                n_infeasible += 1

        # Aggregate
        success_rate = n_success / max(1, n_episodes)
        if scenario_results:
            avg = {k: np.mean([r[k] for r in scenario_results]) for k in scenario_results[0]}
        else:
            avg = {}

        results_by_scenario[scenario] = {
            'success_rate': success_rate,
            'n_success': n_success,
            'n_infeasible': n_infeasible,
            **avg,
        }

        if verbose:
            print(f"  {scenario:>10}: {success_rate:.0%} success | "
                  f"fuel={avg.get('fuel_used', 0):.2f}kg | "
                  f"t={avg.get('flight_time', 0):.1f}s | "
                  f"rev={avg.get('thrust_reversals', 0):.0f} | "
                  f"solve={avg.get('solve_time', 0)*1000:.0f}ms")

    # Overall
    total_success = sum(r['n_success'] for r in results_by_scenario.values())
    total_episodes = n_episodes * len(scenarios)

    return {
        'overall_success_rate': total_success / total_episodes,
        'scenarios': results_by_scenario,
    }


if __name__ == '__main__':
    print("G-FOLD Benchmark — Hopper Scenarios (proper formulation)")
    print("=" * 60)
    results = run_benchmark(n_episodes=50, verbose=True)
    print(f"\nOverall success: {results['overall_success_rate']:.0%}")
