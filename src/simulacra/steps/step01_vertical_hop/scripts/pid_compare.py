# scripts/pid_compare.py
"""
Compare P vs PI vs PID on the vertical-hop rocket.

Usage:
    uv run python scripts/pid_compare.py

This script performs three short simulations (P-only, PI, PID) and plots altitude and thrust.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulacra.steps.step01_vertical_hop.sim import State, Rocket, PID
from simulacra.core.integrators import rk4_step


def run_simple_sim(rocket, pid, z_sp, t_final=6.0, dt=0.001, Tmax=300.0, initial_z=0.0):
    """
    Minimal sim loop that uses rk4_step directly and returns logs.

    - rocket: Rocket instance
    - pid: PID instance (configured for P/PI/PID by passing gains)
    - z_sp: setpoint (m)
    - t_final: total sim time (s)
    - dt: integrator timestep (s)
    - Tmax: maximum thrust (N)
    - initial_z: starting altitude
    """

    state = State(z=initial_z, v=0.0)
    t = 0.0

    # logs
    t_list, z_list, v_list, thrust_list, u_list, sp_list = [], [], [], [], [], []

    pid.reset()

    steps = int(np.ceil(t_final / dt))
    for i in range(steps):
        # measurement (perfect for this comparison)
        z_meas = state.z
        error = z_sp - z_meas

        # controller returns normalized control output (u) in u_min..u_max
        u = pid(error, dt)
        thrust = u * Tmax  # Newtons

        # dynamics for rk4: function from state-vector -> derivative vector
        def f(state_vec):
            # state_vec is numpy array [z, v]
            s = State.from_array(state_vec)
            return rocket.derivatives(s, thrust)

        # step with RK4
        new_arr = rk4_step(f, state.as_array(), dt)
        new_state = State.from_array(new_arr)

        # log
        t_list.append(t)
        z_list.append(state.z)
        v_list.append(state.v)
        thrust_list.append(thrust)
        u_list.append(u)
        sp_list.append(z_sp)

        # advance
        state = new_state
        t += dt

    return {
        "t": np.array(t_list),
        "z": np.array(z_list),
        "v": np.array(v_list),
        "thrust": np.array(thrust_list),
        "u": np.array(u_list),
        "sp": np.array(sp_list),
    }


def make_plots(results_dict, title_suffix=""):
    """
    results_dict: mapping label -> logs (returned by run_simple_sim)
    Will create two subplots: altitude (with setpoint) and thrust (with controller output overlay).
    """

    plt.figure(figsize=(10, 6))
    # altitude subplot
    ax1 = plt.subplot(2, 1, 1)
    for label, logs in results_dict.items():
        ax1.plot(logs["t"], logs["z"], label=label)
    # plot setpoint as dotted line (they're all same sp)
    sp = next(iter(results_dict.values()))["sp"]
    ax1.plot(logs["t"], sp, "k--", lw=1, label="setpoint")
    ax1.set_ylabel("altitude (m)")
    ax1.legend()
    ax1.grid(True)

    # thrust subplot (two y-lines: thrust N, and normalized u secondary axis)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    for label, logs in results_dict.items():
        ax2.plot(logs["t"], logs["thrust"], label=f"{label} thrust (N)")
    ax2.set_ylabel("thrust (N)")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    # also plot normalized u on right y-axis for clarity
    ax3 = ax2.twinx()
    for label, logs in results_dict.items():
        ax3.plot(logs["t"], logs["u"], "--", alpha=0.7, label=f"{label} u")
    ax3.set_ylabel("controller output (u)")
    # combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize="small")

    ax3.set_ylim(-0.1, 1.1)  # expect u in 0..1
    ax2.set_xlabel("time (s)")
    plt.suptitle("P vs PI vs PID comparison " + title_suffix)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    # rocket parameters
    mass = 10.0
    area = 0.05
    cd = 0.6
    g = 9.81
    rho0 = 1.225
    rocket = Rocket(mass_kg=mass, area=area, cd=cd, g=g, rho0=rho0)

    # common sim params
    z_sp = 2.0  # target altitude (m)
    Tmax = 300.0  # max thrust (N) — adjust to be able to lift the rocket
    dt = 0.002  # time step (s) — small but affordable for local runs
    t_final = 6.0

    # 1) P-only
    pid_p = PID(kp=2.0, ki=0.0, kd=0.0, u_min=0.0, u_max=1.0, integrator_limit=None)
    res_p = run_simple_sim(rocket, pid_p, z_sp, t_final=t_final, dt=dt, Tmax=Tmax)

    # 2) PI
    pid_pi = PID(kp=2.0, ki=0.8, kd=0.0, u_min=0.0, u_max=1.0, integrator_limit=8.0)
    res_pi = run_simple_sim(rocket, pid_pi, z_sp, t_final=t_final, dt=dt, Tmax=Tmax)

    # 3) PID
    pid_pid = PID(kp=2.2, ki=1.0, kd=0.7, u_min=0.0, u_max=1.0, integrator_limit=8.0)
    res_pid = run_simple_sim(rocket, pid_pid, z_sp, t_final=t_final, dt=dt, Tmax=Tmax)

    results = {"P-only": res_p, "PI": res_pi, "PID": res_pid}
    make_plots(results, title_suffix=f"(z_sp={z_sp} m, Tmax={Tmax} N)")


if __name__ == "__main__":
    main()
