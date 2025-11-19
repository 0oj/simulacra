import json
import sys

from pydantic import BaseModel


class SimState(BaseModel):
    t: float
    pos_x: float
    pos_y: float
    gimbal_angle: float
    orientation: float
    angular_velocity: float
    thrust_magnitude: float  # True thrust magnitude (N)
    gps_x: float
    gps_y: float
    imu_orientation: float
    imu_angular_velocity: float
    thrust_sensor: float
    rocket_length: float


def write_sim_state(state: SimState) -> None:
    """
    Write the simulation state as a JSON line (NDJSON format) to stdout.

    Args:
        state: The simulation state (as a Pydantic model).
    """
    try:
        json_str = state.model_dump_json(exclude_none=True)
        # Write directly to sys.stdout for more control (better for I/O redirection & scripts)
        sys.stdout.write(json_str + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        # Pipe was closed (e.g., animation script finished reading)
        # This is normal when piping, so we can safely ignore it
        sys.exit(0)


def read_sim_state() -> SimState:
    """
    Read a single SimState from stdin (one JSON line, NDJSON format).

    Returns:
        SimState: The parsed simulation state.

    Raises:
        EOFError: If end of input is reached.
    """
    # Read a single unbuffered line from stdin.
    line = sys.stdin.readline()
    if not line:
        raise EOFError("End of input reached.")
    line = line.strip()
    while line == "":
        # Skip empty lines (don't return None/raise, just look for any content)
        line = sys.stdin.readline()
        if not line:
            raise EOFError("End of input reached.")
        line = line.strip()
    data = json.loads(line)
    return SimState(**data)
