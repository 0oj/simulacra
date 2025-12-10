"""
The record of truth. God writes the simulation state to stdout,
capturing both the true state and what the rocket perceived.
"""

import json
import sys
from typing import Optional, Union, Literal, Tuple

from .physics.vec2 import RocketVec2, WorldVec2
from .physics.vec3 import RocketVec3, WorldVec3

from pydantic import BaseModel, Field

from enum import Enum

SCHEMA_VERSION = "0.1.0"


class MessageType(str, Enum):
    METADATA = "metadata"
    CONSTANT = "constant"
    VARIABLE = "variable"


class BaseMessage(BaseModel):
    message_type: MessageType

    model_config = {
        "use_enum_values": True,
    }


# ============================================================================
# Message Types
# ============================================================================


class MetadataMessage(BaseMessage):
    """Metadata message - information about the simulation run itself."""

    message_type: Literal[MessageType.METADATA] = MessageType.METADATA
    schema_version: str = Field(default=SCHEMA_VERSION)
    run_by: Optional[str] = Field(default=None, description="Who ran the simulation")
    run_at: Optional[str] = Field(
        default=None, description="When the run started (ISO timestamp)"
    )
    git_commit: Optional[str] = Field(default=None, description="Git commit hash")
    scenario_name: Optional[str] = Field(
        default=None, description="Name of the scenario"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")
    dt: Optional[float] = Field(default=None, description="Time step (seconds)")
    duration: Optional[float] = Field(
        default=None, description="Total simulation duration (seconds)"
    )


class ConstantMessage(BaseMessage):
    """Constant message - configuration that can change over time via events."""

    message_type: Literal[MessageType.CONSTANT] = MessageType.CONSTANT

    timestamp: float = Field(
        description="Simulation time when this constant value becomes valid"
    )

    # Rocket properties
    rocket_length: Optional[float] = Field(
        default=None, description="Length of the rocket (meters)"
    )
    rocket_mass: Optional[float] = Field(
        default=None, description="Mass of the rocket (kilograms)"
    )
    rocket_center_of_mass_offset: Optional[RocketVec2] = Field(
        default=None,
        description="Offset from the base of the rocket to the center of mass (meters). The vector is in the rocket frame.",
    )
    rocket_moment_of_inertia: Optional[float] = Field(
        default=None,
        description="Moment of inertia of the rocket (kg*m^2).",
    )
    rocket_thrust_application_offset: Optional[RocketVec2] = Field(
        default=None,
        description="Offset from the base of the rocket to the point where the thrust is applied (meters). The vector is in the rocket frame.",
    )

    # 3D Rocket properties (for 3D scenarios)
    rocket_center_of_mass_offset_3d: Optional[RocketVec3] = Field(
        default=None,
        description="3D offset from the base of the rocket to the center of mass (meters) in the rocket frame.",
    )
    rocket_inertia_diag_3d: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="3D diagonal inertia tensor components (Ix, Iy, Iz) in kg*m^2.",
    )
    rocket_thrust_application_offset_3d: Optional[RocketVec3] = Field(
        default=None,
        description="3D offset from the base to the thrust application point (meters) in the rocket frame.",
    )

    # World/environment configuration (can change via events!)
    world_gravity: Optional[WorldVec2] = Field(
        default=None,
        description="Gravity vector (m/s^2) in world frame.",
    )
    world_wind: Optional[WorldVec2] = Field(
        default=None,
        description="Wind velocity vector (m/s) in world frame.",
    )
    world_air_density: Optional[float] = Field(
        default=None,
        description="Air density (kg/m^3).",
    )

    # 3D Environment configuration
    world_gravity_3d: Optional[WorldVec3] = Field(
        default=None,
        description="Gravity vector (m/s^2) in 3D world frame.",
    )
    world_wind_3d: Optional[WorldVec3] = Field(
        default=None,
        description="Wind velocity vector (m/s) in 3D world frame.",
    )


    # # Sensor configuration
    # sensor_gps_noise_std: Optional[float] = Field(
    #     default=None,
    #     description="GPS noise standard deviation (meters).",
    # )
    # sensor_imu_orientation_noise_std: Optional[float] = Field(
    #     default=None,
    #     description="IMU orientation noise standard deviation (radians).",
    # )
    # sensor_imu_angular_velocity_noise_std: Optional[float] = Field(
    #     default=None,
    #     description="IMU angular velocity noise standard deviation (radians/second).",
    # )
    # sensor_thrust_noise_std: Optional[float] = Field(
    #     default=None,
    #     description="Thrust sensor noise standard deviation (Newtons).",
    # )
    # sensor_velocity_noise_std: Optional[float] = Field(
    #     default=None,
    #     description="Velocity sensor noise standard deviation (m/s).",
    # )

    # Simulation parameters
    # seed: Optional[int] = Field(default=None, description="Random seed")
    # controller_params: Optional[dict] = Field(default=None, description="Controller parameters")


class VariableMessage(BaseMessage):
    """Variable message - dynamic state and sensor data that changes over time."""

    message_type: Literal[MessageType.VARIABLE] = MessageType.VARIABLE

    timestamp: float = Field(
        default=0.0,
        description="Simulation time (seconds) from when - until otherwise - this message is valid.",
    )

    # True state (noumena) - position
    position: Optional[WorldVec2] = Field(
        default=None,
        description="True position (meters) in the world frame.",
    )
    velocity: Optional[WorldVec2] = Field(
        default=None,
        description="True velocity (m/s) in the world frame.",
    )

    # 3D true state (noumena) - position and velocity
    position_3d: Optional[WorldVec3] = Field(
        default=None,
        description="True 3D position (meters) in the world frame.",
    )
    velocity_3d: Optional[WorldVec3] = Field(
        default=None,
        description="True 3D velocity (m/s) in the world frame.",
    )

    # True state (noumena) - orientation
    orientation: Optional[float] = Field(
        default=None,
        description="True orientation (radians) in the world frame.",
    )
    angular_velocity: Optional[float] = Field(
        default=None,
        description="True angular velocity (radians/second) in the world frame.",
    )

    # 3D orientation and angular velocity (body frame)
    orientation_quat: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="True orientation as a quaternion (w, x, y, z) mapping body->world.",
    )
    angular_velocity_body_3d: Optional[RocketVec3] = Field(
        default=None,
        description="True 3D body-frame angular velocity (rad/s).",
    )

    # True state (noumena) - thrust
    thrust_vector: Optional[RocketVec2] = Field(
        default=None,
        description="True thrust vector (Newtons) in the rocket frame.",
    )
    gimbal_angle: Optional[float] = Field(
        default=None, description="True gimbal angle (radians)"
    )

    thrust_vector_body_3d: Optional[RocketVec3] = Field(
        default=None,
        description="True 3D thrust vector (Newtons) in the rocket body frame.",
    )

    # Sensor readings (phenomena) - GPS
    gps: Optional[WorldVec2] = Field(
        default=None,
        description="GPS reading (meters) in the world frame.",
    )

    gps_3d: Optional[WorldVec3] = Field(
        default=None,
        description="3D GPS reading (meters) in the world frame.",
    )

    # Sensor readings (phenomena) - thrust sensor
    thrust_sensor: Optional[float] = Field(
        default=None,
        description="Thrust sensor reading (Newtons) in the rocket frame.",
    )

    # Sensor readings (phenomena) - velocity sensor
    velocity_sensor: Optional[WorldVec2] = Field(
        default=None,
        description="Velocity sensor reading (m/s) in the world frame.",
    )

    velocity_sensor_3d: Optional[WorldVec3] = Field(
        default=None,
        description="3D velocity sensor reading (m/s) in the world frame.",
    )


    # Control commands
    # commanded_thrust: Optional[float] = Field(
    #     default=None,
    #     description="Commanded thrust magnitude (Newtons)",
    # )
    # commanded_gimbal: Optional[float] = Field(
    #     default=None,
    #     description="Commanded gimbal angle (radians)",
    # )

    # Rocket properties (if they change over time, e.g., fuel burn)
    # TODO: Add rocket properties that change over time, e.g., fuel burn
    # rocket_length: Optional[float] = Field(
    #     default=None,
    #     description="Length of the rocket (meters).",
    # )
    # rocket_mass: Optional[float] = Field(
    #     default=None,
    #     description="Mass of the rocket (kilograms).",
    # )
    # rocket_center_of_mass_offset: Optional[SerializableRocketVec2] = Field(
    #     default=None,
    #     description="Offset from the base of the rocket to the center of mass (meters). The vector is in the rocket frame.",
    # )
    # rocket_moment_of_inertia: Optional[float] = Field(
    #     default=None,
    #     description="Moment of inertia of the rocket (kg*m^2).",
    # )
    # rocket_thrust_application_offset: Optional[SerializableRocketVec2] = Field(
    #     default=None,
    #     description="Offset from the base of the rocket to the point where the thrust is applied (meters). The vector is in the rocket frame.",
    # )

    # Error metrics (for analysis)
    # TODO: Add error metrics
    # position_error: Optional[float] = Field(
    #     default=None,
    #     description="Position error (meters)",
    # )
    # orientation_error: Optional[float] = Field(
    #     default=None,
    #     description="Orientation error (radians)",
    # )


# ============================================================================
# Union Type
# ============================================================================

SimMessage = Union[ConstantMessage, VariableMessage, MetadataMessage]


# ============================================================================
# Serialization/Deserialization
# ============================================================================


def write_message(message: SimMessage) -> None:
    """
    Write a simulation message as a JSON line (NDJSON format) to stdout.
    Uses exclude_none=True to ensure no null values appear in the JSON.

    Args:
        message: Any SimMessage type.
    """
    try:
        json_str = message.model_dump_json(exclude_none=True)
        sys.stdout.write(json_str + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def read_message() -> SimMessage:
    """
    Read a single SimMessage from stdin (one JSON line, NDJSON format).

    Returns:
        SimMessage: The parsed message (automatically discriminated by message_type).

    Raises:
        EOFError: If end of input is reached.
        ValueError: If message type is unknown or invalid.
    """

    line = sys.stdin.readline()
    if not line:
        raise EOFError("End of input reached.")
    line = line.strip()
    while line == "":
        line = sys.stdin.readline()
        if not line:
            raise EOFError("End of input reached.")
        line = line.strip()

    data = json.loads(line)
    msg_type = data.get("message_type")

    if msg_type == MessageType.CONSTANT:
        return ConstantMessage(**data)
    elif msg_type == MessageType.VARIABLE:
        return VariableMessage(**data)
    elif msg_type == MessageType.METADATA:
        return MetadataMessage(**data)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
