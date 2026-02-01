"""
The record of truth. God writes the simulation state to stdout,
capturing both the true state and what the rocket perceived.
"""

import json
import sys
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional, Union, Literal, Tuple

from .physics.vec3 import RocketVec3, WorldVec3

from pydantic import BaseModel, Field

from enum import Enum


# ============================================================================
# Source Annotations
# ============================================================================


@dataclass(frozen=True)
class Source:
    """Metadata marker declaring where a field's value comes from at runtime."""
    kind: str       # "state", "sensors", "props", "env"
    attr: str       # attribute name on the root object
    transform: str | None = None


# Convenience type aliases for annotated fields
def State(attr: str, transform: str | None = None):
    return Source("state", attr, transform)

def Sensor(attr: str):
    return Source("sensors", attr)

def Props(attr: str):
    return Source("props", attr)

def Env(attr: str):
    return Source("env", attr)


TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "orientation_to_quat": lambda o: tuple(o.as_array()),
}

SCHEMA_VERSION = "0.2.0"


class MessageType(str, Enum):
    METADATA = "metadata"
    CONSTANT = "constant"
    VARIABLE = "variable"
    DEBUG = "debug"


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
    rocket_length: Annotated[Optional[float], Props("length")] = Field(
        default=None, description="Length of the rocket (meters)",
    )
    rocket_mass: Annotated[Optional[float], Props("mass")] = Field(
        default=None, description="Mass of the rocket (kilograms)",
    )
    rocket_center_of_mass_offset: Annotated[Optional[RocketVec3], Props("center_of_mass_offset")] = Field(
        default=None,
        description="Offset from the base of the rocket to the center of mass (meters) in the rocket frame.",
    )
    rocket_inertia_diag: Annotated[Optional[Tuple[float, float, float]], Props("inertia_diag")] = Field(
        default=None,
        description="Diagonal inertia tensor components (Ix, Iy, Iz) in kg*m^2.",
    )
    rocket_thrust_application_offset: Annotated[Optional[RocketVec3], Props("thrust_application_offset")] = Field(
        default=None,
        description="Offset from the base to the thrust application point (meters) in the rocket frame.",
    )

    # World/environment configuration (can change via events!)
    world_gravity: Annotated[Optional[WorldVec3], Env("gravity")] = Field(
        default=None,
        description="Gravity vector (m/s^2) in world frame.",
    )
    world_wind: Annotated[Optional[WorldVec3], Env("wind")] = Field(
        default=None,
        description="Wind velocity vector (m/s) in world frame.",
    )
    world_air_density: Annotated[Optional[float], Env("air_density")] = Field(
        default=None,
        description="Air density (kg/m^3).",
    )


class VariableMessage(BaseMessage):
    """Variable message - dynamic state and sensor data that changes over time."""

    message_type: Literal[MessageType.VARIABLE] = MessageType.VARIABLE

    timestamp: float = Field(
        default=0.0,
        description="Simulation time (seconds) from when - until otherwise - this message is valid.",
    )

    # True state (noumena) - position and velocity
    position: Annotated[Optional[WorldVec3], State("position")] = Field(
        default=None,
        description="True position (meters) in the world frame.",
    )
    velocity: Annotated[Optional[WorldVec3], State("velocity")] = Field(
        default=None,
        description="True velocity (m/s) in the world frame.",
    )

    # True state (noumena) - orientation and angular velocity
    orientation_quat: Annotated[Optional[Tuple[float, float, float, float]], State("orientation", transform="orientation_to_quat")] = Field(
        default=None,
        description="True orientation as a quaternion (w, x, y, z) mapping body->world.",
    )
    angular_velocity_body: Annotated[Optional[RocketVec3], State("angular_velocity_body")] = Field(
        default=None,
        description="True body-frame angular velocity (rad/s).",
    )

    # True state (noumena) - thrust
    thrust_vector_body: Annotated[Optional[RocketVec3], State("thrust_body")] = Field(
        default=None,
        description="True thrust vector (Newtons) in the rocket body frame.",
    )

    # Sensor readings (phenomena) - GPS
    gps: Annotated[Optional[WorldVec3], Sensor("GPS")] = Field(
        default=None,
        description="GPS reading (meters) in the world frame.",
    )

    # Sensor readings (phenomena) - thrust sensor
    thrust_sensor: Annotated[Optional[float], Sensor("ThrustSensor")] = Field(
        default=None,
        description="Thrust sensor reading (Newtons) in the rocket frame.",
    )

    # Sensor readings (phenomena) - velocity sensor
    velocity_sensor: Annotated[Optional[WorldVec3], Sensor("VelocitySensor")] = Field(
        default=None,
        description="Velocity sensor reading (m/s) in the world frame.",
    )


class DebugMessage(BaseMessage):
    """Free-form debug/annotation message tied to a simulation time."""

    message_type: Literal[MessageType.DEBUG] = MessageType.DEBUG

    timestamp: float = Field(
        description="Simulation time (seconds) this message refers to."
    )
    text: str = Field(
        description="Arbitrary debug or annotation string."
    )


# ============================================================================
# Auto-Collection
# ============================================================================


def _get_source(field_info) -> Source | None:
    """Extract the Source marker from a field's Annotated metadata, if present."""
    for item in field_info.metadata:
        if isinstance(item, Source):
            return item
    return None


def collect(message_cls: type[BaseMessage], timestamp: float, **roots: Any) -> BaseMessage:
    """
    Build a message by reading source annotations from the model's fields.

    roots: named objects to resolve against (state=..., sensors=..., props=..., env=...)
    """
    data: dict[str, Any] = {"timestamp": timestamp}
    for field_name, field_info in message_cls.model_fields.items():
        source = _get_source(field_info)
        if source is None:
            continue
        root_obj = roots[source.kind]
        value = getattr(root_obj, source.attr)
        if source.kind == "sensors":
            value = value.read()
        if source.transform:
            value = TRANSFORMS[source.transform](value)
        data[field_name] = value
    return message_cls(**data)


# ============================================================================
# Union Type
# ============================================================================

SimMessage = Union[ConstantMessage, VariableMessage, MetadataMessage, DebugMessage]


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


def read_message(stream=None) -> SimMessage:
    """
    Read a single SimMessage from a stream (one JSON line, NDJSON format).

    Args:
        stream: Readable text stream. Defaults to sys.stdin.

    Returns:
        SimMessage: The parsed message (automatically discriminated by message_type).

    Raises:
        EOFError: If end of input is reached.
        ValueError: If message type is unknown or invalid.
    """
    if stream is None:
        stream = sys.stdin

    line = stream.readline()
    if not line:
        raise EOFError("End of input reached.")
    line = line.strip()
    while line == "":
        line = stream.readline()
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
    elif msg_type == MessageType.DEBUG:
        return DebugMessage(**data)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
