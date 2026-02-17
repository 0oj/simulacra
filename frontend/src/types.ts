// ============================================================================
// NDJSON message types — mirrors backend/src/simulacra/schema.py
// ============================================================================

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface MetadataMessage {
  message_type: 'metadata'
  schema_version: string
  dt?: number
  duration?: number
  scenario_name?: string
}

export interface ConstantMessage {
  message_type: 'constant'
  timestamp: number
  rocket_length?: number
  rocket_mass?: number
  rocket_center_of_mass_offset?: Vec3
  rocket_inertia_diag?: [number, number, number]
  rocket_thrust_application_offset?: Vec3
  world_gravity?: Vec3
  world_wind?: Vec3
}

export interface VariableMessage {
  message_type: 'variable'
  timestamp: number
  position?: Vec3
  velocity?: Vec3
  orientation_quat?: [number, number, number, number] // w, x, y, z
  angular_velocity_body?: Vec3
  thrust_vector_body?: Vec3
  rocket_mass?: number
  fuel_fraction?: number
  gps?: Vec3
  thrust_sensor?: number
  velocity_sensor?: Vec3
  // IMU sensor readings
  orientation_sensor?: [number, number, number, number] // w, x, y, z
  gyro_sensor?: Vec3
  // Actuator state
  gimbal_outer_angle?: number
  gimbal_inner_angle?: number
  throttle_command?: number
  // Ground contact
  on_ground?: boolean
  tipped_over?: boolean
  impact_speed?: number
  impact_vertical_speed?: number
  landing_angle?: number
}

export interface DebugMessage {
  message_type: 'debug'
  timestamp: number
  text: string
}

export type SimMessage =
  | MetadataMessage
  | ConstantMessage
  | VariableMessage
  | DebugMessage

// ============================================================================
// Parsed flight data
// ============================================================================

export interface FlightFrame {
  time: number
  position: [number, number, number]
  velocity: [number, number, number]
  quatWXYZ: [number, number, number, number]
  thrustBody: [number, number, number]
  angularVelBody: [number, number, number]
  mass: number
  fuelFraction: number | null

  // Sensor readings (phenomena)
  gps: [number, number, number] | null
  thrustSensor: number | null
  velocitySensor: [number, number, number] | null
  orientationSensor: [number, number, number, number] | null
  gyroSensor: [number, number, number] | null

  // Actuator state (phenomena)
  gimbalOuterAngle: number | null
  gimbalInnerAngle: number | null
  throttleCommand: number | null

  // Ground contact
  onGround: boolean
  tippedOver: boolean
  impactSpeed: number | null
  impactVerticalSpeed: number | null
  landingAngle: number | null
}

export interface FlightData {
  dt: number
  rocketLength: number
  wind: [number, number, number] | null
  frames: FlightFrame[]
  debugMsgs: Array<{ time: number; text: string }>
}

export interface Scenario {
  label: string
  color: string
  data: FlightData
}

// ============================================================================
// Chart panel definition
// ============================================================================

export interface ChartDef {
  id: string
  label: string
  unit: string
  extract: (frame: FlightFrame) => number | number[]
  componentLabels?: string[]
}

export const CHART_DEFS: ChartDef[] = [
  {
    id: 'alt',
    label: 'ALT',
    unit: 'm',
    extract: (f) => f.position[1],
  },
  {
    id: 'spd',
    label: 'V',
    unit: 'm/s',
    extract: (f) => Math.hypot(f.velocity[0], f.velocity[1], f.velocity[2]),
  },
  {
    id: 'thr',
    label: 'THR',
    unit: 'N',
    extract: (f) =>
      Math.hypot(f.thrustBody[0], f.thrustBody[1], f.thrustBody[2]),
  },
  {
    id: 'mass',
    label: 'M',
    unit: 'kg',
    extract: (f) => f.mass,
  },
  {
    id: 'tilt',
    label: 'TILT',
    unit: '°',
    extract: (f) => {
      // Nose is body +X. Rotate by quaternion to get world direction.
      // cos(tilt) = nose_world · [0,1,0]
      const [w, x, y, z] = f.quatWXYZ
      // Rotate [1,0,0] by quaternion (w,x,y,z):
      // result.y = 2*(x*y + w*z) ... wait, let me do the full rotation.
      // v' = q * v * q^-1. For v = [1,0,0]:
      const ny = 2 * (x * y + w * z)
      const cosAngle = Math.max(-1, Math.min(1, ny))
      return (Math.acos(cosAngle) * 180) / Math.PI
    },
  },
  {
    id: 'omega',
    label: 'ω',
    unit: 'rad/s',
    extract: (f) => [f.angularVelBody[0], f.angularVelBody[1], f.angularVelBody[2]],
    componentLabels: ['ωx', 'ωy', 'ωz'],
  },
  {
    id: 'fuel',
    label: 'FUEL',
    unit: '%',
    extract: (f) => (f.fuelFraction != null ? f.fuelFraction * 100 : NaN),
  },
  {
    id: 'gps-err',
    label: 'Δ GPS',
    unit: 'm',
    extract: (f) => {
      if (!f.gps) return 0
      const dx = f.gps[0] - f.position[0]
      const dy = f.gps[1] - f.position[1]
      const dz = f.gps[2] - f.position[2]
      return Math.hypot(dx, dy, dz)
    },
  },
  {
    id: 'thr-err',
    label: 'Δ THR',
    unit: 'N',
    extract: (f) => {
      if (f.thrustSensor == null) return 0
      const truthMag = Math.hypot(f.thrustBody[0], f.thrustBody[1], f.thrustBody[2])
      return f.thrustSensor - truthMag
    },
  },
  {
    id: 'ctrl-thr',
    label: 'CTRL',
    unit: 'N',
    extract: (f) => {
      const actual = Math.hypot(f.thrustBody[0], f.thrustBody[1], f.thrustBody[2])
      const cmd = f.throttleCommand ?? actual
      return [cmd, actual]
    },
    componentLabels: ['CMD', 'ACT'],
  },
  {
    id: 'omega-mag',
    label: '|ω|',
    unit: 'rad/s',
    extract: (f) =>
      Math.hypot(f.angularVelBody[0], f.angularVelBody[1], f.angularVelBody[2]),
  },
  {
    id: 'gimbal',
    label: 'GIMBAL',
    unit: '°',
    extract: (f) => {
      const outer = f.gimbalOuterAngle != null ? (f.gimbalOuterAngle * 180) / Math.PI : 0
      const inner = f.gimbalInnerAngle != null ? (f.gimbalInnerAngle * 180) / Math.PI : 0
      return [outer, inner]
    },
    componentLabels: ['YAW', 'PITCH'],
  },
]

// ============================================================================
// Scenario colors
// ============================================================================

// ============================================================================
// Polar (bull's-eye) chart definitions
// ============================================================================

export interface PolarChartDef {
  id: string
  label: string
  /** Extract (x, y) for the polar plot. Center = (0,0). Returns null if no data. */
  extract: (frame: FlightFrame) => [number, number] | null
  /** Maximum meaningful radius (determines scale). */
  maxRadius: number
  /** Degree labels for the concentric rings. */
  ringDegrees: number[]
}

export const POLAR_CHART_DEFS: PolarChartDef[] = [
  {
    id: 'attitude',
    label: 'ATTITUDE',
    extract: (f) => {
      // Project nose vector (body +X rotated by quaternion) onto world XZ plane.
      // Center = perfectly vertical (nose along +Y).
      // Distance from center = sin(tilt angle).
      const [w, x, y, z] = f.quatWXYZ
      // Rotate [1,0,0] by quaternion q = (w,x,y,z):
      const nx = 1 - 2 * (y * y + z * z)
      const nz = 2 * (x * z - w * y)
      return [nx, nz]
    },
    maxRadius: 0.5, // ~30°
    ringDegrees: [10, 20, 30],
  },
  {
    id: 'gimbal',
    label: 'GIMBAL',
    extract: (f) => {
      // Thrust direction in body frame, projected onto YZ plane.
      // Center = thrust along body +X (no gimbal deflection).
      const mag = Math.hypot(f.thrustBody[0], f.thrustBody[1], f.thrustBody[2])
      if (mag < 1) return null // no thrust
      // Body Z component = yaw deflection, Body Y = pitch deflection
      return [f.thrustBody[2] / mag, f.thrustBody[1] / mag]
    },
    maxRadius: 0.3, // ~17°
    ringDegrees: [5, 10, 15],
  },
]

// ============================================================================
// Scenario colors
// ============================================================================

export const SCENARIO_COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#22c55e', // green
  '#f59e0b', // amber
  '#a855f7', // purple
  '#ec4899', // pink
]
