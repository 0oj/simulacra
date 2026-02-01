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
  on_ground?: boolean
  tipped_over?: boolean
  impact_speed?: number
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
  fuelFraction: number
}

export interface FlightData {
  dt: number
  rocketLength: number
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
