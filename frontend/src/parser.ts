import type {
  FlightData,
  FlightFrame,
  MLDebugFrame,
  MLDebugMessage,
  SimMessage,
  VariableMessage,
} from './types'

function parseLine(line: string): SimMessage | null {
  try {
    return JSON.parse(line) as SimMessage
  } catch {
    return null
  }
}

/**
 * Parse an NDJSON string into FlightData.
 * Handles metadata, constant, variable, and debug messages.
 */
export function parseNDJSON(text: string): FlightData {
  let dt = 0.02
  let rocketLength = 6.0
  let wind: [number, number, number] | null = null
  const frames: FlightFrame[] = []
  const debugMsgs: FlightData['debugMsgs'] = []
  const mlDebugByTime = new Map<number, MLDebugFrame>()

  const lines = text.split('\n')
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) continue

    const msg = parseLine(line)
    if (!msg) continue

    switch (msg.message_type) {
      case 'metadata':
        if (msg.dt != null) dt = msg.dt
        break

      case 'constant':
        if (msg.rocket_length != null) rocketLength = msg.rocket_length
        if (msg.world_wind != null) wind = [msg.world_wind.x, msg.world_wind.y, msg.world_wind.z]
        break

      case 'debug':
        debugMsgs.push({ time: msg.timestamp, text: msg.text })
        break

      case 'ml_debug': {
        const m = msg as MLDebugMessage
        mlDebugByTime.set(m.timestamp, {
          rewardTotal: m.reward_total,
          rewardComponents: m.reward_components,
          actionThrottle: m.action_throttle,
          actionTiltX: m.action_tilt_x,
          actionTiltZ: m.action_tilt_z,
          distToTarget: m.dist_to_target,
          velToward: m.vel_toward,
          brakeMargin: m.brake_margin,
          tiltAngle: m.tilt_angle,
          omegaMag: m.omega_mag,
          target: m.target as [number, number, number],
        })
        break
      }

      case 'variable': {
        const v = msg as VariableMessage
        if (v.position == null || v.orientation_quat == null) break

        const pos = v.position
        const vel = v.velocity ?? { x: 0, y: 0, z: 0 }
        const quat = v.orientation_quat
        const thr = v.thrust_vector_body ?? { x: 0, y: 0, z: 0 }
        const ang = v.angular_velocity_body ?? { x: 0, y: 0, z: 0 }

        const gps = v.gps ? [v.gps.x, v.gps.y, v.gps.z] as [number, number, number] : null
        const velSensor = v.velocity_sensor
          ? [v.velocity_sensor.x, v.velocity_sensor.y, v.velocity_sensor.z] as [number, number, number]
          : null
        const orientationSensor = v.orientation_sensor
          ? [v.orientation_sensor[0], v.orientation_sensor[1], v.orientation_sensor[2], v.orientation_sensor[3]] as [number, number, number, number]
          : null
        const gyroSensor = v.gyro_sensor
          ? [v.gyro_sensor.x, v.gyro_sensor.y, v.gyro_sensor.z] as [number, number, number]
          : null

        frames.push({
          time: v.timestamp,
          position: [pos.x, pos.y, pos.z],
          velocity: [vel.x, vel.y, vel.z],
          quatWXYZ: [quat[0], quat[1], quat[2], quat[3]],
          thrustBody: [thr.x, thr.y, thr.z],
          angularVelBody: [ang.x, ang.y, ang.z],
          mass: v.rocket_mass ?? 0,
          fuelFraction: v.fuel_fraction ?? null,
          gps,
          thrustSensor: v.thrust_sensor ?? null,
          velocitySensor: velSensor,
          orientationSensor,
          gyroSensor,
          gimbalOuterAngle: v.gimbal_outer_angle ?? null,
          gimbalInnerAngle: v.gimbal_inner_angle ?? null,
          throttleCommand: v.throttle_command ?? null,
          onGround: v.on_ground ?? false,
          tippedOver: v.tipped_over ?? false,
          impactSpeed: v.impact_speed ?? null,
          impactVerticalSpeed: v.impact_vertical_speed ?? null,
          landingAngle: v.landing_angle ?? null,
          mlDebug: null,
        })
        break
      }
    }
  }

  // Attach ML debug data to matching frames by timestamp
  if (mlDebugByTime.size > 0) {
    for (const frame of frames) {
      const ml = mlDebugByTime.get(frame.time)
      if (ml) frame.mlDebug = ml
    }
  }

  return { dt, rocketLength, wind, frames, debugMsgs }
}

/**
 * Binary search: find the frame index for a given time.
 * Returns the index of the last frame with time <= t.
 */
export function frameIndexAt(frames: FlightFrame[], t: number): number {
  if (frames.length === 0) return -1
  if (t <= frames[0].time) return 0
  if (t >= frames[frames.length - 1].time) return frames.length - 1

  let lo = 0
  let hi = frames.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (frames[mid].time <= t) {
      lo = mid
    } else {
      hi = mid - 1
    }
  }
  return lo
}
