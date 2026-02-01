import type {
  FlightData,
  FlightFrame,
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
  const frames: FlightFrame[] = []
  const debugMsgs: FlightData['debugMsgs'] = []

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
        break

      case 'debug':
        debugMsgs.push({ time: msg.timestamp, text: msg.text })
        break

      case 'variable': {
        const v = msg as VariableMessage
        if (v.position == null || v.orientation_quat == null) break

        const pos = v.position
        const vel = v.velocity ?? { x: 0, y: 0, z: 0 }
        const quat = v.orientation_quat
        const thr = v.thrust_vector_body ?? { x: 0, y: 0, z: 0 }
        const ang = v.angular_velocity_body ?? { x: 0, y: 0, z: 0 }

        frames.push({
          time: v.timestamp,
          position: [pos.x, pos.y, pos.z],
          velocity: [vel.x, vel.y, vel.z],
          quatWXYZ: [quat[0], quat[1], quat[2], quat[3]],
          thrustBody: [thr.x, thr.y, thr.z],
          angularVelBody: [ang.x, ang.y, ang.z],
          mass: v.rocket_mass ?? 0,
          fuelFraction: v.fuel_fraction ?? 0,
        })
        break
      }
    }
  }

  return { dt, rocketLength, frames, debugMsgs }
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
