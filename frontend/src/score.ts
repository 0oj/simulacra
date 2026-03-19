/**
 * Flight quality scoring — computes smoothness, efficiency, and control metrics
 * from NDJSON frame data. All metrics are derived post-hoc from recorded flights.
 */

import type { FlightFrame } from './types'

export interface FlightScore {
  rmsJerk: number           // m/s³ — rate of change of acceleration
  rmsOmega: number          // rad/s — average rotational rate
  peakG: number             // G — maximum acceleration magnitude
  fuelUsed: number          // % — fuel consumed
  gimbalSaturation: number  // % — time at actuator limits
  rmsThrottleRate: number   // N/s — throttle rate of change
  overall: number           // 0-100 composite score
}

const MAX_GIMBAL = 0.26   // radians (from HopperConfig)
const SAT_THRESHOLD = 0.95

/**
 * Compute flight quality metrics from a sequence of frames.
 * Requires at least 3 frames for jerk computation.
 */
export function computeFlightScore(frames: FlightFrame[], _dt: number): FlightScore {
  if (frames.length < 3) {
    return { rmsJerk: 0, rmsOmega: 0, peakG: 0, fuelUsed: 0, gimbalSaturation: 0, rmsThrottleRate: 0, overall: 0 }
  }

  const n = frames.length
  let sumJerkSq = 0
  let sumOmegaSq = 0
  let maxAccel = 0
  let sumDThrustSq = 0
  let satCount = 0
  let jerkCount = 0

  // Compute accelerations first (finite difference on velocity)
  const accels: [number, number, number][] = []
  for (let i = 1; i < n; i++) {
    const f0 = frames[i - 1], f1 = frames[i]
    const adt = f1.time - f0.time
    if (adt < 1e-6) { accels.push([0, 0, 0]); continue }
    accels.push([
      (f1.velocity[0] - f0.velocity[0]) / adt,
      (f1.velocity[1] - f0.velocity[1]) / adt,
      (f1.velocity[2] - f0.velocity[2]) / adt,
    ])
  }

  // Jerk: finite difference on acceleration
  for (let i = 1; i < accels.length; i++) {
    const a0 = accels[i - 1], a1 = accels[i]
    const jdt = frames[i + 1].time - frames[i].time
    if (jdt < 1e-6) continue
    const jx = (a1[0] - a0[0]) / jdt
    const jy = (a1[1] - a0[1]) / jdt
    const jz = (a1[2] - a0[2]) / jdt
    sumJerkSq += jx * jx + jy * jy + jz * jz
    jerkCount++
  }

  // Per-frame metrics
  for (let i = 0; i < n; i++) {
    const f = frames[i]

    // Angular velocity magnitude
    const [wx, wy, wz] = f.angularVelBody
    sumOmegaSq += wx * wx + wy * wy + wz * wz

    // Peak acceleration (G)
    if (i < accels.length) {
      const [ax, ay, az] = accels[i]
      const amag = Math.sqrt(ax * ax + ay * ay + az * az)
      if (amag > maxAccel) maxAccel = amag
    }

    // Gimbal saturation
    const gimbalMag = Math.sqrt(
      (f.gimbalOuterAngle ?? 0) ** 2 + (f.gimbalInnerAngle ?? 0) ** 2
    )
    if (gimbalMag >= SAT_THRESHOLD * MAX_GIMBAL) satCount++

    // Throttle rate
    if (i > 0) {
      const f0 = frames[i - 1]
      const thr0 = Math.sqrt(f0.thrustBody[0] ** 2 + f0.thrustBody[1] ** 2 + f0.thrustBody[2] ** 2)
      const thr1 = Math.sqrt(f.thrustBody[0] ** 2 + f.thrustBody[1] ** 2 + f.thrustBody[2] ** 2)
      const tdt = f.time - f0.time
      if (tdt > 1e-6) {
        const dThrust = (thr1 - thr0) / tdt
        sumDThrustSq += dThrust * dThrust
      }
    }
  }

  const rmsJerk = Math.sqrt(sumJerkSq / Math.max(1, jerkCount))
  const rmsOmega = Math.sqrt(sumOmegaSq / n)
  const peakG = maxAccel / 9.81
  const fuelUsed = frames[0].fuelFraction != null && frames[n - 1].fuelFraction != null
    ? (frames[0].fuelFraction! - frames[n - 1].fuelFraction!) * 100
    : 0
  const gimbalSaturation = (satCount / n) * 100
  const rmsThrottleRate = Math.sqrt(sumDThrustSq / Math.max(1, n - 1))

  // Composite score [0-100]: lower metrics = better score
  // Each component maps to 0-20 points via sigmoid-like scaling
  const overall = (
    scoreComponent(rmsJerk, 3, 15) * 25 +          // jerk: 3 m/s³ = perfect, 15 = bad
    scoreComponent(rmsOmega, 0.15, 0.5) * 25 +     // omega: 0.15 = perfect, 0.5 = bad
    scoreComponent(peakG, 2, 8) * 15 +              // G-force: 2G = perfect, 8G = bad
    scoreComponent(gimbalSaturation, 2, 15) * 15 +  // sat: 2% = perfect, 15% = bad
    scoreComponent(rmsThrottleRate, 50, 200) * 10 +  // throttle: 50 = perfect, 200 = bad
    scoreComponent(fuelUsed, 10, 50) * 10            // fuel: 10% = perfect, 50% = bad
  )

  return { rmsJerk, rmsOmega, peakG, fuelUsed, gimbalSaturation, rmsThrottleRate, overall }
}

/** Maps a metric value to [0, 1] where 0 = bad threshold, 1 = good threshold. */
function scoreComponent(value: number, good: number, bad: number): number {
  if (value <= good) return 1.0
  if (value >= bad) return 0.0
  return 1.0 - (value - good) / (bad - good)
}

/** Format a score for display. */
export function formatScore(score: FlightScore): { label: string; value: string; grade: string }[] {
  return [
    {
      label: 'JERK',
      value: score.rmsJerk.toFixed(1) + ' m/s\u00b3',
      grade: gradeColor(score.rmsJerk, 3, 8, 15),
    },
    {
      label: 'ANGULAR RATE',
      value: score.rmsOmega.toFixed(3) + ' rad/s',
      grade: gradeColor(score.rmsOmega, 0.15, 0.3, 0.5),
    },
    {
      label: 'PEAK G-FORCE',
      value: score.peakG.toFixed(1) + ' G',
      grade: gradeColor(score.peakG, 2, 5, 8),
    },
    {
      label: 'FUEL USED',
      value: score.fuelUsed.toFixed(1) + '%',
      grade: gradeColor(score.fuelUsed, 15, 30, 50),
    },
    {
      label: 'GIMBAL SAT',
      value: score.gimbalSaturation.toFixed(1) + '%',
      grade: gradeColor(score.gimbalSaturation, 2, 8, 15),
    },
    {
      label: 'THROTTLE RATE',
      value: score.rmsThrottleRate.toFixed(0) + ' N/s',
      grade: gradeColor(score.rmsThrottleRate, 50, 100, 200),
    },
  ]
}

function gradeColor(value: number, good: number, mid: number, _bad: number): string {
  if (value <= good) return 'good'
  if (value <= mid) return 'ok'
  return 'bad'
}
