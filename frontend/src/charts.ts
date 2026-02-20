import type { Scenario, ChartDef, PolarChartDef } from './types'
import { CHART_DEFS, POLAR_CHART_DEFS } from './types'
import { frameIndexAt } from './parser'

// ============================================================================
// Colors — text recedes, geometry dominates
// ============================================================================

const BG = '#0f1629'
const GRID = '#151e33'
const TEXT_LABEL = '#3a4455'
const TEXT_AXIS = '#2e3a4d'
const TEXT_READOUT = '#6b7a90'
const CURSOR_COLOR = '#f59e0b'
const COMPONENT_STYLES: Array<[number, number]> = [
  [0, 0],   // solid
  [6, 4],   // dashed
  [2, 3],   // dotted
]

const MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace'

// ============================================================================
// Layout constants
// ============================================================================

const MARGIN = { top: 18, right: 52, bottom: 2, left: 40 }

// ============================================================================
// ChartManager
// ============================================================================

export class ChartManager {
  private canvases: Map<string, HTMLCanvasElement> = new Map()
  private polarCanvases: Map<string, HTMLCanvasElement> = new Map()
  private dpr: number

  constructor(container: HTMLElement) {
    this.dpr = window.devicePixelRatio || 1

    // Polar (bull's-eye) charts first — they're compact and distinctive
    for (const def of POLAR_CHART_DEFS) {
      const wrapper = document.createElement('div')
      wrapper.className = 'chart-slot chart-slot-polar'

      const canvas = document.createElement('canvas')
      canvas.className = 'chart-canvas'
      wrapper.appendChild(canvas)

      container.appendChild(wrapper)
      this.polarCanvases.set(def.id, canvas)
    }

    // Time-series charts
    for (const def of CHART_DEFS) {
      const wrapper = document.createElement('div')
      wrapper.className = 'chart-slot'

      const canvas = document.createElement('canvas')
      canvas.className = 'chart-canvas'
      wrapper.appendChild(canvas)

      container.appendChild(wrapper)
      this.canvases.set(def.id, canvas)
    }
  }

  draw(scenarios: Scenario[], currentTime: number, maxTime: number): void {
    for (const def of POLAR_CHART_DEFS) {
      const canvas = this.polarCanvases.get(def.id)!
      this.drawPolarChart(canvas, def, scenarios, currentTime)
    }
    for (const def of CHART_DEFS) {
      const canvas = this.canvases.get(def.id)!
      this.drawChart(canvas, def, scenarios, currentTime, maxTime)
    }
  }

  private drawChart(
    canvas: HTMLCanvasElement,
    def: ChartDef,
    scenarios: Scenario[],
    currentTime: number,
    maxTime: number,
  ): void {
    const rect = canvas.parentElement!.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    canvas.width = w * this.dpr
    canvas.height = h * this.dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'

    const ctx = canvas.getContext('2d')!
    ctx.scale(this.dpr, this.dpr)

    ctx.fillStyle = BG
    ctx.fillRect(0, 0, w, h)

    const plotX = MARGIN.left
    const plotY = MARGIN.top
    const plotW = w - MARGIN.left - MARGIN.right
    const plotH = h - MARGIN.top - MARGIN.bottom

    if (scenarios.length === 0 || maxTime <= 0) {
      return // silent when empty — no text clutter
    }

    // Y range across all scenarios
    let yMin = Infinity
    let yMax = -Infinity
    const isMulti = def.componentLabels != null

    for (const s of scenarios) {
      for (const frame of s.data.frames) {
        const val = def.extract(frame)
        if (typeof val === 'number') {
          yMin = Math.min(yMin, val)
          yMax = Math.max(yMax, val)
        } else {
          for (const v of val) {
            yMin = Math.min(yMin, v)
            yMax = Math.max(yMax, v)
          }
        }
      }
    }

    if (yMin === yMax) {
      yMin -= 1
      yMax += 1
    }
    const yPad = (yMax - yMin) * 0.08
    yMin -= yPad
    yMax += yPad

    const toX = (t: number) => plotX + (t / maxTime) * plotW
    const toY = (v: number) => plotY + ((yMax - v) / (yMax - yMin)) * plotH

    // Grid — very subtle, geometry background
    ctx.strokeStyle = GRID
    ctx.lineWidth = 1
    const yStep = niceStep(yMin, yMax, 4)
    const yStart = Math.ceil(yMin / yStep) * yStep

    ctx.font = `9px ${MONO}`
    ctx.fillStyle = TEXT_AXIS
    ctx.textAlign = 'right'

    for (let yv = yStart; yv <= yMax; yv += yStep) {
      const sy = toY(yv)
      if (sy < plotY || sy > plotY + plotH) continue
      ctx.beginPath()
      ctx.moveTo(plotX, sy)
      ctx.lineTo(plotX + plotW, sy)
      ctx.stroke()
      ctx.fillText(fmtCompact(yv), plotX - 3, sy + 3)
    }

    // Data lines — the primary visual
    for (const s of scenarios) {
      const { frames } = s.data

      if (isMulti) {
        const nComp = (def.extract(frames[0]) as number[]).length
        for (let ci = 0; ci < nComp; ci++) {
          ctx.strokeStyle = s.color
          ctx.lineWidth = 2
          ctx.globalAlpha = 1.0
          ctx.setLineDash(COMPONENT_STYLES[ci % COMPONENT_STYLES.length])
          ctx.beginPath()
          for (let i = 0; i < frames.length; i++) {
            const val = (def.extract(frames[i]) as number[])[ci]
            const sx = toX(frames[i].time)
            const sy = toY(val)
            if (i === 0) ctx.moveTo(sx, sy)
            else ctx.lineTo(sx, sy)
          }
          ctx.stroke()
          ctx.setLineDash([])
        }
      } else {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 2
        ctx.globalAlpha = 1.0
        ctx.beginPath()
        for (let i = 0; i < frames.length; i++) {
          const val = def.extract(frames[i]) as number
          const sx = toX(frames[i].time)
          const sy = toY(val)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      }
    }

    // Cursor line
    if (currentTime >= 0 && currentTime <= maxTime) {
      const cx = toX(currentTime)
      ctx.strokeStyle = CURSOR_COLOR
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.6
      ctx.beginPath()
      ctx.moveTo(cx, plotY)
      ctx.lineTo(cx, plotY + plotH)
      ctx.stroke()
      ctx.globalAlpha = 1.0

      // Value readout — small, subordinate
      for (let si = 0; si < scenarios.length; si++) {
        const s = scenarios[si]
        const idx = frameIndexAt(s.data.frames, currentTime)
        if (idx < 0) continue
        const frame = s.data.frames[idx]
        const val = def.extract(frame)

        ctx.font = `9px ${MONO}`
        ctx.fillStyle = TEXT_READOUT
        ctx.textAlign = 'left'
        const rx = plotX + plotW + 4
        const ry = plotY + 10 + si * 12

        if (typeof val === 'number') {
          ctx.fillText(fmtCompact(val), rx, ry)
        } else {
          ctx.fillText(val.map(fmtCompact).join(' '), rx, ry)
        }
      }
    }

    // Label — small, dim, subordinate to geometry
    ctx.font = `9px ${MONO}`
    ctx.fillStyle = TEXT_LABEL
    ctx.textAlign = 'left'
    ctx.fillText(`${def.label}  ${def.unit}`, plotX, plotY - 5)

    // Component legend — minimal
    if (isMulti && def.componentLabels) {
      const labels = def.componentLabels
      let lx = plotX + 60
      for (let ci = 0; ci < labels.length; ci++) {
        ctx.strokeStyle = TEXT_AXIS
        ctx.lineWidth = 1
        ctx.setLineDash(COMPONENT_STYLES[ci % COMPONENT_STYLES.length])
        ctx.beginPath()
        ctx.moveTo(lx, plotY - 8)
        ctx.lineTo(lx + 10, plotY - 8)
        ctx.stroke()
        ctx.setLineDash([])

        ctx.fillStyle = TEXT_AXIS
        ctx.font = `8px ${MONO}`
        ctx.fillText(labels[ci], lx + 13, plotY - 5)
        lx += 36
      }
    }
  }

  // ==========================================================================
  // Polar (bull's-eye) chart
  // ==========================================================================

  private drawPolarChart(
    canvas: HTMLCanvasElement,
    def: PolarChartDef,
    scenarios: Scenario[],
    currentTime: number,
  ): void {
    const rect = canvas.parentElement!.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    canvas.width = w * this.dpr
    canvas.height = h * this.dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'

    const ctx = canvas.getContext('2d')!
    ctx.scale(this.dpr, this.dpr)

    ctx.fillStyle = BG
    ctx.fillRect(0, 0, w, h)

    if (scenarios.length === 0) return

    // Square plot area centered in the rectangle
    const pad = { top: 18, bottom: 4, left: 4, right: 4 }
    const availW = w - pad.left - pad.right
    const availH = h - pad.top - pad.bottom
    const size = Math.min(availW, availH)
    const cx = pad.left + availW / 2
    const cy = pad.top + availH / 2
    const plotR = size / 2 - 2 // pixel radius of the outermost ring

    const scale = plotR / def.maxRadius

    // ── Concentric rings ──────────────────────────────────────────────────
    ctx.strokeStyle = GRID
    ctx.lineWidth = 1
    ctx.font = `8px ${MONO}`
    ctx.fillStyle = TEXT_AXIS
    ctx.textAlign = 'left'

    for (const deg of def.ringDegrees) {
      const r = Math.sin((deg * Math.PI) / 180) * scale
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.stroke()
      // Label at 1 o'clock position
      ctx.fillText(`${deg}°`, cx + r * 0.71 + 2, cy - r * 0.71 - 1)
    }

    // ── Crosshairs ────────────────────────────────────────────────────────
    ctx.strokeStyle = GRID
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(cx - plotR, cy)
    ctx.lineTo(cx + plotR, cy)
    ctx.moveTo(cx, cy - plotR)
    ctx.lineTo(cx, cy + plotR)
    ctx.stroke()

    // Center dot
    ctx.fillStyle = TEXT_AXIS
    ctx.beginPath()
    ctx.arc(cx, cy, 1.5, 0, Math.PI * 2)
    ctx.fill()

    // ── Data: trail + current point per scenario ──────────────────────────
    const TRAIL_LENGTH = 200

    for (const s of scenarios) {
      const { frames } = s.data
      const idx = frameIndexAt(frames, currentTime)
      if (idx < 0) continue

      // Draw trail (faint history)
      const trailStart = Math.max(0, idx - TRAIL_LENGTH)
      if (trailStart < idx) {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 1
        ctx.globalAlpha = 0.15
        ctx.beginPath()
        let started = false
        for (let i = trailStart; i <= idx; i++) {
          const val = def.extract(frames[i])
          if (!val) {
            started = false
            continue
          }
          const px = cx + val[0] * scale
          const py = cy - val[1] * scale // Y inverted: positive = up
          if (!started) {
            ctx.moveTo(px, py)
            started = true
          } else {
            ctx.lineTo(px, py)
          }
        }
        ctx.stroke()
        ctx.globalAlpha = 1.0
      }

      // Current point
      const val = def.extract(frames[idx])
      if (!val) continue
      const px = cx + val[0] * scale
      const py = cy - val[1] * scale

      // Halo
      ctx.fillStyle = s.color
      ctx.globalAlpha = 0.2
      ctx.beginPath()
      ctx.arc(px, py, 5, 0, Math.PI * 2)
      ctx.fill()

      // Dot
      ctx.globalAlpha = 1.0
      ctx.beginPath()
      ctx.arc(px, py, 2.5, 0, Math.PI * 2)
      ctx.fill()
    }

    // ── Value readout — top right, one per scenario ────────────────────────
    ctx.font = `9px ${MONO}`
    ctx.textAlign = 'right'
    let ri = 0
    for (const s of scenarios) {
      const { frames } = s.data
      const idx = frameIndexAt(frames, currentTime)
      if (idx < 0) continue
      const val = def.extract(frames[idx])
      if (!val) continue
      const deg = (Math.asin(Math.min(1, Math.hypot(val[0], val[1]))) * 180) / Math.PI
      ctx.fillStyle = TEXT_READOUT
      ctx.globalAlpha = 1.0
      ctx.fillText(`${deg.toFixed(1)}°`, w - pad.right, pad.top + 4 + ri * 12)
      ri++
    }

    // ── Label ─────────────────────────────────────────────────────────────
    ctx.font = `9px ${MONO}`
    ctx.fillStyle = TEXT_LABEL
    ctx.textAlign = 'left'
    ctx.globalAlpha = 1.0
    ctx.fillText(def.label, pad.left + 2, pad.top - 5)
  }
}

// ============================================================================
// Helpers (exported for reuse in interactive mode)
// ============================================================================

export function niceStep(min: number, max: number, target: number): number {
  const rough = (max - min) / target
  const pow = Math.pow(10, Math.floor(Math.log10(rough)))
  const norm = rough / pow
  let step: number
  if (norm < 1.5) step = 1
  else if (norm < 3.5) step = 2
  else if (norm < 7.5) step = 5
  else step = 10
  return step * pow
}

export function fmtCompact(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(2)
  return v.toFixed(3)
}
