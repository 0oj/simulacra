import type { Scenario, ChartDef } from './types'
import { CHART_DEFS } from './types'
import { frameIndexAt } from './parser'

// ============================================================================
// Colors
// ============================================================================

const BG = '#0f1629'
const GRID = '#1a2744'
const TEXT_PRIMARY = '#e8eaed'
const TEXT_DIM = '#5a6577'
const CURSOR_COLOR = '#f59e0b'
const COMPONENT_STYLES: Array<[number, number]> = [
  [0, 0],     // solid
  [6, 4],     // dashed
  [2, 3],     // dotted
]

// ============================================================================
// Layout constants (per chart canvas)
// ============================================================================

const MARGIN = { top: 22, right: 60, bottom: 2, left: 48 }

// ============================================================================
// ChartManager
// ============================================================================

export class ChartManager {
  private canvases: Map<string, HTMLCanvasElement> = new Map()
  private dpr: number

  constructor(container: HTMLElement) {
    this.dpr = window.devicePixelRatio || 1

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
    // Size canvas to container
    const rect = canvas.parentElement!.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    canvas.width = w * this.dpr
    canvas.height = h * this.dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'

    const ctx = canvas.getContext('2d')!
    ctx.scale(this.dpr, this.dpr)

    // Clear
    ctx.fillStyle = BG
    ctx.fillRect(0, 0, w, h)

    const plotX = MARGIN.left
    const plotY = MARGIN.top
    const plotW = w - MARGIN.left - MARGIN.right
    const plotH = h - MARGIN.top - MARGIN.bottom

    if (scenarios.length === 0 || maxTime <= 0) {
      ctx.font = '11px ui-monospace, SFMono-Regular, Consolas, monospace'
      ctx.fillStyle = TEXT_DIM
      ctx.textAlign = 'center'
      ctx.fillText('NO DATA', w / 2, h / 2)
      return
    }

    // Compute Y range across all scenarios
    let yMin = Infinity
    let yMax = -Infinity
    const isMultiComponent = def.componentLabels != null

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

    // Pad Y range
    if (yMin === yMax) {
      yMin -= 1
      yMax += 1
    }
    const yPad = (yMax - yMin) * 0.08
    yMin -= yPad
    yMax += yPad

    const xMin = 0
    const xMax = maxTime

    const toScreenX = (t: number) => plotX + ((t - xMin) / (xMax - xMin)) * plotW
    const toScreenY = (v: number) => plotY + ((yMax - v) / (yMax - yMin)) * plotH

    // Grid lines
    ctx.strokeStyle = GRID
    ctx.lineWidth = 1

    // Horizontal grid (4 lines)
    const yStep = niceStep(yMin, yMax, 4)
    const yStart = Math.ceil(yMin / yStep) * yStep
    ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'
    ctx.fillStyle = TEXT_DIM
    ctx.textAlign = 'right'

    for (let yv = yStart; yv <= yMax; yv += yStep) {
      const sy = toScreenY(yv)
      if (sy < plotY || sy > plotY + plotH) continue
      ctx.beginPath()
      ctx.moveTo(plotX, sy)
      ctx.lineTo(plotX + plotW, sy)
      ctx.stroke()
      ctx.fillText(formatValue(yv), plotX - 4, sy + 3)
    }

    // Data lines
    for (let si = 0; si < scenarios.length; si++) {
      const s = scenarios[si]
      const { frames } = s.data

      if (isMultiComponent) {
        const componentCount = (def.extract(frames[0]) as number[]).length
        for (let ci = 0; ci < componentCount; ci++) {
          ctx.strokeStyle = s.color
          ctx.lineWidth = 1.5
          ctx.globalAlpha = 1.0
          ctx.setLineDash(COMPONENT_STYLES[ci % COMPONENT_STYLES.length])
          ctx.beginPath()
          for (let i = 0; i < frames.length; i++) {
            const val = (def.extract(frames[i]) as number[])[ci]
            const sx = toScreenX(frames[i].time)
            const sy = toScreenY(val)
            if (i === 0) ctx.moveTo(sx, sy)
            else ctx.lineTo(sx, sy)
          }
          ctx.stroke()
          ctx.setLineDash([])
        }
      } else {
        ctx.strokeStyle = s.color
        ctx.lineWidth = 1.5
        ctx.globalAlpha = 1.0
        ctx.beginPath()
        for (let i = 0; i < frames.length; i++) {
          const val = def.extract(frames[i]) as number
          const sx = toScreenX(frames[i].time)
          const sy = toScreenY(val)
          if (i === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      }
    }

    // Cursor line
    if (currentTime >= 0 && currentTime <= maxTime) {
      const cx = toScreenX(currentTime)
      ctx.strokeStyle = CURSOR_COLOR
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.8
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(cx, plotY)
      ctx.lineTo(cx, plotY + plotH)
      ctx.stroke()
      ctx.globalAlpha = 1.0

      // Current value readout
      for (let si = 0; si < scenarios.length; si++) {
        const s = scenarios[si]
        const idx = frameIndexAt(s.data.frames, currentTime)
        if (idx < 0) continue
        const frame = s.data.frames[idx]
        const val = def.extract(frame)

        ctx.font = '11px ui-monospace, SFMono-Regular, Consolas, monospace'
        ctx.fillStyle = s.color
        ctx.textAlign = 'left'
        const readoutX = plotX + plotW + 6
        const readoutY = plotY + 12 + si * 14

        if (typeof val === 'number') {
          ctx.fillText(formatValue(val), readoutX, readoutY)
        } else {
          ctx.fillText(val.map(formatValue).join(' '), readoutX, readoutY)
        }
      }
    }

    // Label
    ctx.font = 'bold 10px ui-monospace, SFMono-Regular, Consolas, monospace'
    ctx.fillStyle = TEXT_PRIMARY
    ctx.textAlign = 'left'
    ctx.fillText(def.label, plotX, plotY - 6)

    ctx.font = '10px ui-monospace, SFMono-Regular, Consolas, monospace'
    ctx.fillStyle = TEXT_DIM
    ctx.fillText(def.unit, plotX + ctx.measureText(def.label + ' ').width + 4, plotY - 6)

    // Component legend for multi-component charts
    if (isMultiComponent && def.componentLabels) {
      const labels = def.componentLabels
      let lx = plotX + 60
      for (let ci = 0; ci < labels.length; ci++) {
        ctx.strokeStyle = TEXT_DIM
        ctx.lineWidth = 1.5
        ctx.setLineDash(COMPONENT_STYLES[ci % COMPONENT_STYLES.length])
        ctx.beginPath()
        ctx.moveTo(lx, plotY - 10)
        ctx.lineTo(lx + 14, plotY - 10)
        ctx.stroke()
        ctx.setLineDash([])

        ctx.fillStyle = TEXT_DIM
        ctx.font = '9px ui-monospace, SFMono-Regular, Consolas, monospace'
        ctx.fillText(labels[ci], lx + 17, plotY - 6)
        lx += 40
      }
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

function niceStep(min: number, max: number, targetLines: number): number {
  const rough = (max - min) / targetLines
  const pow = Math.pow(10, Math.floor(Math.log10(rough)))
  const norm = rough / pow
  let step: number
  if (norm < 1.5) step = 1
  else if (norm < 3.5) step = 2
  else if (norm < 7.5) step = 5
  else step = 10
  return step * pow
}

function formatValue(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 100) return v.toFixed(1)
  if (Math.abs(v) >= 1) return v.toFixed(2)
  return v.toFixed(3)
}
