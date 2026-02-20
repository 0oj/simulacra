/**
 * Interactive mode - real-time rocket control via WebSocket.
 */

import { SceneManager } from './scene'
import { niceStep, fmtCompact } from './charts'

interface RocketState {
  t: number
  position: { x: number; y: number; z: number }
  velocity: { x: number; y: number; z: number }
  orientation: number[]
  thrust: number
  throttle_percent: number
  fuel_fraction: number
  tilt_deg: number
  on_ground: boolean
  crashed: boolean
  tipped: boolean
  mass: number
  gimbal_outer_deg: number
  gimbal_inner_deg: number
}

interface HistoryPoint {
  t: number
  alt: number
  vel: number
  thr: number
}

export class InteractiveApp {
  private sceneManager: SceneManager
  private ws: WebSocket | null = null
  private keysDown = new Set<string>()
  private state: RocketState | null = null
  private timeScale = 1.0

  private history: HistoryPoint[] = []
  private readonly maxHistory = 500

  private statusEl: HTMLElement
  private speedDisplay: HTMLElement
  private timeValueEl: HTMLElement
  private telemGrid: HTMLElement

  private altCanvas: HTMLCanvasElement
  private velCanvas: HTMLCanvasElement
  private thrCanvas: HTMLCanvasElement
  private gimbalCanvas: HTMLCanvasElement

  constructor() {
    document.body.innerHTML = ''
    const root = document.createElement('div')
    root.id = 'app'
    root.innerHTML = `
      <header id="header">
        <div id="brand">SIMULACRA <span class="live-badge">LIVE</span></div>
        <div id="status">CONNECTING...</div>
        <div id="header-actions">
          <button id="reset-btn">RESET (R)</button>
          <button id="disconnect-btn">DISCONNECT</button>
        </div>
      </header>
      <div id="main">
        <div id="scene-container">
          <div id="controls-overlay">
            <div class="ctrl-row"><span class="ctrl-key">W</span><span class="ctrl-key">S</span> THROTTLE</div>
            <div class="ctrl-row"><span class="ctrl-key">A</span><span class="ctrl-key">D</span> YAW</div>
            <div class="ctrl-row"><span class="ctrl-key">Q</span><span class="ctrl-key">E</span> PITCH</div>
            <div class="ctrl-row"><span class="ctrl-key">SPACE</span> CUT</div>
            <div class="ctrl-row"><span class="ctrl-key">SHIFT</span> MAX</div>
          </div>
        </div>
        <div id="charts-container">
          <div class="live-chart-slot gimbal-slot"><canvas id="gimbal-chart"></canvas></div>
          <div class="live-chart-slot"><canvas id="alt-chart"></canvas></div>
          <div class="live-chart-slot"><canvas id="vel-chart"></canvas></div>
          <div class="live-chart-slot"><canvas id="thr-chart"></canvas></div>
          <div class="telemetry-panel">
            <div class="telem-title">TELEMETRY</div>
            <div id="telem-grid"></div>
          </div>
        </div>
      </div>
      <div id="playback-bar">
        <div id="sim-time">
          <span class="time-label">T+</span>
          <span id="time-value">000.00</span>s
        </div>
        <div id="speed-controls">
          <span class="speed-label">TIME</span>
          <button id="speed-down" class="pb-btn" title="Slower">\u2212</button>
          <span id="speed-display">1.0\u00d7</span>
          <button id="speed-up" class="pb-btn" title="Faster">+</button>
        </div>
        <div id="camera-controls">
          <button id="fit-btn" class="pb-btn" title="Fit View (F)">FIT</button>
          <button id="focus-btn" class="pb-btn" title="Focus Rocket (C)">FOCUS</button>
          <button id="reset-cam-btn" class="pb-btn" title="Reset Camera">CAM</button>
        </div>
      </div>
    `
    document.body.appendChild(root)

    this.statusEl = document.getElementById('status')!
    this.speedDisplay = document.getElementById('speed-display')!
    this.timeValueEl = document.getElementById('time-value')!
    this.telemGrid = document.getElementById('telem-grid')!

    this.gimbalCanvas = document.getElementById('gimbal-chart') as HTMLCanvasElement
    this.altCanvas = document.getElementById('alt-chart') as HTMLCanvasElement
    this.velCanvas = document.getElementById('vel-chart') as HTMLCanvasElement
    this.thrCanvas = document.getElementById('thr-chart') as HTMLCanvasElement

    this.sceneManager = new SceneManager(document.getElementById('scene-container')!)

    document.getElementById('reset-btn')!.addEventListener('click', () => this.sendReset())
    document.getElementById('disconnect-btn')!.addEventListener('click', () => this.disconnect())
    document.getElementById('speed-down')!.addEventListener('click', () => this.adjustTimeScale(0.5))
    document.getElementById('speed-up')!.addEventListener('click', () => this.adjustTimeScale(2))
    document.getElementById('fit-btn')!.addEventListener('click', () => this.sceneManager.resetCamera())
    document.getElementById('focus-btn')!.addEventListener('click', () => this.focusOnRocket())
    document.getElementById('reset-cam-btn')!.addEventListener('click', () => this.sceneManager.resetCamera())

    window.addEventListener('keydown', (e) => this.onKeyDown(e))
    window.addEventListener('keyup', (e) => this.onKeyUp(e))

    this.connect()
    this.animate()
  }

  private connect(): void {
    this.statusEl.textContent = 'CONNECTING...'
    this.statusEl.className = ''

    this.ws = new WebSocket('ws://localhost:8765')

    this.ws.onopen = () => {
      this.statusEl.textContent = 'CONNECTED'
      this.statusEl.className = 'connected'
      this.sendTimeScale()
    }

    this.ws.onclose = () => {
      this.statusEl.textContent = 'DISCONNECTED'
      this.statusEl.className = 'error'
      this.ws = null
    }

    this.ws.onerror = () => {
      this.statusEl.textContent = 'CONNECTION ERROR'
      this.statusEl.className = 'error'
    }

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'state' || data.type === 'init' || data.type === 'reset') {
        this.state = data.state
        this.addToHistory(data.state)
        if (data.type === 'reset') {
          this.history = []
        }
      }
    }
  }

  private addToHistory(s: RocketState): void {
    this.history.push({
      t: s.t,
      alt: s.position.y,
      vel: Math.hypot(s.velocity.x, s.velocity.y, s.velocity.z),
      thr: s.thrust,
    })
    if (this.history.length > this.maxHistory) {
      this.history.shift()
    }
  }

  private disconnect(): void {
    this.ws?.close()
    this.ws = null
  }

  private send(msg: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg))
    }
  }

  private sendKeys(): void {
    this.send({ type: 'keys', keys: Array.from(this.keysDown) })
  }

  private sendReset(): void {
    this.send({ type: 'reset' })
    this.history = []
  }

  private sendTimeScale(): void {
    this.send({ type: 'timescale', scale: this.timeScale })
  }

  private adjustTimeScale(multiplier: number): void {
    this.timeScale = Math.max(0.1, Math.min(4, this.timeScale * multiplier))
    this.speedDisplay.textContent = this.timeScale.toFixed(1) + '\u00d7'
    this.sendTimeScale()
  }

  private focusOnRocket(): void {
    if (this.state) {
      this.sceneManager.focusOnPosition(
        this.state.position.x,
        this.state.position.y,
        this.state.position.z
      )
    }
  }

  private onKeyDown(e: KeyboardEvent): void {
    let key = e.key.toLowerCase()
    if (e.key === ' ') key = 'space'
    if (e.shiftKey) key = 'shift'

    if (key === 'r') { this.sendReset(); return }
    if (key === '+' || key === '=') { this.adjustTimeScale(2); return }
    if (key === '-') { this.adjustTimeScale(0.5); return }
    if (key === 'f') { this.sceneManager.resetCamera(); return }
    if (key === 'c') { this.focusOnRocket(); return }

    if (!this.keysDown.has(key)) {
      this.keysDown.add(key)
      this.sendKeys()
    }

    if (['w', 's', 'a', 'd', 'q', 'e', 'space', ' '].includes(key)) {
      e.preventDefault()
    }
  }

  private onKeyUp(e: KeyboardEvent): void {
    let key = e.key.toLowerCase()
    if (e.key === ' ') key = 'space'
    if (!e.shiftKey && this.keysDown.has('shift')) {
      this.keysDown.delete('shift')
    }

    if (this.keysDown.has(key)) {
      this.keysDown.delete(key)
      this.sendKeys()
    }
  }

  private animate(): void {
    requestAnimationFrame(() => this.animate())

    if (!this.state) return

    const s = this.state

    this.sceneManager.updateSingleRocket(
      [s.position.x, s.position.y, s.position.z],
      s.orientation as [number, number, number, number],
      [s.thrust, 0, 0],
      s.on_ground,
    )

    this.timeValueEl.textContent = s.t.toFixed(2).padStart(6, '0')
    this.updateTelemetry()
    this.drawCharts()
  }

  private updateTelemetry(): void {
    const s = this.state!
    const vel = Math.hypot(s.velocity.x, s.velocity.y, s.velocity.z)
    const vVel = s.velocity.y

    const tiltClass = s.tilt_deg > 25 ? 'error' : s.tilt_deg > 15 ? 'warn' : ''
    const fuelClass = s.fuel_fraction < 0.1 ? 'error' : s.fuel_fraction < 0.25 ? 'warn' : ''
    const vVelClass = vVel < -10 ? 'error' : vVel < -5 ? 'warn' : ''

    let status = ''
    if (s.crashed) status = '<span class="telem-value error">CRASHED</span>'
    else if (s.tipped) status = '<span class="telem-value error">TIPPED</span>'
    else if (s.on_ground) status = '<span class="telem-value good">LANDED</span>'

    this.telemGrid.innerHTML = `
      <div class="telem-item"><span class="telem-label">ALT</span><span class="telem-value">${s.position.y.toFixed(1)} m</span></div>
      <div class="telem-item"><span class="telem-label">VEL</span><span class="telem-value">${vel.toFixed(1)} m/s</span></div>
      <div class="telem-item"><span class="telem-label">V-VEL</span><span class="telem-value ${vVelClass}">${vVel.toFixed(1)} m/s</span></div>
      <div class="telem-item"><span class="telem-label">TILT</span><span class="telem-value ${tiltClass}">${s.tilt_deg.toFixed(1)}\u00b0</span></div>
      <div class="telem-item"><span class="telem-label">THR</span><span class="telem-value">${s.throttle_percent.toFixed(0)}%</span></div>
      <div class="telem-item"><span class="telem-label">FUEL</span><span class="telem-value ${fuelClass}">${(s.fuel_fraction * 100).toFixed(0)}%</span></div>
      <div class="telem-item"><span class="telem-label">GMB Y</span><span class="telem-value">${s.gimbal_outer_deg.toFixed(1)}\u00b0</span></div>
      <div class="telem-item"><span class="telem-label">GMB P</span><span class="telem-value">${s.gimbal_inner_deg.toFixed(1)}\u00b0</span></div>
      ${status ? `<div class="telem-item" style="grid-column: span 2">${status}</div>` : ''}
    `
  }

  private drawCharts(): void {
    if (this.state) this.drawGimbalChart()
    if (this.history.length < 2) return

    this.drawLineChart(this.altCanvas, 'ALTITUDE', 'm', this.history.map(h => [h.t, h.alt]), '#3b82f6')
    this.drawLineChart(this.velCanvas, 'VELOCITY', 'm/s', this.history.map(h => [h.t, h.vel]), '#22c55e')
    this.drawLineChart(this.thrCanvas, 'THRUST', 'N', this.history.map(h => [h.t, h.thr]), '#f59e0b')
  }

  private drawGimbalChart(): void {
    const s = this.state!
    const canvas = this.gimbalCanvas
    const rect = canvas.parentElement!.getBoundingClientRect()
    const w = rect.width, h = rect.height
    const dpr = window.devicePixelRatio || 1

    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'

    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)
    ctx.fillStyle = '#0f1629'
    ctx.fillRect(0, 0, w, h)

    const pad = { top: 16, bottom: 4, left: 4, right: 4 }
    const size = Math.min(w - pad.left - pad.right, h - pad.top - pad.bottom)
    const cx = pad.left + (w - pad.left - pad.right) / 2
    const cy = pad.top + (h - pad.top - pad.bottom) / 2
    const maxDeg = 15
    const plotR = size / 2 - 2
    const scale = plotR / maxDeg

    // Grid
    ctx.strokeStyle = '#151e33'
    ctx.lineWidth = 1
    ctx.font = '8px ui-monospace, monospace'
    ctx.fillStyle = '#2e3a4d'
    ctx.textAlign = 'left'

    for (const deg of [5, 10, 15]) {
      const r = deg * scale
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.stroke()
      ctx.fillText(`${deg}\u00b0`, cx + r * 0.71 + 2, cy - r * 0.71 - 1)
    }

    ctx.beginPath()
    ctx.moveTo(cx - plotR, cy)
    ctx.lineTo(cx + plotR, cy)
    ctx.moveTo(cx, cy - plotR)
    ctx.lineTo(cx, cy + plotR)
    ctx.stroke()

    ctx.beginPath()
    ctx.arc(cx, cy, 2, 0, Math.PI * 2)
    ctx.fill()

    // Current position
    const px = cx + s.gimbal_outer_deg * scale
    const py = cy - s.gimbal_inner_deg * scale

    ctx.fillStyle = '#f59e0b'
    ctx.globalAlpha = 0.2
    ctx.beginPath()
    ctx.arc(px, py, 6, 0, Math.PI * 2)
    ctx.fill()

    ctx.globalAlpha = 1.0
    ctx.beginPath()
    ctx.arc(px, py, 3, 0, Math.PI * 2)
    ctx.fill()

    // Labels
    ctx.font = '9px ui-monospace, monospace'
    ctx.fillStyle = '#6b7a90'
    ctx.textAlign = 'right'
    ctx.fillText(`${Math.hypot(s.gimbal_outer_deg, s.gimbal_inner_deg).toFixed(1)}\u00b0`, w - pad.right, pad.top + 4)

    ctx.fillStyle = '#3a4455'
    ctx.textAlign = 'left'
    ctx.fillText('GIMBAL', pad.left + 2, pad.top - 4)
  }

  private drawLineChart(canvas: HTMLCanvasElement, label: string, unit: string, data: [number, number][], color: string): void {
    const rect = canvas.parentElement!.getBoundingClientRect()
    const w = rect.width, h = rect.height
    const dpr = window.devicePixelRatio || 1

    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'

    const ctx = canvas.getContext('2d')!
    ctx.scale(dpr, dpr)
    ctx.fillStyle = '#0f1629'
    ctx.fillRect(0, 0, w, h)

    const margin = { top: 16, right: 45, bottom: 2, left: 35 }
    const plotX = margin.left, plotY = margin.top
    const plotW = w - margin.left - margin.right
    const plotH = h - margin.top - margin.bottom

    if (data.length < 2) return

    const tMin = data[0][0], tMax = data[data.length - 1][0]
    let yMin = Infinity, yMax = -Infinity
    for (const [, y] of data) {
      yMin = Math.min(yMin, y)
      yMax = Math.max(yMax, y)
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1 }
    const yPad = (yMax - yMin) * 0.1
    yMin -= yPad
    yMax += yPad

    const toX = (t: number) => plotX + ((t - tMin) / (tMax - tMin || 1)) * plotW
    const toY = (v: number) => plotY + ((yMax - v) / (yMax - yMin)) * plotH

    // Grid
    ctx.strokeStyle = '#151e33'
    ctx.lineWidth = 1
    ctx.font = '9px ui-monospace, monospace'
    ctx.fillStyle = '#2e3a4d'
    ctx.textAlign = 'right'

    const yStep = niceStep(yMin, yMax, 3)
    for (let yv = Math.ceil(yMin / yStep) * yStep; yv <= yMax; yv += yStep) {
      const sy = toY(yv)
      if (sy < plotY || sy > plotY + plotH) continue
      ctx.beginPath()
      ctx.moveTo(plotX, sy)
      ctx.lineTo(plotX + plotW, sy)
      ctx.stroke()
      ctx.fillText(fmtCompact(yv), plotX - 3, sy + 3)
    }

    // Line
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    data.forEach(([t, v], i) => {
      const sx = toX(t), sy = toY(v)
      i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy)
    })
    ctx.stroke()

    // Current value
    ctx.fillStyle = '#6b7a90'
    ctx.textAlign = 'left'
    ctx.fillText(fmtCompact(data[data.length - 1][1]), plotX + plotW + 4, plotY + 10)

    // Label
    ctx.fillStyle = '#3a4455'
    ctx.fillText(`${label}  ${unit}`, plotX, plotY - 4)
  }
}
