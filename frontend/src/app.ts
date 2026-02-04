import type { Scenario } from './types'
import { SCENARIO_COLORS } from './types'
import { parseNDJSON, frameIndexAt } from './parser'
import { SceneManager } from './scene'
import { ChartManager } from './charts'
import { saveFile, getFile, listRecent, deleteFile } from './store'

export class App {
  private sceneManager: SceneManager
  private chartManager: ChartManager
  private scenarios: Scenario[] = []

  // Playback
  private playing = false
  private speed = 1.0
  private currentTime = 0
  private maxTime = 0
  private lastFrameTime = 0

  // DOM
  private playBtn: HTMLElement
  private timeline: HTMLInputElement
  private timeDisplay: HTMLElement
  private speedDisplay: HTMLElement
  private scenarioList: HTMLElement
  private emptyState: HTMLElement
  private eventMarkersEl: HTMLElement
  private hudEl: HTMLElement

  constructor() {
    document.body.innerHTML = ''
    const root = document.createElement('div')
    root.id = 'app'
    root.innerHTML = `
      <header id="header">
        <div id="brand">SIMULACRA</div>
        <div id="scenario-list"></div>
        <div id="header-actions">
          <div id="recent-wrapper">
            <button id="recent-btn">RECENT</button>
            <div id="recent-dropdown" class="dropdown hidden"></div>
          </div>
          <button id="load-btn">LOAD</button>
          <button id="clear-btn">CLEAR</button>
        </div>
      </header>
      <div id="main">
        <div id="scene-container">
          <div id="empty-state">DROP NDJSON FILES</div>
          <div id="hud"></div>
        </div>
        <div id="charts-container"></div>
      </div>
      <div id="playback-bar">
        <div id="playback-controls">
          <button id="rewind-btn" class="pb-btn" title="Rewind">⏮</button>
          <button id="play-btn" class="pb-btn" title="Play/Pause">▶</button>
        </div>
        <div id="timeline-wrapper">
          <div id="event-markers"></div>
          <input type="range" id="timeline" min="0" max="1000" value="0" />
        </div>
        <div id="time-readout">
          <span id="time-display">T+ 000.00 / 000.00</span>
        </div>
        <div id="speed-controls">
          <button id="speed-down" class="pb-btn" title="Slower">−</button>
          <span id="speed-display">1.0×</span>
          <button id="speed-up" class="pb-btn" title="Faster">+</button>
        </div>
      </div>
    `
    document.body.appendChild(root)

    // References
    const sceneContainer = document.getElementById('scene-container')!
    const chartsContainer = document.getElementById('charts-container')!
    this.playBtn = document.getElementById('play-btn')!
    this.timeline = document.getElementById('timeline')! as HTMLInputElement
    this.timeDisplay = document.getElementById('time-display')!
    this.speedDisplay = document.getElementById('speed-display')!
    this.scenarioList = document.getElementById('scenario-list')!
    this.emptyState = document.getElementById('empty-state')!
    this.eventMarkersEl = document.getElementById('event-markers')!
    this.hudEl = document.getElementById('hud')!

    // Scene and Charts
    this.sceneManager = new SceneManager(sceneContainer)
    this.chartManager = new ChartManager(chartsContainer)

    // Events
    this.setupDragDrop(root)
    this.setupControls()
    this.setupKeyboard()

    // Hidden file input
    const fileInput = document.createElement('input')
    fileInput.type = 'file'
    fileInput.accept = '.ndjson,.jsonl'
    fileInput.multiple = true
    fileInput.style.display = 'none'
    fileInput.id = 'file-input'
    document.body.appendChild(fileInput)
    fileInput.addEventListener('change', () => {
      if (fileInput.files) this.loadFiles(fileInput.files)
      fileInput.value = ''
    })

    document.getElementById('load-btn')!.addEventListener('click', () => {
      fileInput.click()
    })

    document.getElementById('clear-btn')!.addEventListener('click', () => {
      this.clearAll()
    })

    this.setupRecentMenu()

    this.lastFrameTime = performance.now()
    this.animate()
  }

  // ===========================================================================
  // File loading
  // ===========================================================================

  private setupDragDrop(root: HTMLElement): void {
    root.addEventListener('dragover', (e) => {
      e.preventDefault()
      root.classList.add('drag-over')
    })
    root.addEventListener('dragleave', () => {
      root.classList.remove('drag-over')
    })
    root.addEventListener('drop', (e) => {
      e.preventDefault()
      root.classList.remove('drag-over')
      if (e.dataTransfer?.files) {
        this.loadFiles(e.dataTransfer.files)
      }
    })
  }

  private async loadFiles(files: FileList): Promise<void> {
    for (const file of files) {
      const text = await file.text()
      this.loadText(file.name, text)
      saveFile(file.name, text).catch(() => {})
    }
    this.afterLoad()
  }

  private async loadFromStore(name: string): Promise<void> {
    const entry = await getFile(name)
    if (!entry) return
    this.loadText(entry.name, entry.content)
    // Bump savedAt timestamp
    saveFile(entry.name, entry.content).catch(() => {})
    this.afterLoad()
  }

  private loadText(filename: string, text: string): void {
    const data = parseNDJSON(text)
    if (data.frames.length === 0) return

    const label = filename.replace(/\.(ndjson|jsonl)$/i, '')
    const color =
      SCENARIO_COLORS[this.scenarios.length % SCENARIO_COLORS.length]
    const scenario: Scenario = { label, color, data }

    this.scenarios.push(scenario)
    this.sceneManager.addScenario(scenario)
    this.addScenarioChip(scenario)
  }

  private afterLoad(): void {
    if (this.scenarios.length > 0) {
      this.maxTime = Math.max(
        ...this.scenarios.map((s) => {
          const frames = s.data.frames
          return frames[frames.length - 1].time
        }),
      )
      this.currentTime = 0
      this.playing = true
      this.playBtn.textContent = '⏸'
      this.emptyState.style.display = 'none'
      this.sceneManager.frameFit(this.scenarios)
      this.buildEventMarkers()
    }
  }

  private addScenarioChip(scenario: Scenario): void {
    const chip = document.createElement('div')
    chip.className = 'scenario-chip'
    chip.innerHTML = `<span class="chip-dot" style="background:${scenario.color}"></span>${scenario.label}`
    this.scenarioList.appendChild(chip)
  }

  private buildEventMarkers(): void {
    this.eventMarkersEl.innerHTML = ''
    for (const s of this.scenarios) {
      for (const dm of s.data.debugMsgs) {
        if (this.maxTime <= 0) continue
        const pct = (dm.time / this.maxTime) * 100

        const marker = document.createElement('button')
        marker.className = 'event-marker'
        marker.style.left = pct + '%'
        marker.style.setProperty('--marker-color', s.color)
        marker.setAttribute('data-label', dm.text)
        marker.title = `${dm.text} — T+${dm.time.toFixed(2)}s`

        marker.addEventListener('click', (e) => {
          e.stopPropagation()
          this.currentTime = dm.time
          this.playing = false
          this.playBtn.textContent = '▶'
          this.lastFrameTime = performance.now()
        })

        this.eventMarkersEl.appendChild(marker)
      }
    }
  }

  private clearAll(): void {
    this.scenarios = []
    this.sceneManager.clearAll()
    this.scenarioList.innerHTML = ''
    this.eventMarkersEl.innerHTML = ''
    this.playing = false
    this.currentTime = 0
    this.maxTime = 0
    this.playBtn.textContent = '▶'
    this.emptyState.style.display = ''
    this.chartManager.draw([], 0, 0)
    this.hudEl.innerHTML = ''
  }

  // ===========================================================================
  // Recent files
  // ===========================================================================

  private setupRecentMenu(): void {
    const btn = document.getElementById('recent-btn')!
    const dropdown = document.getElementById('recent-dropdown')!

    btn.addEventListener('click', async (e) => {
      e.stopPropagation()
      if (!dropdown.classList.contains('hidden')) {
        dropdown.classList.add('hidden')
        return
      }

      const entries = await listRecent()
      dropdown.innerHTML = ''

      if (entries.length === 0) {
        dropdown.innerHTML = '<div class="dropdown-empty">NO SAVED FILES</div>'
      } else {
        for (const entry of entries) {
          const item = document.createElement('div')
          item.className = 'dropdown-item'

          const label = document.createElement('span')
          label.className = 'dropdown-item-name'
          label.textContent = entry.name.replace(/\.(ndjson|jsonl)$/i, '')

          const meta = document.createElement('span')
          meta.className = 'dropdown-item-meta'
          meta.textContent = fmtSize(entry.size)

          const del = document.createElement('button')
          del.className = 'dropdown-item-del'
          del.textContent = '×'
          del.title = 'Remove from recent'
          del.addEventListener('click', async (ev) => {
            ev.stopPropagation()
            await deleteFile(entry.name)
            item.remove()
            if (dropdown.children.length === 0) {
              dropdown.innerHTML = '<div class="dropdown-empty">NO SAVED FILES</div>'
            }
          })

          item.appendChild(label)
          item.appendChild(meta)
          item.appendChild(del)
          item.addEventListener('click', () => {
            dropdown.classList.add('hidden')
            this.loadFromStore(entry.name)
          })
          dropdown.appendChild(item)
        }
      }

      dropdown.classList.remove('hidden')
    })

    // Close dropdown on outside click
    document.addEventListener('click', () => {
      dropdown.classList.add('hidden')
    })
    dropdown.addEventListener('click', (e) => e.stopPropagation())
  }

  // ===========================================================================
  // Controls
  // ===========================================================================

  private setupControls(): void {
    this.playBtn.addEventListener('click', () => this.togglePlay())

    document.getElementById('rewind-btn')!.addEventListener('click', () => {
      this.currentTime = 0
    })

    document.getElementById('speed-down')!.addEventListener('click', () => {
      this.speed = Math.max(0.25, this.speed / 2)
      this.speedDisplay.textContent = this.speed + '×'
    })

    document.getElementById('speed-up')!.addEventListener('click', () => {
      this.speed = Math.min(16, this.speed * 2)
      this.speedDisplay.textContent = this.speed + '×'
    })

    this.timeline.addEventListener('input', () => {
      const frac = parseFloat(this.timeline.value) / 1000
      this.currentTime = frac * this.maxTime
    })
  }

  private setupKeyboard(): void {
    window.addEventListener('keydown', (e) => {
      if (e.key === ' ') {
        e.preventDefault()
        this.togglePlay()
      } else if (e.key === '+' || e.key === '=') {
        this.speed = Math.min(16, this.speed * 2)
        this.speedDisplay.textContent = this.speed + '×'
      } else if (e.key === '-') {
        this.speed = Math.max(0.25, this.speed / 2)
        this.speedDisplay.textContent = this.speed + '×'
      } else if (e.key === 'Home' || e.key === '0') {
        this.currentTime = 0
      }
    })
  }

  private togglePlay(): void {
    this.playing = !this.playing
    this.playBtn.textContent = this.playing ? '⏸' : '▶'
    this.lastFrameTime = performance.now()
  }

  // ===========================================================================
  // Animation loop
  // ===========================================================================

  private chartThrottle = 0

  private animate(): void {
    requestAnimationFrame(() => this.animate())

    const now = performance.now()
    const wallDt = (now - this.lastFrameTime) / 1000
    this.lastFrameTime = now

    if (this.playing && this.maxTime > 0) {
      this.currentTime += wallDt * this.speed
      if (this.currentTime > this.maxTime) {
        this.currentTime = 0
      }
    }

    if (this.maxTime > 0) {
      this.timeline.value = String(
        Math.round((this.currentTime / this.maxTime) * 1000),
      )
    }

    this.timeDisplay.textContent = `T+ ${pad(this.currentTime)} / ${pad(this.maxTime)}`

    // 3D scene (every frame)
    this.sceneManager.update(this.scenarios, this.currentTime)

    // Charts + HUD (throttled ~15fps)
    this.chartThrottle += wallDt
    if (this.chartThrottle > 0.066) {
      this.chartThrottle = 0
      this.chartManager.draw(this.scenarios, this.currentTime, this.maxTime)
      this.updateHUD()
    }
  }

  private updateHUD(): void {
    if (this.scenarios.length === 0) {
      this.hudEl.innerHTML = ''
      return
    }

    let html = ''
    for (const s of this.scenarios) {
      const idx = frameIndexAt(s.data.frames, this.currentTime)
      if (idx < 0) continue
      const f = s.data.frames[idx]

      const alt = f.position[1].toFixed(1)
      const spd = Math.hypot(f.velocity[0], f.velocity[1], f.velocity[2]).toFixed(1)
      const fuel = (f.fuelFraction * 100).toFixed(0)
      const thr = Math.hypot(f.thrustBody[0], f.thrustBody[1], f.thrustBody[2]).toFixed(0)

      // Wind readout
      let windLine = ''
      const w = s.data.wind
      if (w) {
        const wmag = Math.hypot(w[0], w[1], w[2])
        if (wmag > 0.01) {
          // Find dominant axis
          const abs = [Math.abs(w[0]), Math.abs(w[1]), Math.abs(w[2])]
          const labels = ['X', 'Y', 'Z']
          const signs = [w[0], w[1], w[2]]
          let maxI = 0
          if (abs[1] > abs[maxI]) maxI = 1
          if (abs[2] > abs[maxI]) maxI = 2
          const dir = (signs[maxI] >= 0 ? '+' : '\u2212') + labels[maxI]
          windLine = `<div class="hud-row">WIND ${wmag.toFixed(1)}N \u2192 ${dir}</div>`
        }
      }

      const flags: string[] = []
      if (f.onGround) flags.push('<span class="hud-flag hud-ground">GND</span>')
      if (f.tippedOver) flags.push('<span class="hud-flag hud-tipped">TIP</span>')

      html += `<div class="hud-scenario" style="--sc-color:${s.color}">
        <div class="hud-label">${s.label}</div>
        <div class="hud-row">ALT ${alt}m  V ${spd}m/s</div>
        <div class="hud-row">THR ${thr}N  FUEL ${fuel}%</div>
        ${windLine}
        ${flags.length > 0 ? `<div class="hud-row">${flags.join(' ')}</div>` : ''}
      </div>`
    }
    this.hudEl.innerHTML = html
  }
}

function pad(t: number): string {
  return t.toFixed(2).padStart(7, '0')
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}
