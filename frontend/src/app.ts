import type { Scenario } from './types'
import { SCENARIO_COLORS } from './types'
import { parseNDJSON } from './parser'
import { SceneManager } from './scene'
import { ChartManager } from './charts'

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

  constructor() {
    // Build DOM
    document.body.innerHTML = ''
    const root = document.createElement('div')
    root.id = 'app'
    root.innerHTML = `
      <header id="header">
        <div id="brand">SIMULACRA</div>
        <div id="scenario-list"></div>
        <div id="header-actions">
          <button id="load-btn">LOAD</button>
          <button id="clear-btn">CLEAR</button>
        </div>
      </header>
      <div id="main">
        <div id="scene-container">
          <div id="empty-state">DROP NDJSON FILES</div>
        </div>
        <div id="charts-container"></div>
      </div>
      <div id="playback-bar">
        <div id="playback-controls">
          <button id="rewind-btn" class="pb-btn" title="Rewind">⏮</button>
          <button id="play-btn" class="pb-btn" title="Play/Pause">▶</button>
        </div>
        <input type="range" id="timeline" min="0" max="1000" value="0" />
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

    // Scene and Charts
    this.sceneManager = new SceneManager(sceneContainer)
    this.chartManager = new ChartManager(chartsContainer)

    // Event listeners
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

    // Start animation loop
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
      const data = parseNDJSON(text)
      if (data.frames.length === 0) continue

      const label = file.name.replace(/\.(ndjson|jsonl)$/i, '')
      const color = SCENARIO_COLORS[this.scenarios.length % SCENARIO_COLORS.length]
      const scenario: Scenario = { label, color, data }

      this.scenarios.push(scenario)
      this.sceneManager.addScenario(scenario)
      this.addScenarioChip(scenario)
    }

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
    }
  }

  private addScenarioChip(scenario: Scenario): void {
    const chip = document.createElement('div')
    chip.className = 'scenario-chip'
    chip.innerHTML = `<span class="chip-dot" style="background:${scenario.color}"></span>${scenario.label}`
    this.scenarioList.appendChild(chip)
  }

  private clearAll(): void {
    this.scenarios = []
    this.sceneManager.clearAll()
    this.scenarioList.innerHTML = ''
    this.playing = false
    this.currentTime = 0
    this.maxTime = 0
    this.playBtn.textContent = '▶'
    this.emptyState.style.display = ''
    this.chartManager.draw([], 0, 0)
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

    // Advance playback
    if (this.playing && this.maxTime > 0) {
      this.currentTime += wallDt * this.speed
      if (this.currentTime > this.maxTime) {
        this.currentTime = 0 // loop
      }
    }

    // Update timeline slider
    if (this.maxTime > 0) {
      this.timeline.value = String(
        Math.round((this.currentTime / this.maxTime) * 1000),
      )
    }

    // Update time display
    this.timeDisplay.textContent = `T+ ${pad(this.currentTime)} / ${pad(this.maxTime)}`

    // Update 3D scene (every frame)
    this.sceneManager.update(this.scenarios, this.currentTime)

    // Update charts (throttled to ~15fps for performance)
    this.chartThrottle += wallDt
    if (this.chartThrottle > 0.066) {
      this.chartThrottle = 0
      this.chartManager.draw(this.scenarios, this.currentTime, this.maxTime)
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

function pad(t: number): string {
  return t.toFixed(2).padStart(7, '0')
}
