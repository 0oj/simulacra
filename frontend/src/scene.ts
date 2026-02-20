import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import {
  CSS2DRenderer,
  CSS2DObject,
} from 'three/addons/renderers/CSS2DRenderer.js'
import type { FlightFrame, Scenario } from './types'
import { frameIndexAt } from './parser'

const BG_COLOR = 0x0a0e1a
const GRID_COLOR = 0x1a2744
const GROUND_COLOR = 0x0f1629

// Pre-rotation: mesh is built with nose along +Y, body frame has nose along +X.
// Rotate +Y → +X = rotation of -90° around Z.
const PRE_ROT = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(0, 0, 1),
  -Math.PI / 2,
)

interface Annotation {
  time: number
  text: string
  position: THREE.Vector3
  object: CSS2DObject
}

interface WindState {
  mesh: THREE.LineSegments
  geometry: THREE.BufferGeometry
  centers: Float32Array          // cx, cy, cz per streak
  dir: [number, number, number]  // unit wind direction
  speed: number                  // world units/sec
  halfLen: number
  // bounding box for wrapping
  minX: number; maxX: number
  minY: number; maxY: number
  minZ: number; maxZ: number
}

interface RocketObject {
  group: THREE.Group
  body: THREE.Mesh
  nose: THREE.Mesh
  flame: THREE.Mesh
  trail: THREE.Line
  trailGeometry: THREE.BufferGeometry
  groundTrail: THREE.Line
  groundTrailGeometry: THREE.BufferGeometry
  color: string
  baseColor: THREE.Color
  annotations: Annotation[]
  leaderLine: THREE.Line
  leaderLineGeometry: THREE.BufferGeometry
  wind: WindState | null
}

export class SceneManager {
  private renderer: THREE.WebGLRenderer
  private labelRenderer: CSS2DRenderer
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private lastUpdateTime = 0
  private rockets: Map<string, RocketObject> = new Map()
  private container: HTMLElement
  private grid: THREE.GridHelper
  private ground: THREE.Mesh

  constructor(container: HTMLElement) {
    this.container = container

    // WebGL renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.setClearColor(BG_COLOR)
    container.appendChild(this.renderer.domElement)

    // CSS2D renderer for annotation labels (layered on top)
    this.labelRenderer = new CSS2DRenderer()
    this.labelRenderer.domElement.style.position = 'absolute'
    this.labelRenderer.domElement.style.top = '0'
    this.labelRenderer.domElement.style.left = '0'
    this.labelRenderer.domElement.style.pointerEvents = 'none'
    container.appendChild(this.labelRenderer.domElement)

    // Scene
    this.scene = new THREE.Scene()

    // Ground grid (will be resized in frameFit)
    this.grid = new THREE.GridHelper(500, 50, GRID_COLOR, GRID_COLOR)
    this.grid.material.opacity = 0.4
    this.grid.material.transparent = true
    this.scene.add(this.grid)

    // Ground plane (will be resized in frameFit)
    const groundGeo = new THREE.PlaneGeometry(1, 1)
    const groundMat = new THREE.MeshBasicMaterial({
      color: GROUND_COLOR,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.3,
    })
    this.ground = new THREE.Mesh(groundGeo, groundMat)
    this.ground.rotation.x = -Math.PI / 2
    this.ground.position.y = -0.01
    this.scene.add(this.ground)

    // Origin marker
    const originGeo = new THREE.RingGeometry(0.3, 0.5, 32)
    const originMat = new THREE.MeshBasicMaterial({
      color: 0x3b82f6,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.5,
    })
    const origin = new THREE.Mesh(originGeo, originMat)
    origin.rotation.x = -Math.PI / 2
    origin.position.y = 0.01
    this.scene.add(origin)

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.6)
    this.scene.add(ambient)

    const directional = new THREE.DirectionalLight(0xffffff, 0.8)
    directional.position.set(50, 100, 30)
    this.scene.add(directional)

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000)
    this.camera.position.set(30, 40, 60)

    // Controls — attach to label renderer so it sits on top
    this.controls = new OrbitControls(
      this.camera,
      this.labelRenderer.domElement,
    )
    this.labelRenderer.domElement.style.pointerEvents = 'auto'
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.1
    this.controls.target.set(0, 20, 0)

    // Lock Y-up: prevent camera from flipping past the poles
    this.controls.maxPolarAngle = Math.PI * 0.95
    this.controls.minPolarAngle = Math.PI * 0.05

    this.handleResize()
    window.addEventListener('resize', () => this.handleResize())
  }

  handleResize(): void {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
    this.labelRenderer.setSize(w, h)
  }

  addScenario(scenario: Scenario): void {
    const { label, color, data } = scenario
    const hexColor = new THREE.Color(color).getHex()
    const rocketLen = data.rocketLength

    const group = new THREE.Group()

    // Body cylinder
    const bodyHeight = rocketLen * 0.8
    const bodyRadius = rocketLen * 0.04
    const bodyGeo = new THREE.CylinderGeometry(
      bodyRadius,
      bodyRadius,
      bodyHeight,
      12,
    )
    bodyGeo.translate(0, bodyHeight / 2, 0)
    const bodyMat = new THREE.MeshPhongMaterial({
      color: 0xd4d4d8,
      emissive: hexColor,
      emissiveIntensity: 0.15,
      transparent: true,
      opacity: 0.7,
    })
    const body = new THREE.Mesh(bodyGeo, bodyMat)
    group.add(body)

    // Nose cone
    const noseHeight = rocketLen * 0.2
    const noseGeo = new THREE.ConeGeometry(bodyRadius, noseHeight, 12)
    noseGeo.translate(0, bodyHeight + noseHeight / 2, 0)
    const noseMat = new THREE.MeshPhongMaterial({
      color: hexColor,
      transparent: true,
      opacity: 0.7,
    })
    const nose = new THREE.Mesh(noseGeo, noseMat)
    group.add(nose)

    // Thrust flame (initially invisible)
    const flameGeo = new THREE.ConeGeometry(
      bodyRadius * 1.5,
      rocketLen * 0.3,
      8,
    )
    flameGeo.translate(0, -rocketLen * 0.15, 0)
    const flameMat = new THREE.MeshBasicMaterial({
      color: 0xff6600,
      transparent: true,
      opacity: 0.8,
    })
    const flame = new THREE.Mesh(flameGeo, flameMat)
    flame.visible = false
    group.add(flame)

    this.scene.add(group)

    // Trail line (flight)
    const trailGeo = new THREE.BufferGeometry()
    const trailMat = new THREE.LineBasicMaterial({
      color: hexColor,
      transparent: true,
      opacity: 0.6,
    })
    const trail = new THREE.Line(trailGeo, trailMat)
    this.scene.add(trail)

    // Ground contact trail (highlighted segments when on_ground)
    const groundTrailGeo = new THREE.BufferGeometry()
    const groundTrailMat = new THREE.LineBasicMaterial({
      color: 0xef4444,
      transparent: true,
      opacity: 0.9,
      linewidth: 2,
    })
    const groundTrail = new THREE.Line(groundTrailGeo, groundTrailMat)
    this.scene.add(groundTrail)

    // Annotations — pre-compute positions from debug messages.
    // Labels live at the MARGIN: offset to the side with a leader line,
    // so they don't obscure the rocket or trajectory.
    const LABEL_OFFSET = new THREE.Vector3(8, 2, 0)
    const annotations: Annotation[] = []
    for (const dm of data.debugMsgs) {
      const idx = frameIndexAt(data.frames, dm.time)
      if (idx < 0) continue
      const [px, py, pz] = data.frames[idx].position

      const el = document.createElement('div')
      el.className = 'annotation-label'
      el.style.color = color
      el.textContent = dm.text
      el.style.display = 'none'

      const obj = new CSS2DObject(el)
      // Position at offset so labels don't sit on the trajectory
      obj.position.set(
        px + LABEL_OFFSET.x,
        py + LABEL_OFFSET.y,
        pz + LABEL_OFFSET.z,
      )
      this.scene.add(obj)

      annotations.push({
        time: dm.time,
        text: dm.text,
        position: new THREE.Vector3(px, py, pz),
        object: obj,
      })
    }

    // Auto-generate landing annotation from impact data
    for (let i = data.frames.length - 1; i >= 0; i--) {
      const frame = data.frames[i]
      if (frame.impactSpeed != null) {
        const [px, py, pz] = frame.position
        const angleDeg = frame.landingAngle != null
          ? (frame.landingAngle * 180 / Math.PI).toFixed(1)
          : '?'
        const text = `IMPACT ${frame.impactSpeed.toFixed(1)} m/s  ${angleDeg}°`

        const el = document.createElement('div')
        el.className = 'annotation-label annotation-impact'
        el.style.color = color
        el.textContent = text
        el.style.display = 'none'

        const obj = new CSS2DObject(el)
        obj.position.set(px + LABEL_OFFSET.x, py + 0.5 + LABEL_OFFSET.y, pz)
        this.scene.add(obj)

        annotations.push({
          time: frame.time,
          text,
          position: new THREE.Vector3(px, py, pz),
          object: obj,
        })
        break // only the first (last chronological) impact
      }
    }

    // Leader line: a single reusable line per rocket connecting
    // the trajectory point to the active annotation label.
    const leaderLineGeo = new THREE.BufferGeometry()
    leaderLineGeo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3),
    )
    const leaderLineMat = new THREE.LineBasicMaterial({
      color: hexColor,
      transparent: true,
      opacity: 0.25,
    })
    const leaderLine = new THREE.Line(leaderLineGeo, leaderLineMat)
    leaderLine.visible = false
    this.scene.add(leaderLine)

    // Wind streaks — animated line segments drifting in the wind direction
    let windState: WindState | null = null
    const wind = data.wind
    if (wind) {
      const wmag = Math.hypot(wind[0], wind[1], wind[2])
      if (wmag > 0.01) {
        // Compute bounding box from frames
        let bMinX = Infinity, bMinY = Infinity, bMinZ = Infinity
        let bMaxX = -Infinity, bMaxY = -Infinity, bMaxZ = -Infinity
        for (const f of data.frames) {
          const [px, py, pz] = f.position
          bMinX = Math.min(bMinX, px); bMaxX = Math.max(bMaxX, px)
          bMinY = Math.min(bMinY, py); bMaxY = Math.max(bMaxY, py)
          bMinZ = Math.min(bMinZ, pz); bMaxZ = Math.max(bMaxZ, pz)
        }

        // Pad the bounding box
        const rangeX = bMaxX - bMinX || 10
        const rangeY = bMaxY - bMinY || 10
        const rangeZ = bMaxZ - bMinZ || 10
        const boxPad = Math.max(rangeX, rangeY, rangeZ) * 0.4
        bMinX -= boxPad; bMaxX += boxPad
        bMinZ -= boxPad; bMaxZ += boxPad
        bMinY = 0; bMaxY = bMaxY * 0.8

        const dx = wind[0] / wmag
        const dy = wind[1] / wmag
        const dz = wind[2] / wmag
        const halfLen = Math.min(1.5 + wmag * 0.2, 6.0)

        const COUNT = 60
        const centers = new Float32Array(COUNT * 3)
        const positions = new Float32Array(COUNT * 6)
        for (let i = 0; i < COUNT; i++) {
          const cx = bMinX + Math.random() * (bMaxX - bMinX)
          const cy = bMinY + Math.random() * (bMaxY - bMinY)
          const cz = bMinZ + Math.random() * (bMaxZ - bMinZ)
          centers[i * 3]     = cx
          centers[i * 3 + 1] = cy
          centers[i * 3 + 2] = cz
          const off = i * 6
          positions[off]     = cx - dx * halfLen
          positions[off + 1] = cy - dy * halfLen
          positions[off + 2] = cz - dz * halfLen
          positions[off + 3] = cx + dx * halfLen
          positions[off + 4] = cy + dy * halfLen
          positions[off + 5] = cz + dz * halfLen
        }

        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        const mat = new THREE.LineBasicMaterial({
          color: hexColor,
          transparent: true,
          opacity: 0.18,
        })
        const mesh = new THREE.LineSegments(geo, mat)
        this.scene.add(mesh)

        windState = {
          mesh, geometry: geo, centers,
          dir: [dx, dy, dz],
          speed: wmag * 0.5,
          halfLen,
          minX: bMinX, maxX: bMaxX,
          minY: bMinY, maxY: bMaxY,
          minZ: bMinZ, maxZ: bMaxZ,
        }
      }
    }

    this.rockets.set(label, {
      group,
      body,
      nose,
      flame,
      trail,
      trailGeometry: trailGeo,
      groundTrail,
      groundTrailGeometry: groundTrailGeo,
      color,
      baseColor: new THREE.Color(color),
      annotations,
      leaderLine,
      leaderLineGeometry: leaderLineGeo,
      wind: windState,
    })
  }

  removeScenario(label: string): void {
    const rocket = this.rockets.get(label)
    if (!rocket) return
    this.scene.remove(rocket.group)
    this.scene.remove(rocket.trail)
    this.scene.remove(rocket.groundTrail)
    this.scene.remove(rocket.leaderLine)
    rocket.trailGeometry.dispose()
    rocket.groundTrailGeometry.dispose()
    rocket.leaderLineGeometry.dispose()
    if (rocket.wind) {
      this.scene.remove(rocket.wind.mesh)
      rocket.wind.geometry.dispose()
    }
    for (const ann of rocket.annotations) {
      this.scene.remove(ann.object)
    }
    this.rockets.delete(label)
  }

  clearAll(): void {
    for (const label of Array.from(this.rockets.keys())) {
      this.removeScenario(label)
    }
  }

  update(scenarios: Scenario[], currentTime: number): void {
    const now = performance.now()
    const wallDt = this.lastUpdateTime ? (now - this.lastUpdateTime) / 1000 : 0
    this.lastUpdateTime = now

    for (const scenario of scenarios) {
      const rocket = this.rockets.get(scenario.label)
      if (!rocket) continue

      const { frames } = scenario.data
      const idx = frameIndexAt(frames, currentTime)
      if (idx < 0) continue

      const frame = frames[idx]
      this.updateRocket(rocket, frame)
      this.updateTrail(rocket, frames, idx)
      this.updateAnnotations(rocket, currentTime)
      if (rocket.wind) this.updateWind(rocket.wind, wallDt)
    }

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
    this.labelRenderer.render(this.scene, this.camera)
  }

  private updateRocket(rocket: RocketObject, frame: FlightFrame): void {
    const [px, py, pz] = frame.position
    rocket.group.position.set(px, py, pz)

    // Apply orientation: dataQuat (body→world) * preRot (mesh→body)
    const [w, x, y, z] = frame.quatWXYZ
    const dataQuat = new THREE.Quaternion(x, y, z, w)
    rocket.group.quaternion.copy(dataQuat).multiply(PRE_ROT)

    // Flame visibility
    const thrustMag = Math.hypot(
      frame.thrustBody[0],
      frame.thrustBody[1],
      frame.thrustBody[2],
    )
    rocket.flame.visible = thrustMag > 1.0
    if (thrustMag > 1.0) {
      const scale = Math.min(thrustMag / 500, 3.0)
      rocket.flame.scale.set(scale, scale, scale)
    }

    // Tipped-over indicator: nose turns red
    const noseMat = rocket.nose.material as THREE.MeshPhongMaterial
    if (frame.tippedOver) {
      noseMat.color.setHex(0xef4444)
    } else {
      noseMat.color.copy(rocket.baseColor)
    }
  }

  private updateTrail(
    rocket: RocketObject,
    frames: FlightFrame[],
    endIdx: number,
  ): void {
    const positions: number[] = []
    const groundPositions: number[] = []

    for (let i = 0; i <= endIdx; i++) {
      const [px, py, pz] = frames[i].position
      positions.push(px, py, pz)

      // Build ground contact trail segments
      if (frames[i].onGround) {
        if (groundPositions.length === 0 && i > 0) {
          // Include previous point for continuity
          const [ppx, ppy, ppz] = frames[i - 1].position
          groundPositions.push(ppx, ppy, ppz)
        }
        groundPositions.push(px, py, pz)
      } else if (groundPositions.length > 0) {
        // Add NaN break to separate ground segments
        groundPositions.push(NaN, NaN, NaN)
      }
    }

    rocket.trailGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3),
    )
    rocket.trailGeometry.computeBoundingSphere()

    if (groundPositions.length > 0) {
      rocket.groundTrailGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(groundPositions, 3),
      )
      rocket.groundTrailGeometry.computeBoundingSphere()
    }
  }

  private updateAnnotations(rocket: RocketObject, currentTime: number): void {
    // Only show the MOST RECENT non-impact annotation.
    // Impact annotations are shown once they occur and stay visible.
    // This keeps text subordinate to geometry — one label at a time.

    let activeAnn: Annotation | null = null

    for (const ann of rocket.annotations) {
      const el = ann.object.element as HTMLElement
      const isImpact = el.classList.contains('annotation-impact')

      if (isImpact) {
        el.style.display = ann.time <= currentTime ? '' : 'none'
        continue
      }

      // Track the most recent annotation before current time
      if (ann.time <= currentTime) {
        activeAnn = ann
      }
      el.style.display = 'none'
    }

    // Show only the active annotation with its leader line
    if (activeAnn) {
      const el = activeAnn.object.element as HTMLElement
      el.style.display = ''

      // Update leader line from trajectory point to label offset
      const pos = rocket.leaderLineGeometry.getAttribute(
        'position',
      ) as THREE.BufferAttribute
      pos.setXYZ(
        0,
        activeAnn.position.x,
        activeAnn.position.y,
        activeAnn.position.z,
      )
      pos.setXYZ(
        1,
        activeAnn.object.position.x,
        activeAnn.object.position.y,
        activeAnn.object.position.z,
      )
      pos.needsUpdate = true
      rocket.leaderLine.visible = true
    } else {
      rocket.leaderLine.visible = false
    }
  }

  private updateWind(w: WindState, dt: number): void {
    const [dx, dy, dz] = w.dir
    const step = w.speed * dt
    const posAttr = w.geometry.getAttribute('position') as THREE.BufferAttribute
    const pos = posAttr.array as Float32Array
    const count = w.centers.length / 3

    for (let i = 0; i < count; i++) {
      // Advance center along wind direction
      let cx = w.centers[i * 3]     += dx * step
      let cy = w.centers[i * 3 + 1] += dy * step
      let cz = w.centers[i * 3 + 2] += dz * step

      // Wrap around bounding box — re-enter from the upwind side
      if (cx < w.minX || cx > w.maxX ||
          cy < w.minY || cy > w.maxY ||
          cz < w.minZ || cz > w.maxZ) {
        cx = w.minX + Math.random() * (w.maxX - w.minX)
        cy = w.minY + Math.random() * (w.maxY - w.minY)
        cz = w.minZ + Math.random() * (w.maxZ - w.minZ)
        // Place on the upwind face: push back along wind direction
        cx -= dx * (w.maxX - w.minX) * 0.5
        cz -= dz * (w.maxZ - w.minZ) * 0.5
        // Clamp back into box
        cx = Math.max(w.minX, Math.min(w.maxX, cx))
        cy = Math.max(w.minY, Math.min(w.maxY, cy))
        cz = Math.max(w.minZ, Math.min(w.maxZ, cz))
        w.centers[i * 3]     = cx
        w.centers[i * 3 + 1] = cy
        w.centers[i * 3 + 2] = cz
      }

      const off = i * 6
      pos[off]     = cx - dx * w.halfLen
      pos[off + 1] = cy - dy * w.halfLen
      pos[off + 2] = cz - dz * w.halfLen
      pos[off + 3] = cx + dx * w.halfLen
      pos[off + 4] = cy + dy * w.halfLen
      pos[off + 5] = cz + dz * w.halfLen
    }

    posAttr.needsUpdate = true
  }

  /** Reset camera to default position looking at origin */
  resetCamera(): void {
    this.controls.target.set(0, 20, 0)
    this.camera.position.set(30, 40, 60)
    this.controls.update()
  }

  /** Focus camera on current rocket position */
  focusOnRocket(scenarios: Scenario[], currentTime: number): void {
    if (scenarios.length === 0) return

    // Find the current position of the first rocket
    const scenario = scenarios[0]
    const frames = scenario.data.frames
    const idx = frameIndexAt(frames, currentTime)
    if (idx < 0) return

    const [px, py, pz] = frames[idx].position
    const rocketLen = scenario.data.rocketLength

    // Set target to rocket position
    this.controls.target.set(px, py, pz)

    // Position camera at a good viewing distance
    const dist = Math.max(rocketLen * 8, 20)
    this.camera.position.set(
      px + dist * 0.5,
      py + dist * 0.6,
      pz + dist * 0.8,
    )
    this.controls.update()
  }

  /** Focus camera on a specific position (for interactive mode) */
  focusOnPosition(x: number, y: number, z: number): void {
    this.controls.target.set(x, y, z)
    const dist = 25
    this.camera.position.set(
      x + dist * 0.5,
      y + dist * 0.6,
      z + dist * 0.8,
    )
    this.controls.update()
  }

  frameFit(scenarios: Scenario[]): void {
    if (scenarios.length === 0) return

    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity

    for (const s of scenarios) {
      for (const f of s.data.frames) {
        const [px, py, pz] = f.position
        minX = Math.min(minX, px)
        minY = Math.min(minY, py)
        minZ = Math.min(minZ, pz)
        maxX = Math.max(maxX, px)
        maxY = Math.max(maxY, py)
        maxZ = Math.max(maxZ, pz)
      }
    }

    const cx = (minX + maxX) / 2
    const cy = (minY + maxY) / 2
    const cz = (minZ + maxZ) / 2
    const range = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 10)

    // Resize grid and ground to fit the data with generous padding
    const groundSize = range * 3
    const divisions = Math.max(10, Math.min(100, Math.round(groundSize / 10)))

    this.scene.remove(this.grid)
    this.grid.dispose()
    this.grid = new THREE.GridHelper(groundSize, divisions, GRID_COLOR, GRID_COLOR)
    this.grid.material.opacity = 0.4
    this.grid.material.transparent = true
    this.grid.position.set(cx, 0, cz)
    this.scene.add(this.grid)

    this.ground.scale.set(groundSize, groundSize, 1)
    this.ground.position.set(cx, -0.01, cz)

    this.controls.target.set(cx, cy, cz)
    this.camera.position.set(
      cx + range * 0.6,
      cy + range * 0.8,
      cz + range * 1.0,
    )
  }

  /**
   * Update a single rocket for interactive mode.
   * Creates the rocket if it doesn't exist.
   */
  updateSingleRocket(
    position: [number, number, number],
    orientation: [number, number, number, number],
    thrustBody: [number, number, number],
    onGround: boolean,
  ): void {
    const LIVE_KEY = '__live__'
    let rocket = this.rockets.get(LIVE_KEY)

    if (!rocket) {
      // Create rocket for live mode
      const color = '#4f4'
      const rocketLen = 4.0
      const bodyRadius = 0.2
      const hexColor = parseInt(color.replace('#', ''), 16)

      const group = new THREE.Group()

      // Body
      const bodyGeo = new THREE.CylinderGeometry(bodyRadius, bodyRadius, rocketLen, 16)
      bodyGeo.translate(0, rocketLen / 2, 0)
      const bodyMat = new THREE.MeshBasicMaterial({ color: hexColor })
      const body = new THREE.Mesh(bodyGeo, bodyMat)
      group.add(body)

      // Nose cone
      const noseGeo = new THREE.ConeGeometry(bodyRadius, rocketLen * 0.2, 16)
      noseGeo.translate(0, rocketLen + rocketLen * 0.1, 0)
      const noseMat = new THREE.MeshBasicMaterial({ color: 0xffffff })
      const nose = new THREE.Mesh(noseGeo, noseMat)
      group.add(nose)

      // Flame
      const flameGeo = new THREE.ConeGeometry(bodyRadius * 1.5, rocketLen * 0.3, 8)
      flameGeo.translate(0, -rocketLen * 0.15, 0)
      const flameMat = new THREE.MeshBasicMaterial({
        color: 0xff6600,
        transparent: true,
        opacity: 0.8,
      })
      const flame = new THREE.Mesh(flameGeo, flameMat)
      flame.visible = false
      group.add(flame)

      this.scene.add(group)

      // Empty trail (not used in live mode but required by type)
      const trailGeo = new THREE.BufferGeometry()
      const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: hexColor }))
      const groundTrailGeo = new THREE.BufferGeometry()
      const groundTrail = new THREE.Line(groundTrailGeo, new THREE.LineBasicMaterial({ color: 0xff0000 }))
      const leaderLineGeo = new THREE.BufferGeometry()
      const leaderLine = new THREE.Line(leaderLineGeo, new THREE.LineBasicMaterial({ color: hexColor }))

      rocket = {
        group,
        body,
        nose,
        flame,
        trail,
        trailGeometry: trailGeo,
        groundTrail,
        groundTrailGeometry: groundTrailGeo,
        color,
        baseColor: new THREE.Color(hexColor),
        annotations: [],
        leaderLine,
        leaderLineGeometry: leaderLineGeo,
        wind: null,
      }

      this.rockets.set(LIVE_KEY, rocket)

      // Setup initial camera - look at ground where rocket starts
      this.controls.target.set(0, 5, 0)
      this.camera.position.set(20, 15, 30)
      this.controls.update()
    }

    // Update position
    const [px, py, pz] = position
    rocket.group.position.set(px, py, pz)

    // Update orientation
    const [qw, qx, qy, qz] = orientation
    const bodyQ = new THREE.Quaternion(qx, qy, qz, qw)
    rocket.group.quaternion.copy(bodyQ).multiply(PRE_ROT)

    // Update flame
    const thrustMag = Math.hypot(thrustBody[0], thrustBody[1], thrustBody[2])
    rocket.flame.visible = thrustMag > 10
    if (rocket.flame.visible) {
      const scale = 0.5 + (thrustMag / 2000) * 1.5
      rocket.flame.scale.set(scale, scale, scale)
    }

    // Color feedback
    if (onGround) {
      (rocket.body.material as THREE.MeshBasicMaterial).color.setHex(0x44ff44)
    } else {
      (rocket.body.material as THREE.MeshBasicMaterial).color.copy(rocket.baseColor)
    }

    // Render
    this.renderer.render(this.scene, this.camera)
    this.labelRenderer.render(this.scene, this.camera)
  }
}
