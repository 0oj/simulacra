import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
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

interface RocketObject {
  group: THREE.Group
  body: THREE.Mesh
  nose: THREE.Mesh
  flame: THREE.Mesh
  trail: THREE.Line
  trailPositions: number[]
  trailGeometry: THREE.BufferGeometry
  color: string
}

export class SceneManager {
  private renderer: THREE.WebGLRenderer
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private controls: OrbitControls
  private rockets: Map<string, RocketObject> = new Map()
  private container: HTMLElement

  constructor(container: HTMLElement) {
    this.container = container

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.setClearColor(BG_COLOR)
    container.appendChild(this.renderer.domElement)

    // Scene
    this.scene = new THREE.Scene()

    // Ground grid
    const grid = new THREE.GridHelper(500, 50, GRID_COLOR, GRID_COLOR)
    grid.material.opacity = 0.4
    grid.material.transparent = true
    this.scene.add(grid)

    // Ground plane (subtle)
    const groundGeo = new THREE.PlaneGeometry(500, 500)
    const groundMat = new THREE.MeshBasicMaterial({
      color: GROUND_COLOR,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.3,
    })
    const ground = new THREE.Mesh(groundGeo, groundMat)
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.01
    this.scene.add(ground)

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

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.1
    this.controls.target.set(0, 20, 0)

    this.handleResize()
    window.addEventListener('resize', () => this.handleResize())
  }

  handleResize(): void {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
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
    bodyGeo.translate(0, bodyHeight / 2, 0) // bottom at origin
    const bodyMat = new THREE.MeshPhongMaterial({
      color: 0xd4d4d8,
      emissive: hexColor,
      emissiveIntensity: 0.15,
    })
    const body = new THREE.Mesh(bodyGeo, bodyMat)
    group.add(body)

    // Nose cone
    const noseHeight = rocketLen * 0.2
    const noseGeo = new THREE.ConeGeometry(bodyRadius, noseHeight, 12)
    noseGeo.translate(0, bodyHeight + noseHeight / 2, 0)
    const noseMat = new THREE.MeshPhongMaterial({
      color: hexColor,
    })
    const nose = new THREE.Mesh(noseGeo, noseMat)
    group.add(nose)

    // Thrust flame (initially invisible)
    const flameGeo = new THREE.ConeGeometry(bodyRadius * 1.5, rocketLen * 0.3, 8)
    flameGeo.translate(0, -rocketLen * 0.15, 0) // below base
    const flameMat = new THREE.MeshBasicMaterial({
      color: 0xff6600,
      transparent: true,
      opacity: 0.8,
    })
    const flame = new THREE.Mesh(flameGeo, flameMat)
    flame.visible = false
    group.add(flame)

    this.scene.add(group)

    // Trail line
    const trailGeo = new THREE.BufferGeometry()
    const trailMat = new THREE.LineBasicMaterial({
      color: hexColor,
      transparent: true,
      opacity: 0.6,
    })
    const trail = new THREE.Line(trailGeo, trailMat)
    this.scene.add(trail)

    this.rockets.set(label, {
      group,
      body,
      nose,
      flame,
      trail,
      trailPositions: [],
      trailGeometry: trailGeo,
      color,
    })
  }

  removeScenario(label: string): void {
    const rocket = this.rockets.get(label)
    if (!rocket) return
    this.scene.remove(rocket.group)
    this.scene.remove(rocket.trail)
    rocket.trailGeometry.dispose()
    this.rockets.delete(label)
  }

  clearAll(): void {
    for (const label of Array.from(this.rockets.keys())) {
      this.removeScenario(label)
    }
  }

  update(scenarios: Scenario[], currentTime: number): void {
    for (const scenario of scenarios) {
      const rocket = this.rockets.get(scenario.label)
      if (!rocket) continue

      const { frames } = scenario.data
      const idx = frameIndexAt(frames, currentTime)
      if (idx < 0) continue

      const frame = frames[idx]
      this.updateRocket(rocket, frame, scenario.data.rocketLength)
      this.updateTrail(rocket, frames, idx)
    }

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  private updateRocket(
    rocket: RocketObject,
    frame: FlightFrame,
    _rocketLength: number,
  ): void {
    const [px, py, pz] = frame.position
    rocket.group.position.set(px, py, pz)

    // Apply orientation: dataQuat (body→world) * preRot (mesh→body)
    const [w, x, y, z] = frame.quatWXYZ
    const dataQuat = new THREE.Quaternion(x, y, z, w) // THREE uses (x,y,z,w)
    rocket.group.quaternion.copy(dataQuat).multiply(PRE_ROT)

    // Flame visibility based on thrust magnitude
    const thrustMag = Math.hypot(
      frame.thrustBody[0],
      frame.thrustBody[1],
      frame.thrustBody[2],
    )
    rocket.flame.visible = thrustMag > 1.0
    if (thrustMag > 1.0) {
      // Scale flame by thrust (crude but visible)
      const scale = Math.min(thrustMag / 500, 3.0)
      rocket.flame.scale.set(scale, scale, scale)
    }
  }

  private updateTrail(
    rocket: RocketObject,
    frames: FlightFrame[],
    endIdx: number,
  ): void {
    // Rebuild trail positions up to endIdx
    const positions: number[] = []
    for (let i = 0; i <= endIdx; i++) {
      const [px, py, pz] = frames[i].position
      positions.push(px, py, pz)
    }

    rocket.trailGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3),
    )
    rocket.trailGeometry.computeBoundingSphere()
  }

  /**
   * Frame the camera to fit all trajectory data.
   */
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

    this.controls.target.set(cx, cy, cz)
    this.camera.position.set(cx + range * 0.6, cy + range * 0.8, cz + range * 1.0)
  }

  render(): void {
    this.renderer.render(this.scene, this.camera)
  }
}
