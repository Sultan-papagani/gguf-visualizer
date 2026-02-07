/**
 * Three.js Renderer
 * Sets up the 3D scene with Points geometry, custom shaders,
 * FPS-style free-roam camera (WASD + Space/Shift + mouse look),
 * and handles resizing/animation.
 */

import * as THREE from 'three';

// Custom vertex shader: point size scales with distance + per-vertex color
const vertexShader = `
  attribute vec3 customColor;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uPointSize;
  uniform float uPixelRatio;

  void main() {
    vColor = customColor;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float dist = -mvPosition.z;
    gl_PointSize = uPointSize * uPixelRatio * (80.0 / max(dist, 1.0));
    gl_PointSize = clamp(gl_PointSize, 0.5, 32.0);
    vAlpha = smoothstep(500.0, 50.0, dist);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

// Custom fragment shader: circular points with soft edges
const fragmentShader = `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) discard;
    float alpha = smoothstep(0.5, 0.2, dist) * (0.6 + 0.4 * vAlpha);
    gl_FragColor = vec4(vColor, alpha);
  }
`;

// ─── FPS Camera Controller ──────────────────────────────────────────

class FlyControls {
  constructor(camera, domElement) {
    this.camera = camera;
    this.domElement = domElement;

    this.moveSpeed = 30.0;
    this.lookSpeed = 0.002;
    this.sprintMultiplier = 3.0;

    // Euler angles for mouse look
    this.yaw = 0;
    this.pitch = 0;

    // Movement state
    this.keys = {};
    this.isPointerLocked = false;

    // Scroll-zoom speed adjust
    this.speedLevel = 1.0;

    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onPointerLockChange = this._onPointerLockChange.bind(this);
    this._onClick = this._onClick.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onContextMenu = this._onContextMenu.bind(this);

    document.addEventListener('keydown', this._onKeyDown);
    document.addEventListener('keyup', this._onKeyUp);
    document.addEventListener('mousemove', this._onMouseMove);
    document.addEventListener('pointerlockchange', this._onPointerLockChange);
    this.domElement.addEventListener('click', this._onClick);
    this.domElement.addEventListener('wheel', this._onWheel, { passive: false });
    this.domElement.addEventListener('contextmenu', this._onContextMenu);

    // Initialize yaw/pitch from current camera orientation
    const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
    this.yaw = euler.y;
    this.pitch = euler.x;
  }

  _onContextMenu(e) { e.preventDefault(); }

  _onClick(e) {
    // Only lock on left-click on the canvas itself
    if (e.button !== 0) return;
    if (!this.isPointerLocked) {
      this.domElement.requestPointerLock();
    }
  }

  _onPointerLockChange() {
    this.isPointerLocked = document.pointerLockElement === this.domElement;
  }

  _onKeyDown(e) {
    // Don't capture keys when typing in inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    this.keys[e.code] = true;
  }

  _onKeyUp(e) {
    this.keys[e.code] = false;
  }

  _onMouseMove(e) {
    if (!this.isPointerLocked) return;

    this.yaw   -= e.movementX * this.lookSpeed;
    this.pitch -= e.movementY * this.lookSpeed;
    // Clamp pitch to avoid gimbal flip
    this.pitch = Math.max(-Math.PI * 0.495, Math.min(Math.PI * 0.495, this.pitch));
  }

  _onWheel(e) {
    e.preventDefault();
    // Scroll adjusts movement speed
    if (e.deltaY < 0) {
      this.speedLevel = Math.min(this.speedLevel * 1.25, 20.0);
    } else {
      this.speedLevel = Math.max(this.speedLevel / 1.25, 0.05);
    }
  }

  update(delta) {
    // Apply look rotation
    const quat = new THREE.Quaternion();
    const euler = new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ');
    quat.setFromEuler(euler);
    this.camera.quaternion.copy(quat);

    // Movement vectors
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const right   = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up      = new THREE.Vector3(0, 1, 0);

    const sprint = this.keys['ShiftLeft'] || this.keys['ShiftRight'] ? this.sprintMultiplier : 1.0;
    const speed = this.moveSpeed * this.speedLevel * sprint * delta;

    // WASD movement
    if (this.keys['KeyW'] || this.keys['ArrowUp'])    this.camera.position.addScaledVector(forward, speed);
    if (this.keys['KeyS'] || this.keys['ArrowDown'])   this.camera.position.addScaledVector(forward, -speed);
    if (this.keys['KeyA'] || this.keys['ArrowLeft'])   this.camera.position.addScaledVector(right, -speed);
    if (this.keys['KeyD'] || this.keys['ArrowRight'])  this.camera.position.addScaledVector(right, speed);

    // Space = up, Ctrl / C = down
    if (this.keys['Space'])                             this.camera.position.addScaledVector(up, speed);
    if (this.keys['ControlLeft'] || this.keys['KeyC'])  this.camera.position.addScaledVector(up, -speed);
  }

  dispose() {
    document.removeEventListener('keydown', this._onKeyDown);
    document.removeEventListener('keyup', this._onKeyUp);
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('pointerlockchange', this._onPointerLockChange);
    this.domElement.removeEventListener('click', this._onClick);
    this.domElement.removeEventListener('wheel', this._onWheel);
    this.domElement.removeEventListener('contextmenu', this._onContextMenu);
    if (this.isPointerLocked) document.exitPointerLock();
  }
}

// ─── Renderer ────────────────────────────────────────────────────────

export class ModelRenderer {
  constructor(container) {
    this.container = container;
    this.tensorRegions = [];
    this.pointCloud = null;
    this.connectionLines = null;
    this.clock = new THREE.Clock();

    this._initScene();
    this._initControls();
    this._animate = this._animate.bind(this);
    this._onResize = this._onResize.bind(this);

    window.addEventListener('resize', this._onResize);
    this._animate();
  }

  _initScene() {
    const rect = this.container.getBoundingClientRect();

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(rect.width, rect.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0a0f, 1);
    this.container.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x0a0a0f, 0.003);

    // Camera
    this.camera = new THREE.PerspectiveCamera(70, rect.width / rect.height, 0.1, 2000);
    this.camera.position.set(30, 20, -30);
    this.camera.lookAt(0, 0, 0);

    // Subtle ambient grid helper for orientation
    const gridHelper = new THREE.GridHelper(200, 40, 0x1a1a26, 0x111118);
    gridHelper.position.y = -1;
    this.scene.add(gridHelper);

    // Axes helper (subtle)
    const axesHelper = new THREE.AxesHelper(3);
    axesHelper.position.set(-2, -1, -2);
    this.scene.add(axesHelper);

    // Point size uniform
    this.pointSize = 0.6;
  }

  _initControls() {
    this.controls = new FlyControls(this.camera, this.renderer.domElement);
  }

  /**
   * Set the point cloud data.
   * @param {Float32Array} positions - xyz positions (length = N*3)
   * @param {Float32Array} colors - rgb colors (length = N*3)
   * @param {Array} tensorRegions - Metadata about tensor regions
   */
  setPointCloud(positions, colors, tensorRegions) {
    // Remove old point cloud
    if (this.pointCloud) {
      this.scene.remove(this.pointCloud);
      this.pointCloud.geometry.dispose();
      this.pointCloud.material.dispose();
      this.pointCloud = null;
    }

    this.tensorRegions = tensorRegions || [];

    const pointCount = positions.length / 3;
    if (pointCount === 0) return;

    // Create geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('customColor', new THREE.Float32BufferAttribute(colors, 3));

    // Compute bounding box for camera fit
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    // Create material
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uPointSize: { value: this.pointSize },
        uPixelRatio: { value: this.renderer.getPixelRatio() },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.pointCloud = new THREE.Points(geometry, material);
    this.scene.add(this.pointCloud);

    // Fit camera to model
    this._fitCamera(geometry.boundingBox, geometry.boundingSphere);
  }

  /**
   * Set connection lines (neural pathways between tensor regions).
   * @param {Float32Array} linePositions - xyz pairs for line segments (length = N*6)
   * @param {Float32Array} lineColors - rgb pairs for line segments (length = N*6)
   */
  setConnections(linePositions, lineColors) {
    // Remove old lines
    if (this.connectionLines) {
      this.scene.remove(this.connectionLines);
      this.connectionLines.geometry.dispose();
      this.connectionLines.material.dispose();
      this.connectionLines = null;
    }

    if (!linePositions || linePositions.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.connectionLines = new THREE.LineSegments(geometry, material);
    this.scene.add(this.connectionLines);
  }

  /**
   * Toggle connection lines visibility.
   */
  toggleConnections(visible) {
    if (this.connectionLines) {
      this.connectionLines.visible = visible;
    }
  }

  /**
   * Set connection line opacity.
   */
  setConnectionOpacity(opacity) {
    if (this.connectionLines) {
      this.connectionLines.material.opacity = opacity;
    }
  }

  /**
   * Update only the colors (for when user changes color mode).
   */
  updateColors(colors) {
    if (!this.pointCloud) return;
    const attr = this.pointCloud.geometry.getAttribute('customColor');
    attr.array.set(colors);
    attr.needsUpdate = true;
  }

  /**
   * Set the point size.
   */
  setPointSize(size) {
    this.pointSize = size;
    if (this.pointCloud) {
      this.pointCloud.material.uniforms.uPointSize.value = size;
    }
  }

  /**
   * Fit the camera to show the entire model.
   */
  _fitCamera(bbox, bsphere) {
    const center = new THREE.Vector3();
    bbox.getCenter(center);

    const size = new THREE.Vector3();
    bbox.getSize(size);

    const maxDim = Math.max(size.x, size.y, size.z);
    const distance = maxDim * 1.5;

    this.camera.position.set(
      center.x + distance * 0.6,
      center.y + distance * 0.5,
      center.z - distance * 0.8
    );
    this.camera.lookAt(center);

    // Re-sync yaw/pitch from the new lookAt orientation
    const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.controls.yaw = euler.y;
    this.controls.pitch = euler.x;

    // Scale movement speed to model size
    this.controls.moveSpeed = Math.max(5, maxDim * 0.4);

    this.camera.near = Math.max(0.1, distance * 0.001);
    this.camera.far = Math.max(2000, distance * 10);
    this.camera.updateProjectionMatrix();

    // Update fog density based on model size
    this.scene.fog.density = 1.5 / Math.max(maxDim, 1);
  }

  /**
   * Get the tensor region at a given screen position (for hover tooltips).
   */
  getTensorAtScreen(screenX, screenY) {
    if (!this.pointCloud || !this.tensorRegions.length) return null;

    const rect = this.container.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((screenX - rect.left) / rect.width) * 2 - 1,
      -((screenY - rect.top) / rect.height) * 2 + 1
    );

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.5;
    raycaster.setFromCamera(ndc, this.camera);

    const intersects = raycaster.intersectObject(this.pointCloud);
    if (intersects.length > 0) {
      const idx = intersects[0].index;
      for (const tr of this.tensorRegions) {
        if (idx >= tr.startIdx && idx < tr.endIdx) {
          return tr;
        }
      }
    }
    return null;
  }

  _onResize() {
    const rect = this.container.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    this.camera.aspect = rect.width / rect.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height);
  }

  _animate() {
    requestAnimationFrame(this._animate);
    const delta = this.clock.getDelta();
    this.controls.update(delta);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    window.removeEventListener('resize', this._onResize);
    if (this.pointCloud) {
      this.pointCloud.geometry.dispose();
      this.pointCloud.material.dispose();
    }
    if (this.connectionLines) {
      this.connectionLines.geometry.dispose();
      this.connectionLines.material.dispose();
    }
    this.renderer.dispose();
    this.controls.dispose();
  }
}
