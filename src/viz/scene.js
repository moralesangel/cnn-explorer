// The 3D view: one InstancedMesh of cubes per layer, plus selection highlights, the lines
// to a selected neuron's inputs, and the floating kernel "ghost" for convolutions.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import {
  LAYERS, LAYER_BY_ID, NETWORK_WIDTH, NETWORK_DEPTH,
  cellCenter, cellFootprint, pixelCenter, mapOrigin,
} from './layout.js';
import { ACTIVATION, GRAY, DIVERGING, writeColor } from './colors.js';

const BASE_HEIGHT = 0.12;
const MAX_LINKS = 400;
const MAX_GHOST = 128;
const VIEW_DIR = new THREE.Vector3(-0.06, 1, 0.78).normalize();
const unitBox = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);

const POS = new THREE.Color('#ffab2e');
const NEG = new THREE.Color('#2f7bff');
const MAX = new THREE.Color('#ffffff');

export class NetworkView {
  /**
   * @param {HTMLElement} container
   * @param {{ onHover?: Function, onSelect?: Function, onLayerClick?: Function }} handlers
   */
  constructor(container, handlers = {}) {
    this.container = container;
    this.handlers = handlers;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0b0f18);
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    Object.assign(this.labelRenderer.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
    container.appendChild(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x0b0f18, 700, 1400);
    this.camera = new THREE.PerspectiveCamera(38, 1, 1, 4000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 15;
    this.controls.maxDistance = 1500;

    this.scene.add(new THREE.HemisphereLight(0xdde6ff, 0x1a2030, 2.4));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(-0.5, 1, 0.7);
    this.scene.add(sun);

    this._buildLayers();
    this._buildSelectionObjects();

    this.selection = null;
    this.flight = null;
    this.pointer = { ndc: new THREE.Vector2(), x: 0, y: 0, inside: false, moved: false, down: null };
    this.hovered = null;
    this.raycaster = new THREE.Raycaster();
    this._bindPointer();

    new ResizeObserver(() => this._resize()).observe(container);
    this._resize();
    this.overview(false);

    this.lastFrame = performance.now();
    this.renderer.setAnimationLoop(() => this._frame());
  }

  // ------------------------------------------------------------------ construction

  _buildLayers() {
    const plateMat = new THREE.MeshBasicMaterial({ color: 0x131a2a });
    for (const L of LAYERS) {
      const mesh = new THREE.InstancedMesh(unitBox, new THREE.MeshLambertMaterial(), L.count);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(L.count * 3), 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.userData.layer = L;
      // Heights change all the time, so give raycasting a fixed, generous bounding sphere.
      const r = Math.hypot(L.width / 2, L.depth / 2, L.heightScale) + 2;
      mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(L.cx, L.heightScale / 2, L.z0 + L.depth / 2), r);
      this.scene.add(mesh);

      L.mesh = mesh;
      L.current = new Float32Array(L.count);
      L.target = new Float32Array(L.count);
      L.startAt = 0;
      L.settled = false;
      L.foot = cellFootprint(L);
      L.palette = L.id === 'input' ? GRAY.linear : ACTIVATION.linear;
      L.centers = new Float32Array(L.count * 2);
      for (let i = 0; i < L.count; i++) {
        const c = cellCenter(L, i);
        L.centers[2 * i] = c.x;
        L.centers[2 * i + 1] = c.z;
      }
      this._writeLayer(L);

      // Dark plates under each feature map (or the whole vector) so maps read as units.
      const plates = L.kind === 'map'
        ? Array.from({ length: L.maps }, (_, m) => {
          const o = mapOrigin(L, m);
          return { x: o.x + L.size / 2, z: o.z + L.size / 2, w: L.size + 0.8, d: L.size + 0.8 };
        })
        : [{ x: L.cx, z: 0, w: L.width + 1.2, d: L.depth + 0.6 }];
      for (const p of plates) {
        const plate = new THREE.Mesh(new THREE.PlaneGeometry(p.w, p.d), plateMat);
        plate.rotation.x = -Math.PI / 2;
        plate.position.set(p.x, -0.02, p.z);
        this.scene.add(plate);
      }

      const el = document.createElement('div');
      el.className = 'layer-label';
      el.innerHTML = `<div class="t">${L.title}</div><div class="s">${L.sub}</div>`;
      el.addEventListener('click', () => this.handlers.onLayerClick?.(L.id));
      const label = new CSS2DObject(el);
      label.position.set(L.cx, 0, L.z0 - 9);
      this.scene.add(label);
      L.labelEl = el;
    }

    // Digit names next to the output cubes.
    const out = LAYER_BY_ID.output;
    this.digitLabels = [];
    for (let d = 0; d < out.units; d++) {
      const el = document.createElement('div');
      el.className = 'digit-label';
      el.textContent = d;
      const label = new CSS2DObject(el);
      const c = cellCenter(out, d);
      label.position.set(c.x + out.cube / 2 + 4, 0.5, c.z);
      this.scene.add(label);
      this.digitLabels.push(el);
    }
  }

  _buildSelectionObjects() {
    const edges = new THREE.EdgesGeometry(unitBox);
    this.hoverBox = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x9fb0d8 }));
    this.selectBox = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0xffffff }));
    this.hoverBox.visible = this.selectBox.visible = false;
    this.scene.add(this.hoverBox, this.selectBox);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_LINKS * 6), 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX_LINKS * 8), 4).setUsage(THREE.DynamicDrawUsage));
    geo.setDrawRange(0, 0);
    this.linkLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, depthWrite: false,
    }));
    this.linkLines.frustumCulled = false;
    this.scene.add(this.linkLines);

    this.ghost = new THREE.InstancedMesh(unitBox, new THREE.MeshLambertMaterial(), MAX_GHOST);
    this.ghost.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_GHOST * 3), 3);
    this.ghost.count = 0;
    this.ghost.frustumCulled = false;
    this.scene.add(this.ghost);
  }

  _bindPointer() {
    const el = this.renderer.domElement;
    const setPointer = (e) => {
      const r = el.getBoundingClientRect();
      this.pointer.x = e.clientX - r.left;
      this.pointer.y = e.clientY - r.top;
      this.pointer.ndc.set((this.pointer.x / r.width) * 2 - 1, -(this.pointer.y / r.height) * 2 + 1);
      this.pointer.inside = true;
      this.pointer.moved = true;
    };
    el.addEventListener('pointermove', setPointer);
    el.addEventListener('pointerdown', (e) => {
      setPointer(e);
      this.pointer.down = { x: e.clientX, y: e.clientY, button: e.button };
      this.flight = null; // user takes over the camera
    });
    el.addEventListener('pointerup', (e) => {
      const d = this.pointer.down;
      this.pointer.down = null;
      if (!d || d.button !== 0 || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;
      setPointer(e);
      const hit = this._pick();
      if (hit) this.handlers.onSelect?.(hit);
    });
    el.addEventListener('pointerleave', () => {
      this.pointer.inside = false;
      this._setHover(null);
    });
    el.addEventListener('wheel', () => { this.flight = null; }, { passive: true });
  }

  _resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------ values

  /**
   * Sets the displayed activation (normalized to 0..1) for some or all layers.
   * With `stagger`, layers animate one after another like a forward pass.
   */
  setValues(values, { stagger = 45, immediate = false } = {}) {
    const now = performance.now();
    let order = 0;
    for (const L of LAYERS) {
      const v = values[L.id];
      if (!v) continue;
      L.target.set(v);
      if (immediate) {
        L.current.set(v);
        L.settled = true;
        this._writeLayer(L);
      } else {
        L.settled = false;
        L.startAt = now + order * stagger;
      }
      order++;
    }
    this.selectionDirty = true;
  }

  /** Drops every layer back to zero and lets the forward pass play again, slowly. */
  replay() {
    for (const L of LAYERS) {
      L.current.fill(0);
      this._writeLayer(L);
    }
    const now = performance.now();
    LAYERS.forEach((L, i) => { L.settled = false; L.startAt = now + 150 + i * 320; });
  }

  setTopDigit(d) {
    this.digitLabels.forEach((el, i) => el.classList.toggle('top', i === d));
  }

  setActiveLayer(id) {
    for (const L of LAYERS) L.labelEl.classList.toggle('active', L.id === id);
  }

  _writeLayer(L) {
    const m = L.mesh.instanceMatrix.array;
    const col = L.mesh.instanceColor.array;
    const f = L.foot;
    for (let i = 0; i < L.count; i++) {
      const v = L.current[i];
      const o = i * 16;
      m[o] = f;
      m[o + 5] = BASE_HEIGHT + v * L.heightScale;
      m[o + 10] = f;
      m[o + 12] = L.centers[2 * i];
      m[o + 14] = L.centers[2 * i + 1];
      writeColor(L.palette, v, col, i * 3);
    }
    L.mesh.instanceMatrix.needsUpdate = true;
    L.mesh.instanceColor.needsUpdate = true;
  }

  _animateLayers(now, dt) {
    const k = 1 - Math.exp(-dt * 9);
    let changed = false;
    for (const L of LAYERS) {
      if (L.settled || now < L.startAt) continue;
      let maxDelta = 0;
      for (let i = 0; i < L.count; i++) {
        const d = L.target[i] - L.current[i];
        L.current[i] += d * k;
        if (Math.abs(d) > maxDelta) maxDelta = Math.abs(d);
      }
      if (maxDelta < 0.003) {
        L.current.set(L.target);
        L.settled = true;
      }
      this._writeLayer(L);
      changed = true;
    }
    return changed;
  }

  // ------------------------------------------------------------------ selection

  _top(layerId, index) {
    const L = LAYER_BY_ID[layerId];
    return {
      x: L.centers[2 * index],
      y: BASE_HEIGHT + L.current[index] * L.heightScale,
      z: L.centers[2 * index + 1],
    };
  }

  _placeBox(box, ref) {
    if (!ref) {
      box.visible = false;
      return;
    }
    const L = LAYER_BY_ID[ref.layer];
    const t = this._top(ref.layer, ref.index);
    const pad = L.kind === 'vector' ? 0.5 : 0.3;
    box.position.set(t.x, -0.05, t.z);
    box.scale.set(L.foot + pad, t.y + 0.1 + pad / 2, L.foot + pad);
    box.visible = true;
  }

  /**
   * @param {{layer, index} | null} ref
   * @param {{ links, ghost }} field  from connections.js
   */
  setSelection(ref, field = { links: [], ghost: [] }) {
    this.selection = ref ? { ref, ...field } : null;
    this.selectionDirty = true;

    const ghost = ref ? field.ghost.slice(0, MAX_GHOST) : [];
    const m = new THREE.Matrix4();
    ghost.forEach((g, i) => {
      const S = LAYER_BY_ID[g.layer];
      const c = pixelCenter(S, g.map, g.y, g.x);
      m.makeScale(0.94, 0.35, 0.94).setPosition(c.x, S.heightScale + 4, c.z);
      this.ghost.setMatrixAt(i, m);
      writeColor(DIVERGING.linear, 0.5 + 0.5 * g.t, this.ghost.instanceColor.array, i * 3);
    });
    this.ghost.count = ghost.length;
    this.ghost.instanceMatrix.needsUpdate = true;
    this.ghost.instanceColor.needsUpdate = true;
  }

  _updateSelectionGraphics() {
    const sel = this.selection;
    this._placeBox(this.selectBox, sel?.ref);
    this._placeBox(this.hoverBox, this.hovered && !sameRef(this.hovered, sel?.ref) ? this.hovered : null);

    const geo = this.linkLines.geometry;
    if (!sel) {
      geo.setDrawRange(0, 0);
      return;
    }
    const pos = geo.attributes.position.array;
    const col = geo.attributes.color.array;
    const to = this._top(sel.ref.layer, sel.ref.index);
    const links = sel.links.slice(0, MAX_LINKS);
    links.forEach((l, i) => {
      const from = this._top(l.layer, l.index);
      pos.set([from.x, from.y, from.z, to.x, to.y, to.z], i * 6);
      const c = l.max ? MAX : l.s >= 0 ? POS : NEG;
      const a = l.max ? 1 : 0.1 + 0.9 * Math.abs(l.s);
      col.set([c.r, c.g, c.b, a, c.r, c.g, c.b, a], i * 8);
    });
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.setDrawRange(0, links.length * 2);
  }

  _pick() {
    this.raycaster.setFromCamera(this.pointer.ndc, this.camera);
    const hits = this.raycaster.intersectObjects(LAYERS.map((L) => L.mesh), false);
    const hit = hits.find((h) => h.instanceId !== undefined);
    return hit ? { layer: hit.object.userData.layer.id, index: hit.instanceId } : null;
  }

  _setHover(ref) {
    this.hovered = ref;
    this.selectionDirty = true;
    this.handlers.onHover?.(ref, this.pointer.x, this.pointer.y);
  }

  // ------------------------------------------------------------------ camera

  _viewFor(target, span) {
    const vfov = THREE.MathUtils.degToRad(this.camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const dist = Math.max((span.w / 2) / Math.tan(hfov / 2), (span.d / 2) / Math.tan(vfov / 2)) * 1.1 + 10;
    return { target, position: target.clone().addScaledVector(VIEW_DIR, dist) };
  }

  _flyTo(view, animate = true) {
    if (!animate) {
      this.controls.target.copy(view.target);
      this.camera.position.copy(view.position);
      this.controls.update();
      return;
    }
    this.flight = {
      fromPos: this.camera.position.clone(), toPos: view.position,
      fromTarget: this.controls.target.clone(), toTarget: view.target,
      t0: performance.now(), dur: 900,
    };
  }

  overview(animate = true) {
    this._flyTo(this._viewFor(new THREE.Vector3(0, 0, 8), { w: NETWORK_WIDTH, d: NETWORK_DEPTH * 0.75 }), animate);
  }

  /** Frames one layer, or a range of layers (e.g. a layer and the one feeding it). */
  focusLayers(ids) {
    const Ls = ids.map((id) => LAYER_BY_ID[id]);
    const x0 = Math.min(...Ls.map((L) => L.x0));
    const x1 = Math.max(...Ls.map((L) => L.x0 + L.width));
    const d = Math.max(...Ls.map((L) => L.depth));
    const w = Math.max(x1 - x0, 40);
    this._flyTo(this._viewFor(new THREE.Vector3((x0 + x1) / 2, 0, 2), { w: w * 1.1, d: Math.max(d, 40) * 1.05 }));
  }

  // ------------------------------------------------------------------ loop

  _frame() {
    const now = performance.now();
    const dt = Math.min((now - this.lastFrame) / 1000, 0.05);
    this.lastFrame = now;

    if (this._animateLayers(now, dt)) this.selectionDirty = true;

    if (this.flight) {
      const f = this.flight;
      const t = Math.min(1, (now - f.t0) / f.dur);
      const e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
      this.camera.position.lerpVectors(f.fromPos, f.toPos, e);
      this.controls.target.lerpVectors(f.fromTarget, f.toTarget, e);
      if (t >= 1) this.flight = null;
    }
    this.controls.update();

    if (this.pointer.moved && this.pointer.inside && !this.pointer.down) {
      this.pointer.moved = false;
      this._setHover(this._pick());
    }

    if (this.selectionDirty) {
      this.selectionDirty = false;
      this._updateSelectionGraphics();
    }

    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }
}

const sameRef = (a, b) => !!a && !!b && a.layer === b.layer && a.index === b.index;
