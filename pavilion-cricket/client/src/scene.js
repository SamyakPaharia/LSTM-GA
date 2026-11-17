// Stadium, pitch, stumps, ball and markers. Everything is built from primitives and canvas
// textures at load time.

import * as THREE from 'three';
import { HALF, STUMP_H, BOUNDARY_RX, BOUNDARY_RZ, BALL_R } from '../../shared/constants.js';

const PITCH_W = 3.05;
const PITCH_L = 24.4;
const FIELD = 150;

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function tex(c, aniso, repeat) {
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = aniso;
  if (repeat) {
    t.wrapS = THREE.RepeatWrapping;
    t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(repeat[0], repeat[1]);
  }
  return t;
}

// Seeded so the ground looks the same on every load.
function noiseRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function fieldTexture() {
  const N = 2048;
  const c = canvas(N, N);
  const g = c.getContext('2d');
  const px = N / FIELD;
  const X = (x) => (x + FIELD / 2) * px;
  const Z = (z) => (z + FIELD / 2) * px;

  const stripe = 5.2;
  for (let i = -15; i < 15; i++) {
    g.fillStyle = i % 2 === 0 ? '#3f8a3a' : '#357c32';
    g.fillRect(X(i * stripe), 0, stripe * px + 1, N);
  }
  // Cross-mown squares near the middle for a chequered look.
  g.globalAlpha = 0.07;
  for (let j = -15; j < 15; j++) {
    if (j % 2 === 0) continue;
    g.fillStyle = '#ffffff';
    g.fillRect(0, Z(j * stripe), N, stripe * px);
  }
  g.globalAlpha = 1;

  // The square: shorter, paler grass around the pitch.
  g.fillStyle = 'rgba(190, 210, 120, 0.22)';
  g.fillRect(X(-11), Z(-15), 22 * px, 30 * px);

  // Outside the rope.
  g.save();
  g.beginPath();
  g.rect(0, 0, N, N);
  g.ellipse(X(0), Z(0), BOUNDARY_RX * px, BOUNDARY_RZ * px, 0, 0, Math.PI * 2, true);
  g.fillStyle = 'rgba(20, 50, 20, 0.35)';
  g.fill('evenodd');
  g.restore();

  // Fielding circle: semicircles round each set of stumps joined by straight lines.
  const r = 27.43;
  g.strokeStyle = 'rgba(255,255,255,0.85)';
  g.lineWidth = 0.13 * px;
  g.setLineDash([0.9 * px, 0.9 * px]);
  g.beginPath();
  g.arc(X(0), Z(-HALF), r * px, Math.PI, 0);
  g.lineTo(X(r), Z(HALF));
  g.arc(X(0), Z(HALF), r * px, 0, Math.PI);
  g.closePath();
  g.stroke();
  g.setLineDash([]);

  // Grain.
  const img = g.getImageData(0, 0, N, N);
  const d = img.data;
  const rnd = noiseRng(7);
  for (let i = 0; i < d.length; i += 4) {
    const n = (rnd() - 0.5) * 14;
    d[i] += n;
    d[i + 1] += n * 1.2;
    d[i + 2] += n * 0.6;
  }
  g.putImageData(img, 0, 0);
  return c;
}

function pitchTexture() {
  const W = 256;
  const H = 2048;
  const c = canvas(W, H);
  const g = c.getContext('2d');
  const px = W / PITCH_W;
  const X = (x) => (x + PITCH_W / 2) * px;
  const Z = (z) => (z + PITCH_L / 2) * px;

  const grad = g.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#b9a774');
  grad.addColorStop(0.5, '#cdbb86');
  grad.addColorStop(1, '#b9a774');
  g.fillStyle = grad;
  g.fillRect(0, 0, W, H);

  const rnd = noiseRng(11);
  // Wear around each crease and down the middle.
  for (const zc of [-8.6, 8.6]) {
    for (let i = 0; i < 260; i++) {
      const x = (rnd() - 0.5) * 1.6;
      const z = zc + (rnd() - 0.5) * 2.4;
      g.fillStyle = `rgba(${120 + rnd() * 40}, ${100 + rnd() * 30}, 60, ${0.08 + rnd() * 0.12})`;
      g.beginPath();
      g.ellipse(X(x), Z(z), (0.05 + rnd() * 0.15) * px, (0.05 + rnd() * 0.2) * px, rnd() * 3, 0, Math.PI * 2);
      g.fill();
    }
  }
  const img = g.getImageData(0, 0, W, H);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (rnd() - 0.5) * 22;
    d[i] += n;
    d[i + 1] += n;
    d[i + 2] += n * 0.8;
  }
  g.putImageData(img, 0, 0);

  g.fillStyle = '#f8f6ee';
  const line = 0.05 * px;
  for (const s of [-1, 1]) {
    const pop = s * (HALF - 1.22);
    const bowl = s * HALF;
    g.fillRect(0, Z(pop) - line / 2, W, line);
    g.fillRect(X(-1.32), Z(bowl) - line / 2, 2.64 * px, line);
    const z0 = Math.min(pop, s * (HALF + 1.22));
    const z1 = Math.max(pop, s * (HALF + 1.22));
    for (const x of [-1.32, 1.32]) g.fillRect(X(x) - line / 2, Z(z0), line, (z1 - z0) * px);
  }
  return c;
}

function crowdTexture() {
  const W = 1024;
  const H = 512;
  const c = canvas(W, H);
  const g = c.getContext('2d');
  g.fillStyle = '#2a2f3a';
  g.fillRect(0, 0, W, H);
  const rnd = noiseRng(23);
  const shirts = ['#d8d8d8', '#2b8ae6', '#f2c230', '#c62836', '#2aa84a', '#ff8a1e', '#7b1e3a', '#ffffff', '#1f389e', '#0d5c46', '#e9e2cf'];
  const skins = ['#f1c9a5', '#d9a273', '#b97c50', '#8a5a38', '#6b432a'];
  const rows = 26;
  const rowH = H / rows;
  for (let r = 0; r < rows; r++) {
    g.fillStyle = r % 2 ? '#323846' : '#2b303c';
    g.fillRect(0, r * rowH, W, rowH);
    for (let x = 2; x < W; x += 9 + rnd() * 3) {
      if (rnd() < 0.12) continue;
      const y = r * rowH + rowH * 0.35;
      g.fillStyle = shirts[Math.floor(rnd() * shirts.length)];
      g.fillRect(x, y, 7, rowH * 0.62);
      g.fillStyle = skins[Math.floor(rnd() * skins.length)];
      g.beginPath();
      g.arc(x + 3.5, y - 1, 3, 0, Math.PI * 2);
      g.fill();
    }
  }
  return c;
}

function boardTexture() {
  const W = 2048;
  const H = 64;
  const c = canvas(W, H);
  const g = c.getContext('2d');
  const boards = [
    ['#163a2f', '#e8c477', 'PAVILION CRICKET'], ['#e8c477', '#163a2f', 'WILLOW & LEATHER'], ['#9e2b25', '#f6efe0', 'SIX AND OUT'],
    ['#f6efe0', '#163a2f', 'THE LONG ROOM'], ['#14324f', '#f6efe0', 'SQUARE LEG TEAS'], ['#163a2f', '#f6efe0', 'FOLLOW ON'],
  ];
  const w = W / boards.length;
  boards.forEach(([bg, fg, text], i) => {
    g.fillStyle = bg;
    g.fillRect(i * w, 0, w, H);
    g.fillStyle = fg;
    g.font = 'bold 34px "Saira Condensed", "Arial Narrow", Arial, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(text, i * w + w / 2, H / 2 + 2);
  });
  return c;
}

function ellipseStrip(rx, rz, y0, y1, n, uRepeat) {
  const pos = [];
  const uv = [];
  const idx = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2;
    const x = Math.cos(a) * rx;
    const z = Math.sin(a) * rz;
    pos.push(x, y0, z, x, y1, z);
    const u = (i / n) * uRepeat;
    uv.push(u, 0, u, 1);
    if (i < n) {
      const k = i * 2;
      idx.push(k, k + 2, k + 1, k + 1, k + 2, k + 3);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}

export class Stumps {
  constructor(z, mat) {
    this.group = new THREE.Group();
    this.group.position.z = z;
    this.parts = [];
    const stumpGeo = new THREE.CylinderGeometry(0.0175, 0.0175, STUMP_H, 8).translate(0, STUMP_H / 2, 0);
    const bailGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.11, 6).rotateZ(Math.PI / 2);
    for (const x of [-0.096, 0, 0.096]) {
      const m = new THREE.Mesh(stumpGeo, mat);
      m.position.x = x;
      this.group.add(m);
      this.parts.push({ m, kind: 'stump', home: m.position.clone() });
    }
    for (const x of [-0.048, 0.048]) {
      const m = new THREE.Mesh(bailGeo, mat);
      m.position.set(x, STUMP_H + 0.012, 0);
      this.group.add(m);
      this.parts.push({ m, kind: 'bail', home: m.position.clone() });
    }
    this.active = false;
  }

  // dir: +1 if the stumps are knocked toward +z, -1 toward -z.
  break(dir, power = 1, side = 0) {
    this.active = true;
    for (const p of this.parts) {
      if (p.kind === 'bail') {
        p.v = new THREE.Vector3((Math.random() - 0.5) * 3 + side, 2.5 + Math.random() * 2.5 * power, dir * (2 + Math.random() * 3) * power);
        p.w = new THREE.Vector3(Math.random() * 20, Math.random() * 20, Math.random() * 20);
      } else {
        p.v = null;
        p.tilt = 0;
        p.tiltV = (0.6 + Math.random()) * power * 9;
        p.tiltMax = 0.25 + Math.random() * 0.7 * power;
        p.axis = new THREE.Vector3(1, 0, (Math.random() - 0.5) * 0.6).normalize();
        p.dir = dir;
      }
    }
  }

  reset() {
    this.active = false;
    for (const p of this.parts) {
      p.m.position.copy(p.home);
      p.m.rotation.set(0, 0, 0);
    }
  }

  update(dt) {
    if (!this.active) return;
    for (const p of this.parts) {
      if (p.kind === 'bail') {
        if (p.m.position.y <= 0.01 && p.v.y <= 0) continue;
        p.v.y -= 9.81 * dt;
        p.m.position.addScaledVector(p.v, dt);
        p.m.rotation.x += p.w.x * dt;
        p.m.rotation.y += p.w.y * dt;
        if (p.m.position.y < 0.01) {
          p.m.position.y = 0.01;
          p.v.multiplyScalar(0.3);
          p.v.y = 0;
          p.w.multiplyScalar(0.2);
        }
      } else if (p.tilt < p.tiltMax) {
        p.tilt = Math.min(p.tiltMax, p.tilt + p.tiltV * dt);
        p.m.quaternion.setFromAxisAngle(p.axis, p.tilt * p.dir);
      }
    }
  }
}

export class Scene3D {
  constructor(canvasEl) {
    const renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer = renderer;
    const aniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());

    const scene = new THREE.Scene();
    this.scene = scene;
    const horizon = new THREE.Color(0xcfe0ea);
    scene.fog = new THREE.Fog(horizon, 160, 420);
    scene.background = horizon;

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 900);
    this.camera.position.set(0, 20, 60);

    scene.add(new THREE.HemisphereLight(0xdfefff, 0x4a6b33, 1.15));
    const sun = new THREE.DirectionalLight(0xfff1d6, 1.9);
    sun.position.set(60, 110, 30);
    scene.add(sun);

    // Sky dome with a vertical gradient.
    const skyGeo = new THREE.SphereGeometry(600, 24, 12);
    const colors = [];
    const top = new THREE.Color(0x4f86c9);
    const pos = skyGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const t = Math.max(0, pos.getY(i) / 600);
      const c = horizon.clone().lerp(top, Math.pow(t, 0.6));
      colors.push(c.r, c.g, c.b);
    }
    skyGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    scene.add(new THREE.Mesh(skyGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false })));

    // Ground.
    const outer = new THREE.Mesh(new THREE.CircleGeometry(260, 48).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ color: 0x2f6a2c }));
    outer.position.y = -0.02;
    scene.add(outer);
    const field = new THREE.Mesh(new THREE.PlaneGeometry(FIELD, FIELD).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ map: tex(fieldTexture(), aniso) }));
    scene.add(field);

    const pitch = new THREE.Mesh(new THREE.PlaneGeometry(PITCH_W, PITCH_L).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ map: tex(pitchTexture(), aniso) }));
    pitch.position.y = 0.006;
    scene.add(pitch);

    // Rope.
    const pts = [];
    for (let i = 0; i < 160; i++) {
      const a = (i / 160) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * BOUNDARY_RX, 0.07, Math.sin(a) * BOUNDARY_RZ));
    }
    const rope = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, true), 320, 0.07, 6, true), new THREE.MeshLambertMaterial({ color: 0xf4f1e6 }));
    scene.add(rope);

    // Advertising boards just outside the rope.
    const boards = new THREE.Mesh(ellipseStrip(BOUNDARY_RX + 3, BOUNDARY_RZ + 3, 0, 0.95, 160, 8), new THREE.MeshBasicMaterial({ map: tex(boardTexture(), aniso, [1, 1]), side: THREE.DoubleSide }));
    boards.material.map.wrapS = THREE.RepeatWrapping;
    scene.add(boards);

    // Stands.
    const sz = BOUNDARY_RZ / BOUNDARY_RX;
    const tier = new THREE.Mesh(
      new THREE.LatheGeometry([new THREE.Vector2(82, 1.2), new THREE.Vector2(104, 22)], 96),
      new THREE.MeshLambertMaterial({ map: tex(crowdTexture(), aniso, [14, 1]), side: THREE.DoubleSide }),
    );
    tier.scale.z = sz;
    scene.add(tier);
    const concrete = new THREE.MeshLambertMaterial({ color: 0x8d949e, side: THREE.DoubleSide });
    const wall = new THREE.Mesh(new THREE.LatheGeometry([new THREE.Vector2(81, 0), new THREE.Vector2(81, 2.2), new THREE.Vector2(82, 1.2)], 96), concrete);
    wall.scale.z = sz;
    scene.add(wall);
    const back = new THREE.Mesh(new THREE.LatheGeometry([new THREE.Vector2(104, 22), new THREE.Vector2(106, 31), new THREE.Vector2(90, 33)], 96), new THREE.MeshLambertMaterial({ color: 0xe9e4d8, side: THREE.DoubleSide }));
    back.scale.z = sz;
    scene.add(back);
    const apron = new THREE.Mesh(new THREE.RingGeometry(0.01, 1, 96, 1).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ color: 0x5c6b4e }));
    apron.geometry = new THREE.RingGeometry(BOUNDARY_RX + 6, 82, 96, 1).rotateX(-Math.PI / 2);
    apron.scale.z = sz;
    apron.position.y = 0.01;
    scene.add(apron);

    // Floodlight towers.
    const towerMat = new THREE.MeshLambertMaterial({ color: 0x9aa2ab });
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xfff6dc });
    for (let k = 0; k < 4; k++) {
      const a = Math.PI / 4 + (k * Math.PI) / 2;
      const x = Math.cos(a) * 112;
      const z = Math.sin(a) * 112 * sz;
      const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 1.2, 52, 8).translate(0, 26, 0), towerMat);
      tower.position.set(x, 0, z);
      scene.add(tower);
      const head = new THREE.Group();
      head.position.set(x, 53, z);
      head.lookAt(0, 0, 0);
      const frame = new THREE.Mesh(new THREE.BoxGeometry(10, 6, 0.6), towerMat);
      const lamps = new THREE.Mesh(new THREE.PlaneGeometry(9.2, 5.2), lampMat);
      lamps.position.z = 0.32;
      head.add(frame, lamps);
      scene.add(head);
    }

    // Sight screens.
    const screenMat = new THREE.MeshLambertMaterial({ color: 0xf2f2ee });
    for (const s of [-1, 1]) {
      const scr = new THREE.Mesh(new THREE.BoxGeometry(20, 8, 0.6).translate(0, 4, 0), screenMat);
      scr.position.z = s * (BOUNDARY_RZ + 7);
      scene.add(scr);
    }

    // Stumps.
    const stumpMat = new THREE.MeshLambertMaterial({ color: 0xf1e7cc });
    this.stumpsS = new Stumps(HALF, stumpMat);
    this.stumpsN = new Stumps(-HALF, stumpMat);
    scene.add(this.stumpsS.group, this.stumpsN.group);

    // Ball.
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), new THREE.MeshLambertMaterial({ color: 0xffffff, emissive: 0x555555 }));
    scene.add(this.ball);
    this.ballShadow = new THREE.Mesh(new THREE.CircleGeometry(0.06, 12).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }));
    this.ballShadow.renderOrder = 2;
    scene.add(this.ballShadow);
    this.ball.visible = false;
    this.ballShadow.visible = false;

    // Markers on the pitch.
    const marker = (color, inner, outer) => {
      const g = new THREE.Group();
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false });
      g.add(new THREE.Mesh(new THREE.RingGeometry(inner, outer, 32).rotateX(-Math.PI / 2), mat));
      g.add(new THREE.Mesh(new THREE.CircleGeometry(inner * 0.35, 16).rotateX(-Math.PI / 2), mat));
      g.position.y = 0.015;
      g.renderOrder = 3;
      g.visible = false;
      g.mat = mat;
      scene.add(g);
      return g;
    };
    this.aimMarker = marker(0xffd36a, 0.15, 0.21);
    this.bounceMarker = marker(0xff5a4a, 0.12, 0.17);

    this.resize();
  }

  setBall(p, visible = true) {
    this.ball.visible = visible;
    this.ballShadow.visible = visible;
    if (!visible) return;
    this.ball.position.set(p[0], Math.max(BALL_R, p[1]), p[2]);
    this.ballShadow.position.set(p[0], 0.018, p[2]);
    const h = Math.max(0, p[1]);
    this.ballShadow.scale.setScalar(1 + h * 0.4);
    this.ballShadow.material.opacity = Math.max(0.08, 0.35 - h * 0.03);
    // Keep the ball readable at distance.
    const d = this.camera.position.distanceTo(this.ball.position);
    this.ball.scale.setScalar(Math.max(1, d / 22));
  }

  resize() {
    const c = this.renderer.domElement;
    const w = c.clientWidth || window.innerWidth;
    const h = c.clientHeight || window.innerHeight;
    if (c.width !== Math.floor(w * this.renderer.getPixelRatio()) || c.height !== Math.floor(h * this.renderer.getPixelRatio())) {
      this.renderer.setSize(w, h, false);
    }
    this.camera.aspect = w / Math.max(1, h);
    // Narrow screens need a wider view to keep the pitch in frame.
    this.camera.fov = this.camera.aspect < 1 ? 58 : 42;
    this.camera.updateProjectionMatrix();
  }

  update(dt) {
    this.stumpsS.update(dt);
    this.stumpsN.update(dt);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
