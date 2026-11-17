// Player rig and procedural animation.
// A pose gives torso angles, leg angles, hand targets and a bat direction in the body frame
// (x = the figure's left, y = up, z = the way it faces; origin at the hips). Arms are solved
// with two-bone IK so the hands always reach their targets.

import * as THREE from 'three';

const PI = Math.PI;
const HIP_H = 0.92;
const THIGH = 0.44;
const SHIN = 0.44;
const UPPER = 0.28;
const FORE = 0.27;
const SH_X = 0.2;
const SH_Y = 0.47;
const SPINE_Y = 0.05;

// Pose layout
const I = {
  hipY: 0, lean: 1, twist: 2, side: 3, headX: 4, headY: 5, lHand: 6, rHand: 9,
  lHipX: 12, lHipZ: 13, lKnee: 14, rHipX: 15, rHipZ: 16, rKnee: 17, bat: 18, yaw: 21,
};
const NJ = 22;

function pose(o = {}) {
  const p = new Float32Array(NJ);
  p.set([0.25, -0.02, 0.05], I.lHand);
  p.set([-0.25, -0.02, 0.05], I.rHand);
  p.set([0, -1, 0.1], I.bat);
  for (const k in o) {
    const v = o[k];
    if (Array.isArray(v)) p.set(v, I[k]);
    else p[I[k]] = v;
  }
  return p;
}

function mixInto(out, a, b, t) {
  for (let i = 0; i < NJ; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  return out;
}

const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const v3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

// Piecewise-linear lookup over [[s, value], ...]; value is a number or a 3-vector.
function key(keys, s) {
  if (s <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (s <= keys[i][0]) {
      const t = (s - keys[i - 1][0]) / (keys[i][0] - keys[i - 1][0]);
      const a = keys[i - 1][1];
      const b = keys[i][1];
      return typeof a === 'number' ? lerp(a, b, t) : v3(a, b, t);
    }
  }
  return keys[keys.length - 1][1];
}

// Bat direction from an angle in the vertical plane through the bowler (0 = straight down,
// positive swings toward the bowler) and a tilt toward the off side.
function batAngle(deg, out = 0) {
  const b = (deg * PI) / 180;
  const c = Math.cos(out);
  return [Math.sin(b) * c, -Math.cos(b) * c, Math.sin(out)];
}

// ---- static poses ----------------------------------------------------------------------

const STAND = pose({});
const READY = pose({
  lean: 0.5, headX: -0.35, lHand: [0.18, -0.05, 0.42], rHand: [-0.18, -0.05, 0.42],
  lHipX: -0.75, lKnee: 1.0, rHipX: -0.75, rKnee: 1.0, lHipZ: 0.2, rHipZ: -0.2,
});
const CROUCH = pose({
  lean: 0.55, headX: -0.4, lHand: [0.08, -0.16, 0.42], rHand: [-0.08, -0.16, 0.42],
  lHipX: -1.75, lKnee: 2.45, rHipX: -1.75, rKnee: 2.45, lHipZ: 0.35, rHipZ: -0.35,
});
const CATCH_HIGH = pose({ headX: -0.45, lean: -0.08, lHand: [0.1, 0.98, 0.2], rHand: [-0.1, 0.98, 0.2] });
const PICK = pose({
  lean: 1.25, headX: -0.5, lHand: [0.1, -0.4, 0.45], rHand: [-0.1, -0.4, 0.45],
  lHipX: -1.5, lKnee: 2.0, rHipX: -1.3, rKnee: 1.9, lHipZ: 0.15, rHipZ: -0.15,
});
const CELEBRATE = pose({ headX: -0.3, lean: -0.1, lHand: [0.4, 1.0, 0], rHand: [-0.4, 1.0, 0] });
const THROW = [
  [0, pose({ twist: -0.7, lean: -0.1, rHand: [-0.45, 0.75, -0.25], lHand: [0.3, 0.5, 0.35], lHipX: -0.45, rHipX: 0.2, lKnee: 0.2 })],
  [0.5, pose({ twist: 0.4, lean: 0.3, rHand: [-0.15, 0.98, 0.35], lHand: [0.3, 0.2, 0.1], lHipX: -0.5, lKnee: 0.3, rHipX: 0.35, rKnee: 0.3 })],
  [1, pose({ twist: 0.85, lean: 0.55, rHand: [0.15, 0.1, 0.5], lHand: [0.3, 0.1, -0.2], lHipX: -0.4, lKnee: 0.4, rHipX: -0.3, rKnee: 0.7 })],
];
const UMPIRE = {
  stand: pose({ lHand: [0.12, -0.05, -0.16], rHand: [-0.12, -0.05, -0.16] }),
  out: pose({ rHand: [-0.2, 1.06, 0.14] }),
  six: pose({ lHand: [0.22, 1.06, 0.05], rHand: [-0.22, 1.06, 0.05] }),
  wide: pose({ lHand: [0.8, 0.5, 0], rHand: [-0.8, 0.5, 0] }),
  noball: pose({ rHand: [-0.8, 0.5, 0] }),
  fourA: pose({ rHand: [-0.55, 0.42, 0.3] }),
  fourB: pose({ rHand: [0.12, 0.42, 0.45] }),
};

// ---- batting (authored for a right-hander; left-handers are mirrored) ----------------------
// Body frame: +x toward the bowler, +z toward the off side.

const STANCE_LEGS = { lHipX: -0.42, lKnee: 0.68, rHipX: -0.42, rKnee: 0.68, lHipZ: 0.22, rHipZ: -0.22 };
const BAT_STANCE = pose({
  ...STANCE_LEGS, lean: 0.45, headY: 1.15, headX: -0.12,
  rHand: [-0.1, -0.05, 0.3], bat: [-0.38, -0.92, 0.05],
});
const BAT_BACKLIFT = pose({
  ...STANCE_LEGS, lean: 0.4, headY: 1.15, headX: -0.12,
  rHand: [-0.3, 0.2, 0.28], bat: batAngle(-140, 0.12),
});
const BAT_IDLE = pose({ rHand: [-0.3, 0.0, 0.1], bat: [0.05, -1, 0.12], lHand: [0.25, -0.02, 0.05] });
const BAT_LEAVE = pose({
  ...STANCE_LEGS, lean: 0.25, twist: -0.12, headY: 1.0, hipY: 0.02,
  rHand: [-0.28, 0.42, 0.2], bat: [-0.25, 0.95, -0.1],
});

function swingPose(fn) {
  const steps = 16;
  const clip = [];
  for (let i = 0; i <= steps; i++) clip.push([i / steps, pose(fn(i / steps))]);
  return clip;
}

const SWINGS = {
  drive: swingPose((s) => ({
    headY: lerp(1.15, 0.85, s), headX: -0.1,
    lean: key([[0, 0.4], [0.5, 0.58], [0.75, 0.42], [1, 0.22]], s),
    twist: key([[0, 0], [0.5, 0.35], [0.75, 0.7], [1, 0.95]], s),
    rHand: key([[0, [-0.3, 0.2, 0.28]], [0.3, [-0.1, 0.02, 0.32]], [0.5, [0.26, -0.06, 0.34]], [0.72, [0.42, 0.25, 0.3]], [1, [0.3, 0.66, 0.2]]], s),
    bat: batAngle(key([[0, -140], [0.3, -85], [0.5, -6], [0.72, 95], [1, 205]], s), 0.06),
    lHipZ: key([[0, 0.22], [0.5, 0.6], [1, 0.6]], s), lHipX: key([[0, -0.42], [0.5, -0.6], [1, -0.55]], s), lKnee: key([[0, 0.68], [0.5, 0.8], [1, 0.75]], s),
    rHipZ: key([[0, -0.22], [0.5, -0.12], [1, -0.1]], s), rHipX: key([[0, -0.42], [0.5, 0.12], [1, 0.2]], s), rKnee: key([[0, 0.68], [0.5, 0.4], [1, 0.5]], s),
  })),
  backDrive: swingPose((s) => ({
    headY: lerp(1.15, 0.9, s), headX: -0.1, hipY: key([[0, 0], [0.5, 0.05], [1, 0.02]], s),
    lean: key([[0, 0.38], [0.5, 0.3], [1, 0.18]], s),
    twist: key([[0, 0], [0.5, 0.3], [1, 0.7]], s),
    rHand: key([[0, [-0.3, 0.2, 0.28]], [0.3, [-0.12, 0.18, 0.32]], [0.5, [0.16, 0.08, 0.34]], [0.75, [0.36, 0.3, 0.3]], [1, [0.32, 0.52, 0.24]]], s),
    bat: batAngle(key([[0, -140], [0.3, -80], [0.5, -4], [0.75, 80], [1, 150]], s), 0.06),
    lHipZ: 0.2, lHipX: key([[0, -0.42], [0.5, -0.25], [1, -0.25]], s), lKnee: key([[0, 0.68], [0.5, 0.3], [1, 0.3]], s),
    rHipZ: -0.26, rHipX: key([[0, -0.42], [0.5, -0.12], [1, -0.1]], s), rKnee: key([[0, 0.68], [0.5, 0.2], [1, 0.2]], s),
  })),
  loft: swingPose((s) => ({
    headY: lerp(1.15, 0.7, s), headX: key([[0, -0.1], [1, -0.35]], s),
    lean: key([[0, 0.4], [0.5, 0.42], [1, -0.05]], s),
    twist: key([[0, -0.1], [0.5, 0.4], [1, 1.1]], s),
    rHand: key([[0, [-0.34, 0.3, 0.26]], [0.3, [-0.14, 0.06, 0.32]], [0.5, [0.24, -0.02, 0.34]], [0.72, [0.42, 0.36, 0.28]], [1, [0.2, 0.8, 0.12]]], s),
    bat: batAngle(key([[0, -150], [0.3, -90], [0.5, 10], [0.72, 110], [1, 235]], s), 0.05),
    lHipZ: key([[0, 0.22], [0.5, 0.55], [1, 0.5]], s), lHipX: key([[0, -0.42], [0.5, -0.5], [1, -0.3]], s), lKnee: key([[0, 0.68], [0.5, 0.6], [1, 0.35]], s),
    rHipZ: -0.16, rHipX: key([[0, -0.42], [0.5, 0.05], [1, 0.1]], s), rKnee: key([[0, 0.68], [0.5, 0.4], [1, 0.45]], s),
  })),
  pull: swingPose((s) => ({
    headY: lerp(1.15, 0.5, s), headX: -0.1, hipY: 0.02,
    lean: key([[0, 0.38], [0.5, 0.28], [1, 0.2]], s),
    twist: key([[0, 0], [0.5, 0.55], [0.75, 1.2], [1, 1.7]], s),
    rHand: key([[0, [-0.3, 0.22, 0.28]], [0.3, [-0.1, 0.36, 0.36]], [0.5, [0.15, 0.34, 0.42]], [0.75, [0.36, 0.38, 0.22]], [1, [0.32, 0.46, -0.05]]], s),
    bat: key([[0, [-0.6, 0.76, 0.12]], [0.3, [-0.72, 0.3, 0.62]], [0.5, [0.1, 0.02, 1]], [0.75, [0.95, 0.05, 0.2]], [1, [0.5, 0.25, -0.85]]], s),
    lHipZ: 0.25, lHipX: key([[0, -0.42], [0.5, -0.3], [1, -0.3]], s), lKnee: key([[0, 0.68], [0.5, 0.4], [1, 0.4]], s),
    rHipZ: -0.3, rHipX: key([[0, -0.42], [0.5, -0.2], [1, -0.15]], s), rKnee: key([[0, 0.68], [0.5, 0.35], [1, 0.3]], s),
  })),
  cut: swingPose((s) => ({
    headY: lerp(1.15, 0.6, s), headX: -0.15, hipY: 0.02,
    lean: key([[0, 0.38], [0.5, 0.42], [1, 0.3]], s),
    twist: key([[0, -0.1], [0.3, -0.28], [0.5, -0.05], [0.75, 0.4], [1, 0.7]], s),
    rHand: key([[0, [-0.3, 0.22, 0.28]], [0.3, [-0.26, 0.4, 0.34]], [0.5, [-0.02, 0.26, 0.5]], [0.75, [0.2, 0.3, 0.45]], [1, [0.34, 0.5, 0.2]]], s),
    bat: key([[0, [-0.6, 0.76, 0.12]], [0.3, [-0.62, 0.74, 0.25]], [0.5, [-0.25, -0.08, 0.96]], [0.75, [0.75, 0.1, 0.65]], [1, [0.6, 0.6, -0.4]]], s),
    lHipZ: 0.24, lHipX: -0.35, lKnee: 0.5, rHipZ: -0.3, rHipX: key([[0, -0.42], [0.5, -0.3], [1, -0.25]], s), rKnee: key([[0, 0.68], [0.5, 0.5], [1, 0.45]], s),
  })),
  glance: swingPose((s) => ({
    headY: lerp(1.15, 0.6, s), headX: -0.2,
    lean: key([[0, 0.4], [0.5, 0.5], [1, 0.42]], s),
    twist: key([[0, 0], [0.5, 0.5], [1, 0.9]], s),
    rHand: key([[0, [-0.3, 0.2, 0.28]], [0.5, [0.18, -0.02, 0.22]], [1, [0.25, 0.1, 0.06]]], s),
    bat: key([[0, batAngle(-140, 0.12)], [0.5, [-0.1, -0.97, -0.2]], [1, [0.3, -0.8, -0.5]]], s),
    lHipZ: key([[0, 0.22], [0.5, 0.42], [1, 0.42]], s), lHipX: -0.5, lKnee: 0.7, rHipZ: -0.18, rHipX: key([[0, -0.42], [1, -0.05]], s), rKnee: 0.45,
  })),
  lateCut: swingPose((s) => ({
    headY: lerp(1.15, 0.2, s), headX: -0.25,
    lean: key([[0, 0.4], [0.5, 0.52], [1, 0.5]], s),
    twist: key([[0, 0], [0.5, -0.3], [1, -0.45]], s),
    rHand: key([[0, [-0.3, 0.2, 0.28]], [0.5, [-0.12, 0.0, 0.42]], [1, [-0.2, -0.05, 0.5]]], s),
    bat: key([[0, batAngle(-140, 0.12)], [0.5, [-0.55, -0.6, 0.55]], [1, [-0.8, -0.45, 0.35]]], s),
    ...STANCE_LEGS,
  })),
  defendF: swingPose((s) => {
    const u = Math.min(1, s * 2);
    return {
      headY: lerp(1.15, 0.95, u), headX: -0.3, lean: lerp(0.4, 0.62, u), twist: lerp(0, 0.2, u),
      rHand: v3([-0.3, 0.2, 0.28], [0.22, -0.02, 0.33], u), bat: batAngle(lerp(-140, -13, u), 0.04),
      lHipZ: lerp(0.22, 0.5, u), lHipX: lerp(-0.42, -0.6, u), lKnee: lerp(0.68, 0.85, u),
      rHipZ: lerp(-0.22, -0.14, u), rHipX: lerp(-0.42, 0.05, u), rKnee: lerp(0.68, 0.4, u),
    };
  }),
  defendB: swingPose((s) => {
    const u = Math.min(1, s * 2);
    return {
      headY: lerp(1.15, 1.0, u), headX: -0.3, lean: lerp(0.4, 0.3, u), hipY: lerp(0, 0.04, u), twist: lerp(0, 0.12, u),
      rHand: v3([-0.3, 0.2, 0.28], [0.02, 0.16, 0.33], u), bat: batAngle(lerp(-140, -7, u), 0.03),
      lHipZ: 0.2, lHipX: lerp(-0.42, -0.2, u), lKnee: lerp(0.68, 0.25, u),
      rHipZ: -0.24, rHipX: lerp(-0.42, -0.12, u), rKnee: lerp(0.68, 0.2, u),
    };
  }),
};

const DEG = PI / 180;

// Which swing to play for a shot. phi: 0 straight, + toward the off side (radians).
export function swingFor(phi, foot, loft, defend) {
  if (defend) return { clip: foot === 'back' ? 'defendB' : 'defendF', yaw: 0, stride: foot === 'back' ? -0.25 : 0.4 };
  const a = phi / DEG;
  const stride = foot === 'back' ? -0.3 : 0.45;
  if (a > 125) return { clip: 'lateCut', yaw: 0, stride: -0.1 };
  if (a < -125) return { clip: 'glance', yaw: 0.3, stride };
  if (a > 60) return foot === 'back' || loft ? { clip: 'cut', yaw: clamp(-(a - 90) * DEG, -0.4, 0.4), stride } : { clip: 'drive', yaw: -1.05, stride };
  if (a < -60) return { clip: 'pull', yaw: clamp(-(a + 90) * DEG * 0.6, -0.3, 0.5), stride };
  return { clip: loft ? 'loft' : foot === 'back' ? 'backDrive' : 'drive', yaw: -a * DEG, stride };
}

// ---- bowling (right arm; left-armers are mirrored). Body faces the batter (+z). -------------

const BOWL_RELEASE = 0.62;
function armCircle(deg, x = -0.22) {
  const a = deg * DEG;
  return [x, 0.5 + 0.55 * Math.cos(a), 0.55 * Math.sin(a)];
}
const BOWL = swingPose((u) => ({
  hipY: key([[0, 0.02], [0.18, 0.14], [0.36, 0], [1, 0]], u),
  lean: key([[0, 0.05], [0.3, -0.18], [0.62, 0.42], [1, 0.8]], u),
  twist: key([[0, -0.5], [0.3, -1.05], [0.62, 0.05], [1, 0.75]], u),
  headY: key([[0, 0.4], [0.3, 0.9], [0.62, 0], [1, -0.3]], u),
  rHand: key([
    [0, [-0.1, 0.55, 0.25]], [0.16, [-0.14, 0.3, 0.2]], [0.3, armCircle(190)], [0.4, armCircle(245)], [0.48, armCircle(290)],
    [0.56, armCircle(335)], [0.62, armCircle(362)], [0.7, armCircle(410)], [0.8, armCircle(470, -0.1)], [1, [0.15, 0.0, 0.3]],
  ], u),
  lHand: key([[0, [0.15, 0.6, 0.3]], [0.3, [0.16, 1.02, 0.28]], [0.5, [0.2, 0.75, 0.4]], [0.62, [0.22, 0.12, 0.15]], [1, [0.28, 0.05, -0.22]]], u),
  lHipX: key([[0, -0.3], [0.2, -0.9], [0.36, -1.0], [0.55, -0.6], [0.62, -0.5], [1, 0.3]], u),
  lKnee: key([[0, 0.5], [0.2, 0.9], [0.36, 0.6], [0.55, 0.05], [0.62, 0.08], [1, 0.5]], u),
  rHipX: key([[0, 0.2], [0.2, -0.2], [0.36, 0.0], [0.62, 0.55], [0.8, -0.5], [1, -0.85]], u),
  rKnee: key([[0, 0.6], [0.2, 0.7], [0.36, 0.15], [0.62, 0.7], [0.8, 1.1], [1, 0.9]], u),
}));

function sampleClip(clip, s, out) {
  const n = clip.length - 1;
  const f = clamp(s, 0, 1) * n;
  const i = Math.min(n - 1, Math.floor(f));
  return mixInto(out, clip[i][1], clip[i + 1][1], f - i);
}

function sampleKeys(clip, s, out) {
  if (s <= clip[0][0]) return mixInto(out, clip[0][1], clip[0][1], 0);
  for (let i = 1; i < clip.length; i++) {
    if (s <= clip[i][0]) return mixInto(out, clip[i - 1][1], clip[i][1], smooth((s - clip[i - 1][0]) / (clip[i][0] - clip[i - 1][0])));
  }
  const last = clip[clip.length - 1][1];
  return mixInto(out, last, last, 0);
}

function runPose(out, phase, hasBat) {
  const s = Math.sin(phase);
  const c = Math.cos(phase);
  out.fill(0);
  out[I.lean] = 0.3;
  out[I.lHipX] = -0.2 - 0.85 * s;
  out[I.rHipX] = -0.2 + 0.85 * s;
  out[I.lKnee] = 0.45 + 0.8 * Math.max(0, c);
  out[I.rKnee] = 0.45 + 0.8 * Math.max(0, -c);
  out.set([0.26, 0.1 + 0.08 * Math.abs(s), 0.1 - 0.3 * s], I.lHand);
  out.set([-0.26, 0.1 + 0.08 * Math.abs(s), 0.1 + 0.3 * s], I.rHand);
  if (hasBat) {
    out.set([-0.28, 0.02, 0.18 + 0.12 * s], I.rHand);
    out.set([0, -0.45, 0.9], I.bat);
  } else {
    out.set([0, -1, 0.1], I.bat);
  }
  return out;
}

// ---- rig ---------------------------------------------------------------------------------

const SKINS = [0xf1c9a5, 0xd9a273, 0xb97c50, 0x8a5a38, 0x6b432a, 0xe3b48e];
const geoCache = {};
function capsule(r, len) {
  const k = r + ':' + len;
  if (!geoCache[k]) {
    const g = new THREE.CapsuleGeometry(r, Math.max(0.01, len - 2 * r), 2, 7);
    g.translate(0, -len / 2, 0);
    geoCache[k] = g;
  }
  return geoCache[k];
}
function shared(name, make) {
  if (!geoCache[name]) geoCache[name] = make();
  return geoCache[name];
}

const WHITE = new THREE.MeshLambertMaterial({ color: 0xf2f2ee });
const DARK = new THREE.MeshLambertMaterial({ color: 0x20242c });
const WOOD = new THREE.MeshLambertMaterial({ color: 0xdcb98a });
const SHADOW = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28, depthWrite: false });
const skinMats = SKINS.map((c) => new THREE.MeshLambertMaterial({ color: c }));

export function makeKit() {
  return {
    shirt: new THREE.MeshLambertMaterial({ color: 0xffffff }),
    trim: new THREE.MeshLambertMaterial({ color: 0xffffff }),
    pants: new THREE.MeshLambertMaterial({ color: 0xffffff }),
  };
}
export function setKit(kit, team) {
  kit.shirt.color.setHex(team.shirt);
  kit.trim.color.setHex(team.trim);
  kit.pants.color.setHex(team.pants);
}

const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _d = new THREE.Vector3();
const _m = new THREE.Matrix4();
const DOWN = new THREE.Vector3(0, -1, 0);

export class Rig {
  constructor({ kit, skin = 0, bat = false, pads = false, helmet = false, gloves = false, umpire = false }) {
    const skinMat = skinMats[skin % skinMats.length];
    const shirt = umpire ? WHITE : kit.shirt;
    const pants = umpire ? DARK : kit.pants;
    const trim = umpire ? DARK : kit.trim;
    const handMat = gloves ? WHITE : skinMat;

    this.root = new THREE.Group();
    this.flip = new THREE.Group();
    this.root.add(this.flip);
    this.body = new THREE.Group();
    this.flip.add(this.body);

    const shadow = new THREE.Mesh(shared('shadow', () => new THREE.CircleGeometry(0.42, 14).rotateX(-PI / 2)), SHADOW);
    shadow.position.y = 0.025;
    shadow.renderOrder = 1;
    this.root.add(shadow);

    const pelvis = new THREE.Mesh(shared('pelvis', () => new THREE.BoxGeometry(0.32, 0.17, 0.2)), pants);
    this.body.add(pelvis);

    this.spine = new THREE.Group();
    this.spine.position.y = SPINE_Y;
    this.body.add(this.spine);
    const torso = new THREE.Mesh(shared('torso', () => new THREE.BoxGeometry(0.4, 0.5, 0.23).translate(0, 0.28, 0)), shirt);
    this.spine.add(torso);
    const stripe = new THREE.Mesh(shared('stripe', () => new THREE.BoxGeometry(0.405, 0.07, 0.235).translate(0, 0.4, 0)), trim);
    this.spine.add(stripe);

    this.head = new THREE.Group();
    this.head.position.y = 0.56;
    this.spine.add(this.head);
    const skull = new THREE.Mesh(shared('head', () => new THREE.SphereGeometry(0.105, 12, 9).translate(0, 0.12, 0)), skinMat);
    this.head.add(skull);
    if (helmet) {
      const h = new THREE.Mesh(shared('helmet', () => new THREE.SphereGeometry(0.128, 12, 8, 0, PI * 2, 0, PI * 0.62).translate(0, 0.125, 0)), trim);
      this.head.add(h);
      const grill = new THREE.Mesh(shared('grill', () => new THREE.BoxGeometry(0.16, 0.07, 0.02).translate(0, 0.08, 0.13)), DARK);
      this.head.add(grill);
    } else {
      const capMat = umpire ? WHITE : trim;
      const cap = new THREE.Mesh(shared('cap', () => new THREE.SphereGeometry(0.112, 12, 6, 0, PI * 2, 0, PI * 0.5).translate(0, 0.135, 0)), capMat);
      this.head.add(cap);
      const brim = new THREE.Mesh(shared('brim', () => new THREE.BoxGeometry(0.15, 0.015, 0.11).translate(0, 0.14, 0.13)), capMat);
      this.head.add(brim);
    }

    // Arms are placed by IK each frame, so they hang off the body group rather than each other.
    const arm = () => {
      const upper = new THREE.Group();
      upper.add(new THREE.Mesh(capsule(0.048, UPPER), shirt));
      const fore = new THREE.Group();
      fore.add(new THREE.Mesh(capsule(0.04, FORE), skinMat));
      const hand = new THREE.Mesh(shared('hand', () => new THREE.SphereGeometry(gloves ? 0.06 : 0.047, 8, 6)), handMat);
      hand.position.y = -FORE;
      fore.add(hand);
      this.body.add(upper, fore);
      return { upper, fore, hand };
    };
    this.armL = arm();
    this.armR = arm();

    const leg = (x) => {
      const hip = new THREE.Group();
      hip.position.set(x, 0, 0);
      hip.add(new THREE.Mesh(capsule(0.075, THIGH), pants));
      const knee = new THREE.Group();
      knee.position.y = -THIGH;
      knee.add(new THREE.Mesh(capsule(0.058, SHIN), pants));
      if (pads) knee.add(new THREE.Mesh(shared('pad', () => new THREE.BoxGeometry(0.15, 0.52, 0.09).translate(0, -0.2, 0.05)), WHITE));
      const foot = new THREE.Mesh(shared('foot', () => new THREE.BoxGeometry(0.1, 0.07, 0.25).translate(0, -SHIN + 0.01, 0.06)), WHITE);
      knee.add(foot);
      hip.add(knee);
      this.body.add(hip);
      return { hip, knee };
    };
    this.legL = leg(0.095);
    this.legR = leg(-0.095);

    this.bat = null;
    if (bat) {
      this.bat = new THREE.Group();
      const handle = new THREE.Mesh(shared('handle', () => new THREE.CylinderGeometry(0.017, 0.017, 0.3, 6).translate(0, -0.03, 0)), DARK);
      const blade = new THREE.Mesh(shared('blade', () => new THREE.BoxGeometry(0.108, 0.56, 0.04).translate(0, -0.46, 0)), WOOD);
      this.bat.add(handle, blade);
      this.body.add(this.bat);
    }

    this.pose = pose();
    this.setMirror(false);
    this.apply(this.pose);
  }

  setMirror(m) {
    this.mirror = m;
    this.flip.scale.x = m ? -1 : 1;
  }

  blend(target, k) {
    const p = this.pose;
    for (let i = 0; i < NJ; i++) p[i] += (target[i] - p[i]) * k;
    this.apply(p);
  }

  solveArm(arm, side, tx, ty, tz) {
    // Shoulder in the body frame.
    _a.set(side * SH_X, SH_Y, 0).applyQuaternion(this.spine.quaternion);
    _a.y += SPINE_Y;
    _b.set(tx, ty, tz).sub(_a);
    let d = _b.length();
    const max = UPPER + FORE - 0.005;
    if (d > max) {
      _b.multiplyScalar(max / d);
      d = max;
    } else if (d < 0.08) {
      _b.set(0, -0.08, 0);
      d = 0.08;
    }
    const cosA = clamp((UPPER * UPPER + d * d - FORE * FORE) / (2 * UPPER * d), -1, 1);
    const sinA = Math.sqrt(1 - cosA * cosA);
    _c.copy(_b).multiplyScalar(1 / d);
    // Elbow bends toward a pole pointing down, out and a little back.
    _d.set(side * 0.45, -1, -0.5);
    _d.addScaledVector(_c, -_d.dot(_c));
    if (_d.lengthSq() < 1e-6) _d.set(side, 0, 0);
    _d.normalize();
    const ex = _c.x * cosA + _d.x * sinA;
    const ey = _c.y * cosA + _d.y * sinA;
    const ez = _c.z * cosA + _d.z * sinA;
    arm.upper.position.copy(_a);
    _c.set(ex, ey, ez);
    arm.upper.quaternion.setFromUnitVectors(DOWN, _c);
    const elx = _a.x + ex * UPPER;
    const ely = _a.y + ey * UPPER;
    const elz = _a.z + ez * UPPER;
    arm.fore.position.set(elx, ely, elz);
    _c.set(_a.x + _b.x - elx, _a.y + _b.y - ely, _a.z + _b.z - elz).normalize();
    arm.fore.quaternion.setFromUnitVectors(DOWN, _c);
  }

  apply(p) {
    const ext = (hx, hz, k) => Math.cos(hz) * (THIGH * Math.cos(hx) + SHIN * Math.cos(hx + k));
    const ground = Math.max(ext(p[I.lHipX], p[I.lHipZ], p[I.lKnee]), ext(p[I.rHipX], p[I.rHipZ], p[I.rKnee]));
    this.body.position.y = ground + 0.04 + p[I.hipY];
    this.body.rotation.y = p[I.yaw];
    _e.set(p[I.lean], p[I.twist], p[I.side], 'YXZ');
    this.spine.quaternion.setFromEuler(_e);
    this.head.rotation.set(p[I.headX], p[I.headY] - p[I.twist] * 0.6, 0, 'YXZ');
    this.legL.hip.rotation.set(p[I.lHipX], 0, p[I.lHipZ]);
    this.legL.knee.rotation.x = p[I.lKnee];
    this.legR.hip.rotation.set(p[I.rHipX], 0, p[I.rHipZ]);
    this.legR.knee.rotation.x = p[I.rKnee];

    let lx = p[I.lHand];
    let ly = p[I.lHand + 1];
    let lz = p[I.lHand + 2];
    const rx = p[I.rHand];
    const ry = p[I.rHand + 1];
    const rz = p[I.rHand + 2];
    if (this.bat) {
      _a.set(p[I.bat], p[I.bat + 1], p[I.bat + 2]).normalize();
      // Top hand sits just above the bottom hand on the handle.
      lx = rx - _a.x * 0.1;
      ly = ry - _a.y * 0.1;
      lz = rz - _a.z * 0.1;
      _b.copy(_a).negate();
      _c.set(1, 0, 0.15);
      _c.addScaledVector(_b, -_c.dot(_b));
      if (_c.lengthSq() < 1e-4) _c.set(0, 1, 0).addScaledVector(_b, -_b.y);
      _c.normalize();
      _d.crossVectors(_b, _c);
      _m.makeBasis(_d, _b, _c);
      this.bat.quaternion.setFromRotationMatrix(_m);
      this.bat.position.set(rx, ry, rz);
    }
    this.solveArm(this.armL, 1, lx, ly, lz);
    this.solveArm(this.armR, -1, rx, ry, rz);
  }
}

// ---- actor: a rig plus position, movement and a current animation -------------------------

const tmp = pose();
const tmp2 = pose();

export class Actor {
  constructor(rig, hasBat = false) {
    this.rig = rig;
    this.hasBat = hasBat;
    this.x = 0;
    this.z = 0;
    this.heading = 0;
    this.tx = 0;
    this.tz = 0;
    this.speed = 6;
    this.moving = false;
    this.anim = 'stand';
    this.t = 0;
    this.s = 0;
    this.phase = Math.random() * 6;
    this.look = null;
    this.swing = null;
    this.reach = null;
    this.offset = 0;
    this.stiff = 14;
  }

  place(x, z, heading) {
    this.x = this.tx = x;
    this.z = this.tz = z;
    if (heading !== undefined) this.heading = heading;
    this.moving = false;
  }

  goTo(x, z, speed = 6.5) {
    this.tx = x;
    this.tz = z;
    this.speed = speed;
    this.moving = Math.hypot(x - this.x, z - this.z) > 0.05;
  }

  set(anim, s = 0) {
    if (this.anim !== anim) this.t = 0;
    this.anim = anim;
    this.s = s;
  }

  // World-space point the hands should reach for (a ball), or null.
  reachFor(p) {
    this.reach = p;
  }

  update(dt) {
    this.t += dt;
    const rig = this.rig;
    let target = STAND;
    let stiff = this.stiff;

    if (this.moving) {
      const dx = this.tx - this.x;
      const dz = this.tz - this.z;
      const d = Math.hypot(dx, dz);
      const step = this.speed * dt;
      if (d <= step) {
        this.x = this.tx;
        this.z = this.tz;
        this.moving = false;
      } else {
        this.x += (dx / d) * step;
        this.z += (dz / d) * step;
        this.turn(Math.atan2(dx, dz), dt, 14);
        this.phase += dt * this.speed * 2.6;
        target = runPose(tmp, this.phase, this.hasBat);
        stiff = 20;
      }
    }

    if (!this.moving) {
      switch (this.anim) {
        case 'ready': target = READY; break;
        case 'crouch': target = CROUCH; break;
        case 'catch': target = CATCH_HIGH; break;
        case 'pick': target = PICK; break;
        case 'celebrate': target = CELEBRATE; tmp.set(CELEBRATE); tmp[I.hipY] = Math.abs(Math.sin(this.t * 7)) * 0.12; target = tmp; break;
        case 'throw': target = sampleKeys(THROW, Math.min(1, this.t / 0.42), tmp); stiff = 30; break;
        case 'bowl': target = sampleClip(BOWL, this.s, tmp); stiff = 60; break;
        case 'jog': this.phase += dt * 9; target = runPose(tmp, this.phase, this.hasBat); stiff = 20; break;
        case 'stance': target = BAT_STANCE; break;
        case 'backlift': target = BAT_BACKLIFT; stiff = 10; break;
        case 'leave': target = BAT_LEAVE; stiff = 12; break;
        case 'batIdle': target = BAT_IDLE; break;
        case 'swing': {
          target = sampleClip(SWINGS[this.swing.clip], this.s, tmp);
          target[I.yaw] = this.swing.yaw * smooth(Math.min(1, this.s * 2.2));
          stiff = 45;
          break;
        }
        case 'out': target = UMPIRE.out; break;
        case 'six': target = UMPIRE.six; break;
        case 'wide': target = UMPIRE.wide; break;
        case 'noball': target = UMPIRE.noball; break;
        case 'four': target = mixInto(tmp, UMPIRE.fourA, UMPIRE.fourB, 0.5 + 0.5 * Math.sin(this.t * 9)); break;
        case 'umpire': target = UMPIRE.stand; break;
        default: target = STAND;
      }
      if (this.look && this.anim !== 'bowl' && !this.hasBat) this.turn(Math.atan2(this.look[0] - this.x, this.look[1] - this.z), dt, 8);
    }

    if (this.reach) {
      // Hands toward a world point, expressed in the body frame.
      tmp2.set(target);
      const cs = Math.cos(-this.heading);
      const sn = Math.sin(-this.heading);
      const wx = this.reach[0] - this.x;
      const wz = this.reach[2] - this.z;
      let bx = wx * cs + wz * sn;
      const bz = -wx * sn + wz * cs;
      if (rig.mirror) bx = -bx;
      const by = this.reach[1] - rig.body.position.y;
      tmp2.set([bx + 0.07, by, bz], I.lHand);
      tmp2.set([bx - 0.07, by, bz], I.rHand);
      target = tmp2;
      stiff = 26;
    }

    rig.blend(target, 1 - Math.exp(-dt * stiff));
    rig.root.position.set(this.x, 0, this.z);
    rig.root.rotation.y = this.heading;
  }

  turn(want, dt, rate) {
    let diff = want - this.heading;
    while (diff > PI) diff -= 2 * PI;
    while (diff < -PI) diff += 2 * PI;
    this.heading += diff * (1 - Math.exp(-dt * rate));
  }
}

export { BOWL_RELEASE };
