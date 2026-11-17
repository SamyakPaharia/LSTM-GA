// Bowling model: aim + meter + bowler → a piecewise-parabolic ball trajectory.

import { G, BALL_R, HALF, BOWLER_CREASE_Z, clamp, lerp, offSignFor } from './constants.js';

export const PACE_TYPES = [
  { id: 'stock', name: 'Seam up' },
  { id: 'out', name: 'Outswinger' },
  { id: 'in', name: 'Inswinger' },
  { id: 'cutter', name: 'Cutter' },
  { id: 'slower', name: 'Slower ball' },
];

export const SPIN_TYPES = [
  { id: 'stock', name: 'Stock ball' },
  { id: 'arm', name: 'Arm ball' },
  { id: 'wrong', name: "Wrong 'un" },
  { id: 'flight', name: 'Flighted' },
];

export const AIM_LIMITS = { x: 1.25, zMin: -2.2, zMax: 9.5 };
export const METER = { sweetLo: 0.8, sweetHi: 0.95, noBall: 0.97 };

export const isSpin = (style) => style === 'OS' || style === 'LS' || style === 'SLA';
export const bowlingArm = (style) => (style === 'LF' || style === 'LFM' || style === 'SLA' ? 'L' : 'R');
export const typesFor = (style) => (isSpin(style) ? SPIN_TYPES : PACE_TYPES);

export function styleName(style) {
  return {
    RF: 'Right-arm fast', RFM: 'Right-arm fast-medium', RM: 'Right-arm medium', LF: 'Left-arm fast',
    LFM: 'Left-arm fast-medium', OS: 'Off spin', LS: 'Leg spin', SLA: 'Left-arm spin',
  }[style] || 'Part-time';
}

export function topSpeedKmh(style, rating) {
  if (style === 'RF' || style === 'LF') return 139 + rating * 0.09;
  if (style === 'RFM' || style === 'LFM') return 130 + rating * 0.07;
  if (isSpin(style)) return 82 + rating * 0.06;
  return 119 + rating * 0.06;
}

// A player with no listed style can still be asked to bowl: gentle medium pace.
const effectiveStyle = (p) => p.style || 'RM';

export function makeDelivery({ bowler, type, aim, effort, strikerHand, rng }) {
  const style = effectiveStyle(bowler);
  const spin = isSpin(style);
  const off = offSignFor(strikerHand);
  const skill = clamp(bowler.bowl / 100, 0.05, 1);
  const e = clamp(effort, 0, 1);
  const types = typesFor(style);
  if (!types.some((t) => t.id === type)) type = 'stock';

  const noBall = e > METER.noBall;
  const meterAcc = e > METER.sweetHi ? 0.6 : e >= METER.sweetLo ? 1 : 0.45 + 0.55 * (e / METER.sweetLo);
  const acc = (0.3 + 0.7 * skill) * meterAcc;
  const sx = 0.05 + (1 - acc) * 0.34;
  const sz = 0.18 + (1 - acc) * 1.1;
  const bx = clamp(aim.x + rng.gauss() * sx, -1.9, 1.9);
  const bz = clamp(aim.z + rng.gauss() * sz, -3.6, 10.7);

  let pace = lerp(0.86, 1, e);
  if (!spin && type === 'slower') pace *= 0.8;
  if (spin && type === 'flight') pace *= 0.9;
  if (spin && type === 'arm') pace *= 1.1;
  const kmh = topSpeedKmh(style, bowler.bowl) * pace * (1 + rng.gauss() * 0.01);
  const s = kmh / 3.6;

  const arm = bowlingArm(style);
  const rx = arm === 'R' ? 0.45 : -0.45;
  const ry = spin ? 2.0 : 2.15;
  const rz = BOWLER_CREASE_Z + 0.25;

  const dx = bx - rx;
  const dz = bz - rz;
  const t1 = Math.sqrt(dx * dx + dz * dz) / s;

  let ax = 0;
  if (!spin) {
    const sw = 4 + 5 * skill;
    if (type === 'out') ax = off * sw;
    else if (type === 'in') ax = -off * sw;
    else if (type === 'stock') ax = rng.gauss() * 0.7;
  } else if (type === 'flight' || type === 'stock') {
    ax = rng.gauss() * 0.5;
  }

  const vx = (dx - 0.5 * ax * t1 * t1) / t1;
  const vz = dz / t1;
  const vy = (BALL_R - ry + 0.5 * G * t1 * t1) / t1;

  const segs = [{ t0: 0, p: [rx, ry, rz], v: [vx, vy, vz], ax }];

  // First bounce: lose pace, rebound, take the sideways kick.
  let bvx = (vx + ax * t1) * 0.7;
  let bvz = vz * (spin ? 0.74 : 0.78);
  let bvy = -(vy - G * t1) * (spin ? 0.5 : 0.55);
  if (!spin) {
    if (type === 'cutter') bvx += -off * (0.9 + 0.8 * skill);
    else bvx += rng.gauss() * 0.25;
  } else {
    const turn = style === 'OS' ? -1 : 1;
    const amount = 0.9 + 1.1 * skill;
    if (type === 'stock') bvx += turn * amount;
    else if (type === 'wrong') bvx += -turn * amount * 0.7;
    else if (type === 'flight') bvx += turn * amount * 1.25;
    else bvx += turn * 0.15;
  }

  let t = t1;
  let px = bx;
  let pz = bz;
  for (let i = 0; i < 3; i++) {
    segs.push({ t0: t, p: [px, BALL_R, pz], v: [bvx, bvy, bvz], ax: 0 });
    if (pz > HALF + 14) break;
    const hang = (2 * bvy) / G;
    t += hang;
    px += bvx * hang;
    pz += bvz * hang;
    bvy *= 0.5;
    bvx *= 0.82;
    bvz *= 0.82;
  }

  return { type, kmh: Math.round(kmh * 10) / 10, noBall, spin, arm, bounce: [bx, bz], segs };
}

function segAt(d, t) {
  let s = d.segs[0];
  for (let i = 1; i < d.segs.length; i++) {
    if (t >= d.segs[i].t0) s = d.segs[i];
    else break;
  }
  return s;
}

export function ballAt(d, t, out = [0, 0, 0]) {
  const s = segAt(d, t);
  const u = t - s.t0;
  out[0] = s.p[0] + s.v[0] * u + 0.5 * s.ax * u * u;
  out[1] = Math.max(BALL_R, s.p[1] + s.v[1] * u - 0.5 * G * u * u);
  out[2] = s.p[2] + s.v[2] * u;
  return out;
}

export function velAt(d, t, out = [0, 0, 0]) {
  const s = segAt(d, t);
  const u = t - s.t0;
  out[0] = s.v[0] + s.ax * u;
  out[1] = s.v[1] - G * u;
  out[2] = s.v[2];
  return out;
}

// Time at which the ball crosses the plane z. z velocity is positive in every segment.
export function timeAtZ(d, z) {
  for (let i = 0; i < d.segs.length; i++) {
    const s = d.segs[i];
    const next = d.segs[i + 1];
    if (!next || z <= next.p[2]) return s.t0 + (z - s.p[2]) / s.v[2];
  }
  return 0;
}

export function lengthName(d) {
  const z = d.bounce[1];
  if (d.spin) {
    if (z > 8.6) return 'yorker';
    if (z > 7.2) return 'full';
    if (z > 4.6) return 'good length';
    return 'short';
  }
  if (z > 8.2) return 'yorker';
  if (z > 6.2) return 'full';
  if (z > 3.0) return 'good length';
  if (z > 0.4) return 'short';
  return 'bouncer';
}

export function typeName(d) {
  const list = d.spin ? SPIN_TYPES : PACE_TYPES;
  return (list.find((t) => t.id === d.type) || list[0]).name;
}
