// Shot model: a delivery + the batter's input → miss, edge or hit with a launch vector.
// Pure function of its arguments, so the batting client can predict the host's answer.

import {
  HALF, STUMP_H, STUMP_HALF_W, BALL_R, STRIKER_CREASE_Z, FRONT_FOOT_Z, BACK_FOOT_Z, SWING_TIME,
  clamp, lerp, offSignFor,
} from './constants.js';
import { makeRng } from './rng.js';
import { ballAt, velAt, timeAtZ } from './delivery.js';

// window scales the timing windows for a human batter. "bot" is the window used whenever the
// computer bats; its skill at each difficulty comes from botSigma instead.
export const LEVELS = {
  easy: { window: 1.3, botSigma: 0.092, botAcc: 0.5, catchSkill: 0.86, botAgg: -0.08, margin: 0.6 },
  medium: { window: 0.86, botSigma: 0.072, botAcc: 0.72, catchSkill: 0.95, botAgg: 0, margin: 0.35 },
  hard: { window: 0.7, botSigma: 0.054, botAcc: 0.9, catchSkill: 1.0, botAgg: 0.06, margin: 0.3 },
  bot: { window: 1.0, botSigma: 0.072, botAcc: 0.72, catchSkill: 0.95, botAgg: 0, margin: 0.35 },
};

export const GUARD_LIMIT = 0.45;
export const WIDE_OFF = 1.0;
export const WIDE_LEG = 0.55;
export const WIDE_HIGH = 2.0;
const DEG = Math.PI / 180;

export const planeFor = (foot) => (foot === 'back' ? BACK_FOOT_Z : FRONT_FOOT_Z);
export const bodyXFor = (off, guard) => -off * 0.28 + (guard || 0);

// Ideal press time on the ball clock for a given footwork.
export function idealPressTime(d, foot) {
  return timeAtZ(d, planeFor(foot)) - SWING_TIME;
}

// How well the ball's line suits a shot played at angle phi (0 straight, + off side).
export function lineFit(rel, phi, defend) {
  const ideal = defend ? 0.2 : 0.22 + 0.3 * Math.sin(phi);
  const tol = defend ? 0.36 : 0.28;
  return clamp(1 - (Math.abs(rel - ideal) - tol) / 0.55, 0, 1);
}

// How well the ball's height at the contact plane suits the footwork.
export function lengthFit(d, foot, y, phi, defend) {
  const plane = planeFor(foot);
  if (foot === 'front') {
    const yorker = d.bounce[1] > plane - 0.8 && d.bounce[1] < plane + 1.2 && y < 0.25;
    if (yorker) return defend ? 0.9 : 0.5;
    if (y <= 0.85) return 1;
    if (y <= 1.25) return lerp(1, 0.35, (y - 0.85) / 0.4);
    return clamp(lerp(0.35, 0.05, (y - 1.25) / 0.4), 0, 0.35);
  }
  if (y < 0.3) return defend ? 0.6 : 0.35;
  if (y < 0.55) return lerp(defend ? 0.6 : 0.35, 1, (y - 0.3) / 0.25);
  if (y <= 1.45) return 1;
  const hook = phi < -60 * DEG && phi > -160 * DEG;
  return clamp(lerp(1, hook ? 0.6 : 0.2, (y - 1.45) / 0.45), 0, 1);
}

export function shotName(phi, foot, loft, defend, y, spin) {
  if (defend) return foot === 'back' ? 'back-foot defence' : 'forward defence';
  const a = phi / DEG;
  const m = Math.abs(a);
  const offSide = a >= 0;
  if (loft) {
    if (m < 28) return 'lofted straight drive';
    if (offSide) return m < 75 ? 'lofted drive over cover' : m < 125 ? 'upper cut' : 'ramp over third man';
    if (m < 75) return 'slog over midwicket';
    if (m < 125) return foot === 'back' ? (y > 1.35 ? 'hook' : 'pull') : spin ? 'slog sweep' : 'pick-up over square leg';
    return 'scoop over fine leg';
  }
  if (m < 25) return foot === 'back' ? 'back-foot punch' : 'straight drive';
  if (offSide) {
    if (m < 70) return foot === 'back' ? 'punch through cover' : 'cover drive';
    if (m < 115) return foot === 'back' ? 'square cut' : 'square drive';
    return 'late cut';
  }
  if (m < 70) return foot === 'back' ? 'whip through midwicket' : 'on-drive';
  if (m < 115) return foot === 'back' ? 'pull' : spin ? 'sweep' : 'flick';
  return 'leg glance';
}

function afterMiss(res, d, off, bodyX, rng, offered) {
  const ts = timeAtZ(d, HALF);
  const ps = ballAt(d, ts);
  const hitsStumps = Math.abs(ps[0]) <= STUMP_HALF_W + BALL_R && ps[1] <= STUMP_H + BALL_R * 0.5;
  const tc = timeAtZ(d, STRIKER_CREASE_Z);
  const pc = ballAt(d, tc);
  res.stumps = hitsStumps;
  if (hitsStumps) {
    const inLine = Math.abs(pc[0] - bodyX) < 0.32;
    const pPad = inLine ? (offered ? 0.45 : 0.6) : 0.08;
    if (rng() < pPad) {
      res.pad = true;
      const bounced = d.bounce[1] < STRIKER_CREASE_Z;
      const outsideLeg = bounced && d.bounce[0] * off < -(STUMP_HALF_W + 0.03);
      if (!outsideLeg && rng() < 0.85) res.dismissal = 'lbw';
    } else {
      res.dismissal = 'bowled';
    }
    return res;
  }
  const relStump = pc[0] * off;
  if (relStump > WIDE_OFF || relStump < -WIDE_LEG || pc[1] > WIDE_HIGH) res.wide = true;
  return res;
}

export function evaluateShot({ d, shot, batter, bowlerSkill = 0.8, level = 'medium', seed }) {
  const rng = makeRng(seed);
  const lv = LEVELS[level] || LEVELS.medium;
  const off = offSignFor(batter.hand);
  const guard = clamp(shot.guard || 0, -GUARD_LIMIT, GUARD_LIMIT);
  const bodyX = bodyXFor(off, guard);
  const res = {
    kind: 'none', timing: 'none', delta: 0, t: timeAtZ(d, BACK_FOOT_Z), pos: null, launch: null,
    name: '', dismissal: null, wide: false, pad: false, stumps: false, q: 0,
  };

  if (!shot || shot.kind === 'none') {
    res.pos = ballAt(d, res.t);
    return afterMiss(res, d, off, bodyX, rng, false);
  }

  const defend = shot.kind === 'defend';
  const foot = shot.foot === 'back' ? 'back' : 'front';
  const loft = !defend && !!shot.loft;
  const tc = timeAtZ(d, planeFor(foot));
  const p = ballAt(d, tc);
  const delta = shot.t - (tc - SWING_TIME);
  const skill = clamp(batter.bat / 100, 0.1, 1);

  let w = (0.8 + 0.4 * skill) * lv.window;
  if (defend) w *= 1.6;
  else if (loft) w *= 0.88;
  const Wp = 0.03 * w;
  const Wg = 0.065 * w;
  const Wo = 0.115 * w;
  const ad = Math.abs(delta);

  res.kind = 'miss';
  res.t = tc;
  res.pos = p;
  res.delta = delta;
  res.timing = ad <= Wp ? 'perfect' : ad <= Wg ? 'good' : ad <= Wo ? (delta < 0 ? 'early' : 'late') : 'missed';

  let dx = shot.dir ? shot.dir[0] : 0;
  let dz = shot.dir ? shot.dir[1] : 0;
  if (dx === 0 && dz === 0) dz = -1;
  const phi = Math.atan2(dx * off, -dz);
  res.name = shotName(phi, foot, loft, defend, p[1], d.spin);

  const rel = (p[0] - bodyX) * off;
  const y = p[1];
  const reachable = rel < 1.35 && rel > -0.95 && y < (foot === 'back' ? 2.0 : 1.9);
  if (ad > Wo || !reachable) return afterMiss(res, d, off, bodyX, rng, true);

  const fit = lineFit(rel, phi, defend) * lengthFit(d, foot, y, phi, defend);
  const tq = ad <= Wp ? 1 : ad <= Wg ? lerp(1, 0.74, (ad - Wp) / (Wg - Wp)) : lerp(0.74, 0.36, (ad - Wg) / (Wo - Wg));
  const cq = tq * (0.4 + 0.6 * fit);
  res.q = cq;

  const threat = 0.85 + 0.3 * bowlerSkill;
  let pMiss = (clamp((0.4 - cq) * 1.5, 0, 0.85) + (fit < 0.12 ? 0.4 : 0)) * threat;
  let pEdge = clamp((0.62 - cq) * 0.8, 0, 0.45) * threat;
  if (defend) {
    pMiss *= 0.5;
    pEdge *= 0.4;
  }
  // A lofted shot can be skied off the top edge, more so when it doesn't suit the ball.
  const pTop = loft ? (0.03 + (1 - fit) * 0.12 + (1 - tq) * 0.1) * threat : 0;
  const roll = rng();
  if (roll < pMiss) return afterMiss(res, d, off, bodyX, rng, true);

  const v = velAt(d, tc);
  const ballSpeed = Math.sqrt(v[0] * v[0] + v[2] * v[2]);
  const pos = [p[0], Math.max(0.12, p[1]), p[2]];
  let speed;
  let elev;
  let angle;

  if (roll < pMiss + pEdge) {
    res.kind = 'edge';
    const outside = rng() < 0.72;
    angle = outside ? 2.55 + rng() * 0.5 : -(2.5 + rng() * 0.55);
    speed = ballSpeed * (0.5 + 0.3 * rng());
    elev = (2 + rng() * 14) * DEG;
  } else if (roll < pMiss + pEdge + pTop) {
    res.kind = 'edge';
    res.name = 'top edge';
    angle = phi + rng.gauss() * 0.5;
    speed = 11 + 8 * rng();
    elev = (52 + 18 * rng()) * DEG;
  } else {
    res.kind = 'hit';
    const late = clamp(delta / Wo, -1, 1);
    angle = phi + (phi >= 0 ? 1 : -1) * late * 0.42 + rng.gauss() * (1 - cq) * 0.2;
    const power = 0.9 + 0.2 * skill;
    const paceBonus = 0.88 + 0.24 * clamp(ballSpeed / 38, 0, 1.2);
    if (defend) {
      speed = 2.5 + 4 * rng();
      elev = 1.5 * DEG;
    } else if (loft) {
      speed = (11.5 + 17.5 * Math.pow(cq, 1.4)) * power * (0.94 + 0.12 * clamp(ballSpeed / 38, 0, 1.2));
      const e = 30 + (late < 0 ? -late * 20 : -late * 8) + rng.gauss() * 3;
      elev = clamp(e, 14, 62) * DEG;
    } else {
      const behind = Math.abs(phi) > 1.9;
      speed = behind ? 6 + (ballSpeed * 0.5 + 8) * cq : (7 + 22 * Math.pow(cq, 1.4)) * power * paceBonus;
      let e = -3 + Math.max(0, -late) * 14 * (1 - fit * 0.5) + Math.abs(rng.gauss()) * 1.5;
      if (foot === 'back' && y > 1.45) e += 16;
      elev = e * DEG;
    }
  }

  const vh = speed * Math.cos(elev);
  res.launch = [Math.sin(angle) * off * vh, speed * Math.sin(elev), -Math.cos(angle) * vh];
  res.pos = pos;
  return res;
}
