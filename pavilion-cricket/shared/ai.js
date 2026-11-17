// Computer player: toss, bowler choice, field, delivery, shot and running.

import {
  HALF, STUMP_H, STUMP_HALF_W, BALL_R, FRONT_FOOT_Z, BACK_FOOT_Z, SWING_TIME, STRIKER_CREASE_Z,
  clamp, offSignFor,
} from './constants.js';
import { isSpin, typesFor, bowlingArm, ballAt, timeAtZ } from './delivery.js';
import { LEVELS, lineFit, lengthFit, bodyXFor, WIDE_OFF, WIDE_LEG, WIDE_HIGH } from './batting.js';
import { RunPhase, RUN_TICKS } from './fielding.js';

const DEG = Math.PI / 180;
const SHOT_ANGLES = [-150, -112, -82, -52, -22, 0, 22, 48, 78, 108, 142];

export class Bot {
  constructor(level, rng) {
    this.level = level;
    this.lv = LEVELS[level] || LEVELS.medium;
    this.rng = rng;
  }

  toss() {
    return this.rng() < 0.55 ? 'bat' : 'bowl';
  }

  pickBowler(match) {
    const I = match.I;
    const team = match.bowlingTeam;
    const elig = match.eligibleBowlers();
    const over = Math.floor(I.balls / 6);
    const total = match.cfg.overs;
    const middle = over >= total * 0.3 && over < total * 0.75;
    return this.rng.weighted(elig, (idx) => {
      const p = team.players[idx];
      if (!p.style) return 0.02;
      let w = Math.pow(p.bowl / 100, 4);
      const spin = isSpin(p.style);
      if (middle ? spin : !spin) w *= 1.8;
      return w;
    });
  }

  field(match) {
    const I = match.I;
    const bowler = match.bowlingTeam.players[I.bowler];
    const frac = I.balls / (match.cfg.overs * 6);
    const chasingHard = I.target != null && (I.target - I.runs) / Math.max(1, match.cfg.overs * 6 - I.balls) > 1.6;
    if (frac >= 0.8 || chasingHard) return 'defensive';
    if (isSpin(bowler.style)) return 'spin';
    return frac < 0.3 ? 'attacking' : 'balanced';
  }

  bowl(match) {
    const rng = this.rng;
    const I = match.I;
    const bowler = match.bowlingTeam.players[I.bowler];
    const striker = match.battingTeam.players[I.st];
    const style = bowler.style || 'RM';
    const spin = isSpin(style);
    const off = offSignFor(striker.hand);
    const death = I.balls >= match.cfg.overs * 6 * 0.8;

    const lengths = spin
      ? [['good', 0.6, 5.0, 6.6], ['full', 0.2, 7.0, 8.0], ['short', 0.1, 3.0, 4.5], ['yorker', 0.1, 8.4, 9.0]]
      : [['good', 0.5, 3.4, 5.2], ['full', 0.18, 6.2, 7.6], ['yorker', death ? 0.3 : 0.08, 8.3, 9.0], ['short', 0.16, 1.0, 2.8], ['bouncer', 0.05, -0.6, 0.6]];
    const len = rng.weighted(lengths, (l) => l[1]);
    const z = rng.range(len[2], len[3]);
    const types = typesFor(style);
    const weights = spin
      ? { stock: 0.5, arm: 0.15, wrong: 0.15, flight: 0.2 }
      : { stock: 0.4, out: 0.2, in: 0.15, cutter: 0.15, slower: death ? 0.25 : 0.1 };
    const type = rng.weighted(types, (t) => weights[t.id] || 0.1).id;

    // Aim so that the movement brings the ball back toward off stump, not down the leg side.
    let moveIn = 0;
    if (spin) {
      const turn = (style === 'OS' ? -1 : 1) * (type === 'wrong' ? -1 : 1) * off;
      moveIn = type === 'arm' ? 0 : -turn;
    } else if (type === 'in' || type === 'cutter') {
      moveIn = 1;
    } else if (type === 'out') {
      moveIn = -1;
    }
    let x = rng() < 0.12 ? rng.range(-0.08, 0.06) : rng.range(0.04, 0.38);
    x = off * (x + moveIn * (spin ? 0.3 : 0.2));

    const sharp = this.lv.botAcc * (0.6 + 0.4 * (bowler.bowl / 100));
    const effort = rng() < sharp ? rng.range(0.82, 0.94) : rng.range(0.55, 0.985);
    return { type, x, z, e: effort };
  }

  bat(match, d, field) {
    const rng = this.rng;
    const I = match.I;
    const striker = match.battingTeam.players[I.st];
    const bowler = match.bowlingTeam.players[I.bowler];
    const off = offSignFor(striker.hand);
    const skill = striker.bat / 100;
    const overs = match.cfg.overs;
    const ballsLeft = Math.max(1, overs * 6 - I.balls);
    const wktsLeft = 10 - I.wkts;

    let agg = 0.34 + 0.45 * (1 - ballsLeft / (overs * 6));
    if (overs <= 5) agg += 0.18;
    if (I.target != null) {
      const rrr = ((I.target - I.runs) / ballsLeft) * 6;
      agg = rrr < 5 ? Math.max(0.25, agg - 0.15) : agg + (rrr - 7.5) * 0.06;
    }
    if (wktsLeft <= 3) agg -= 0.14;
    agg = clamp(agg + this.lv.botAgg, 0.1, 0.95);
    if (I.freeHit) agg = 0.95;

    const bodyX = bodyXFor(off, 0);
    const tF = timeAtZ(d, FRONT_FOOT_Z);
    const yF = ballAt(d, tF)[1];
    const foot = yF > 0.92 ? 'back' : 'front';
    const plane = foot === 'back' ? BACK_FOOT_Z : FRONT_FOOT_Z;
    const tc = timeAtZ(d, plane);
    const p = ballAt(d, tc);
    const rel = (p[0] - bodyX) * off;
    const y = p[1];
    const ps = ballAt(d, timeAtZ(d, HALF));
    const onStumps = Math.abs(ps[0]) <= STUMP_HALF_W + BALL_R + 0.03 && ps[1] <= STUMP_H + 0.05;
    const pc = ballAt(d, timeAtZ(d, STRIKER_CREASE_Z));
    const wideBall = pc[0] * off > WIDE_OFF || pc[0] * off < -WIDE_LEG || pc[1] > WIDE_HIGH;

    if (rel > 1.25 || rel < -0.85 || y > 1.85 || (wideBall && !onStumps && rng() < 0.8)) return { kind: 'none', t: tc, guard: 0 };
    if (!onStumps && rel > 0.8 && agg < 0.5 && rng() < 0.45) return { kind: 'none', t: tc, guard: 0 };

    const pDefend = clamp(onStumps ? 0.52 - 0.6 * agg : 0.16 - 0.3 * agg, 0, 0.6);
    const defend = rng() < pDefend;
    const loft = !defend && rng() < agg * agg * 0.5;

    let dir = [0, -1];
    if (!defend) {
      const origin = [p[0], plane];
      const best = rng.weighted(SHOT_ANGLES, (deg) => {
        const phi = deg * DEG;
        const fit = lineFit(rel, phi, false) * lengthFit(d, foot, y, phi, false);
        const square = Math.abs(deg) > 60;
        const lengthBias = foot === 'back' ? (square ? 1.3 : 0.7) : square ? 0.75 : 1.2;
        const gap = this.gapScore(phi, off, origin, field, loft);
        return (fit * fit + 0.01) * lengthBias * Math.pow(gap, 1.6);
      });
      const phi = (best + rng.range(-9, 9)) * DEG;
      dir = [Math.sin(phi) * off, -Math.cos(phi)];
    }

    const sigma = this.lv.botSigma * (1.3 - 0.55 * skill) * (loft ? 1.2 : 1) * (defend ? 0.7 : 1) * (0.9 + 0.25 * (bowler.bowl / 100));
    const t = tc - SWING_TIME + rng.gauss() * sigma;
    return { kind: defend ? 'defend' : 'stroke', foot, loft, dir, t, guard: 0 };
  }

  // Angular room around a shot direction: 1 is a wide-open gap, small is straight at a fielder.
  gapScore(phi, off, origin, field, loft) {
    let min = Math.PI;
    for (let i = 1; i < field.length; i++) {
      const f = field[i];
      const dx = f.x - origin[0];
      const dz = f.z - origin[1];
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (loft ? dist < 35 : dist > 45) continue;
      let diff = Math.abs(Math.atan2(dx * off, -dz) - phi);
      if (diff > Math.PI) diff = 2 * Math.PI - diff;
      if (diff < min) min = diff;
    }
    return clamp(min / (26 * DEG), 0.12, 1.25);
  }

  // Largest number of runs that leaves a safety margin against the throw.
  runs(plan, protectedBall) {
    if (plan.outcome === 'boundary') return [];
    if (plan.outcome === 'caught' && !protectedBall) return [];
    const opts = { noCatchOut: protectedBall };
    const margin = Math.round((this.lv.margin + this.rng.range(-0.1, 0.15)) * 60);
    for (let n = 3; n >= 1; n--) {
      const cmds = [];
      for (let j = 0; j < n; j++) cmds.push([1, j === 0 ? 14 : 14 + j * (RUN_TICKS + 13) - 45]);
      const r = RunPhase.replay(plan, cmds, Infinity, opts);
      if (!r.runOut && r.completed === n && r.deadTick - r.lastHome >= margin) return cmds;
    }
    return [];
  }
}

export { bowlingArm };
