// Field settings, ball flight after the hit, fielder intercepts, and the running phase.
// planFielding and RunPhase use only + - * / sqrt on their inputs and a fixed tick, so two
// browsers given the same launch vector and seed produce the same plan.

import { G, BALL_R, HALF, TICK, RUN_TIME, TURN_TIME, insideBoundary, clamp } from './constants.js';
import { makeRng } from './rng.js';
import { LEVELS } from './batting.js';

// Positions for a right-hander (off side = +x). Mirrored in x for a left-hander.
const SPOTS = {
  slip1: [1.9, 19.9, 'first slip'], slip2: [3.1, 19.4, 'second slip'], gully: [7.5, 15.5, 'gully'],
  point: [22, 9.5, 'point'], bwdPoint: [20, 14, 'backward point'], cover: [20, -2, 'cover'],
  extraCover: [16, -9, 'extra cover'], midOff: [8, -19, 'mid-off'], midOn: [-8, -19, 'mid-on'],
  midwicket: [-20, -2, 'midwicket'], squareLeg: [-22, 9.5, 'square leg'], shortFine: [-12, 24, 'short fine leg'],
  thirdMan: [27, 53, 'third man'], deepPoint: [56, 13, 'deep point'], sweeper: [51, -22, 'deep cover'],
  longOff: [15, -59, 'long-off'], longOn: [-15, -59, 'long-on'], deepMidwicket: [-50, -24, 'deep midwicket'],
  deepSquare: [-56, 13, 'deep square leg'], fineLeg: [-25, 55, 'fine leg'],
};

export const FIELD_PRESETS = {
  attacking: ['slip1', 'slip2', 'gully', 'point', 'cover', 'midOff', 'midOn', 'midwicket', 'fineLeg'],
  balanced: ['slip1', 'point', 'cover', 'midOff', 'midOn', 'midwicket', 'deepSquare', 'thirdMan', 'sweeper'],
  defensive: ['bwdPoint', 'extraCover', 'midOn', 'shortFine', 'longOff', 'deepMidwicket', 'deepSquare', 'thirdMan', 'sweeper'],
  spin: ['slip1', 'point', 'cover', 'midOff', 'midOn', 'midwicket', 'shortFine', 'longOn', 'deepMidwicket'],
};
export const FIELD_NAMES = { attacking: 'Attacking', balanced: 'Balanced', defensive: 'Defensive', spin: 'Spin ring' };
export const FIELD_ORDER = ['attacking', 'balanced', 'defensive', 'spin'];

// Index 0 is the keeper, 1 the bowler, 2..10 the fielders.
export function fieldPositions(preset, off, spin, arm) {
  const keys = FIELD_PRESETS[preset] || FIELD_PRESETS.balanced;
  const out = [
    { x: 0.3 * off, z: HALF + (spin ? 1.5 : 9), role: 'keeper', name: 'the keeper' },
    { x: arm === 'L' ? -1.0 : 1.0, z: -6.4, role: 'bowler', name: 'the bowler' },
  ];
  for (const k of keys) {
    const s = SPOTS[k];
    out.push({ x: s[0] * off, z: s[1], role: 'field', name: s[2] });
  }
  return out;
}

const SUB = 4;
const DT = 1 / 120;
export const SAMPLE_DT = SUB * DT;

function simulatePath(pos, vel) {
  const xs = [pos[0]];
  const ys = [pos[1]];
  const zs = [pos[2]];
  let x = pos[0];
  let y = pos[1];
  let z = pos[2];
  let vx = vel[0];
  let vy = vel[1];
  let vz = vel[2];
  let rolling = false;
  let stopped = false;
  let firstBounce = Infinity;
  let boundary = null;
  let extra = 0;
  const maxSteps = 14 * 120;
  for (let step = 1; step <= maxSteps; step++) {
    if (!stopped) {
      if (!rolling) {
        vy -= G * DT;
        x += vx * DT;
        y += vy * DT;
        z += vz * DT;
        if (y <= BALL_R && vy < 0) {
          y = BALL_R;
          if (firstBounce === Infinity) firstBounce = step * DT;
          const steep = clamp(-vy / 6, 0, 1);
          const keep = 1 - 0.3 * steep;
          vy = -vy * 0.42;
          vx *= keep;
          vz *= keep;
          if (vy < 0.9) {
            rolling = true;
            vy = 0;
          }
        }
      } else {
        const sp = Math.sqrt(vx * vx + vz * vz);
        const dec = (2.2 + 0.12 * sp) * DT;
        if (sp <= dec + 0.3) {
          stopped = true;
        } else {
          const k = (sp - dec) / sp;
          vx *= k;
          vz *= k;
          x += vx * DT;
          z += vz * DT;
        }
      }
    }
    if (!boundary && !insideBoundary(x, z)) boundary = { t: step * DT, runs: firstBounce === Infinity ? 6 : 4 };
    if (boundary) extra++;
    if (step % SUB === 0) {
      xs.push(x);
      ys.push(y);
      zs.push(z);
      if (stopped || extra > 96) break;
    }
  }
  return { xs, ys, zs, n: xs.length, firstBounce, boundary, endT: (xs.length - 1) * SAMPLE_DT };
}

// Close catchers get extra reach for an edge, which comes at them; a deliberate stroke is
// placed away from them.
function paramsFor(f, pos, edge) {
  if (f.role === 'keeper') return { react: edge ? 0.12 : 0.2, reach: edge ? 1.6 : 0.9, speed: 5.0 };
  if (f.role === 'bowler') return { react: 0.5, reach: 0.8, speed: 5.5 };
  const dx = f.x - pos[0];
  const dz = f.z - pos[2];
  return Math.sqrt(dx * dx + dz * dz) < 14 ? { react: 0.16, reach: edge ? 1.4 : 1.0, speed: 6.5 } : { react: 0.3, reach: 0.9, speed: 6.9 };
}

const dist2 = (ax, az, bx, bz) => Math.sqrt((ax - bx) * (ax - bx) + (az - bz) * (az - bz));

export function planFielding({ pos, vel, field, seed, level = 'medium', edge = false }) {
  const rng = makeRng(seed);
  const lv = LEVELS[level] || LEVELS.medium;
  const path = simulatePath(pos, vel);
  const limit = path.boundary ? Math.floor(path.boundary.t / SAMPLE_DT) : path.n - 1;

  let first = null;
  for (let i = 0; i < field.length; i++) {
    const f = field[i];
    const prm = paramsFor(f, pos, edge);
    let found = null;
    for (let k = 1; k <= limit; k++) {
      if (path.ys[k] > 2.45) continue;
      const t = k * SAMPLE_DT;
      const d = dist2(path.xs[k], path.zs[k], f.x, f.z);
      if (d <= prm.reach + prm.speed * Math.max(0, t - prm.react)) {
        found = { i, k, t, d, prm, rest: false };
        break;
      }
    }
    if (!found && !path.boundary) {
      const k = path.n - 1;
      const d = dist2(path.xs[k], path.zs[k], f.x, f.z);
      found = { i, k, t: Math.max(path.endT, prm.react + Math.max(0, d - prm.reach) / prm.speed), d, prm, rest: true };
    }
    if (found && (!first || found.t < first.t || (found.t === first.t && found.d < first.d))) first = found;
  }

  const plan = {
    path, field, boundary: path.boundary, outcome: 'boundary', t: path.boundary ? path.boundary.t : path.endT,
    touch: 0, k: path.n - 1, fielder: -1, pick: [0, 0], air: false, dropped: false, throwDelay: 0.8,
    moves: [], recv: { S: { i: 0, t: 0 }, N: { i: 1, t: 0 } },
  };

  if (first) {
    const f = field[first.i];
    const need = first.prm.react + Math.max(0, first.d - first.prm.reach) / first.prm.speed;
    plan.fielder = first.i;
    plan.k = first.k;
    plan.touch = first.t;
    plan.t = first.t;
    plan.pick = [path.xs[first.k], path.zs[first.k]];
    plan.outcome = 'fielded';
    plan.air = !first.rest && first.t < path.firstBounce;
    if (plan.air) {
      let p = clamp(0.6 + (first.t - need) * 0.5, 0.42, 0.96);
      if (f.role === 'keeper' && first.t < 1.2) p = Math.max(p, 0.8);
      else if (first.t < 0.75) p *= 0.75;
      p *= lv.catchSkill;
      if (rng() < p) {
        plan.outcome = 'caught';
      } else {
        plan.dropped = true;
        plan.t = first.t + 0.95;
      }
    }
    plan.moves.push({ i: first.i, x: plan.pick[0], z: plan.pick[1], t0: first.prm.react, t1: Math.max(first.prm.react + 0.05, Math.min(first.t, need)) });
  }

  // Supporting movement: two nearest fielders back up, keeper and bowler cover the stumps.
  const target = first ? plan.pick : [path.xs[limit], path.zs[limit]];
  const others = [];
  for (let i = 2; i < field.length; i++) {
    if (i === plan.fielder) continue;
    others.push({ i, d: dist2(field[i].x, field[i].z, target[0], target[1]) });
  }
  others.sort((a, b) => a.d - b.d);
  const covering = new Set();
  const cover = (end, defaultIdx, sx, sz) => {
    let i = defaultIdx;
    if (plan.fielder === defaultIdx) {
      let bestD = Infinity;
      for (let j = 2; j < field.length; j++) {
        if (j === plan.fielder || covering.has(j)) continue;
        const d = dist2(field[j].x, field[j].z, sx, sz);
        if (d < bestD) {
          bestD = d;
          i = j;
        }
      }
    }
    covering.add(i);
    const d = dist2(field[i].x, field[i].z, sx, sz);
    const t1 = 0.25 + d / 5.5;
    plan.moves.push({ i, x: sx, z: sz, t0: 0.25, t1 });
    plan.recv[end] = { i, t: t1 };
  };
  cover('S', 0, 0.2, HALF + 0.8);
  cover('N', 1, -0.25, -HALF - 0.8);
  let backups = 0;
  for (const o of others) {
    if (backups >= 2 || o.d > 45) break;
    if (covering.has(o.i)) continue;
    const f = field[o.i];
    const k = Math.max(0, (o.d - 3.5) / Math.max(o.d, 0.001));
    plan.moves.push({ i: o.i, x: f.x + (target[0] - f.x) * k, z: f.z + (target[1] - f.z) * k, t0: 0.35, t1: 0.35 + Math.max(0.1, (o.d - 3.5) / 6.5) });
    backups++;
  }
  return plan;
}

// Ball position while it is on its hit path (before any throw).
export function ballOnPath(plan, t, out = [0, 0, 0]) {
  const p = plan.path;
  const stopAt = plan.outcome === 'boundary' ? p.endT : plan.touch;
  if (t >= stopAt && plan.outcome !== 'boundary') {
    out[0] = plan.pick[0];
    out[2] = plan.pick[1];
    const yAt = p.ys[plan.k];
    if (plan.dropped) {
      const u = clamp((t - plan.touch) / 0.35, 0, 1);
      out[1] = yAt + (BALL_R - yAt) * u * u;
    } else {
      out[1] = plan.air ? Math.max(yAt, 0.4) : 0.35;
    }
    return out;
  }
  const f = clamp(t, 0, p.endT) / SAMPLE_DT;
  const i = Math.min(p.n - 2, Math.floor(f));
  const u = f - i;
  out[0] = p.xs[i] + (p.xs[i + 1] - p.xs[i]) * u;
  out[1] = p.ys[i] + (p.ys[i + 1] - p.ys[i]) * u;
  out[2] = p.zs[i] + (p.zs[i + 1] - p.zs[i]) * u;
  return out;
}

const RUN_DP = TICK / RUN_TIME;
const TURN_TICKS = Math.round(TURN_TIME / TICK);
export const RUN_TICKS = Math.round(RUN_TIME / TICK);

// Running between the wickets and the throw, on a fixed 60 Hz tick.
// "a" is the striker who faced the ball, "b" the non-striker.
export class RunPhase {
  constructor(plan, opts = {}) {
    this.plan = plan;
    this.catchCounts = !opts.noCatchOut;
    this.tick = 0;
    this.completed = 0;
    this.p = 0;
    this.dir = 0;
    this.queued = 0;
    this.wait = 0;
    this.ball = 'path';
    this.holdUntil = 0;
    this.arrive = 0;
    this.throwEnd = null;
    this.throws = [];
    this.dead = false;
    this.deadTick = 0;
    this.runOut = null;
    this.caught = false;
    this.boundary = 0;
    this.lastHome = 0;
    this.cmdLog = [];
  }

  command(run) {
    if (this.dead) return false;
    this.cmdLog.push([run ? 1 : 0, this.tick]);
    this.apply(run);
    return true;
  }

  apply(run) {
    if (run) {
      if (this.dir === 0 && this.wait === 0) this.dir = 1;
      else if (this.dir === 1 || this.wait > 0) this.queued = 1;
      else this.dir = 1;
    } else {
      this.queued = 0;
      if (this.dir === 1 && this.p < 0.92) this.dir = -1;
    }
  }

  get midPitch() {
    return this.dir !== 0 && this.p > 0 && this.p < 1;
  }

  get idle() {
    return this.dir === 0 && this.wait === 0 && this.queued === 0;
  }

  kill() {
    this.dead = true;
    this.deadTick = this.tick;
  }

  step() {
    if (this.dead) return;
    this.tick++;
    const t = this.tick * TICK;
    const plan = this.plan;

    if (this.wait > 0) {
      this.wait--;
      if (this.wait === 0 && this.queued > 0) {
        this.queued = 0;
        this.dir = 1;
      }
    } else if (this.dir !== 0) {
      this.p += this.dir * RUN_DP;
      if (this.p >= 1) {
        this.p = 0;
        this.completed++;
        this.dir = 0;
        this.lastHome = this.tick;
        if (this.queued > 0) this.wait = TURN_TICKS;
      } else if (this.p <= 0) {
        this.p = 0;
        this.dir = 0;
        this.lastHome = this.tick;
      }
    }

    if (this.ball === 'path') {
      if (plan.outcome === 'boundary') {
        if (t >= plan.boundary.t) {
          this.boundary = plan.boundary.runs;
          this.kill();
        }
      } else if (plan.outcome === 'caught' && this.catchCounts) {
        if (t >= plan.t) {
          this.caught = true;
          this.kill();
        }
      } else if (t >= plan.t) {
        this.ball = 'held';
        this.holdUntil = plan.t + plan.throwDelay;
      }
    } else if (this.ball === 'held') {
      if (t >= this.holdUntil) {
        if (this.idle) {
          this.ball = 'idle';
          this.arrive = t + 0.35;
        } else {
          let best = null;
          for (const end of ['S', 'N']) {
            const sz = end === 'S' ? HALF : -HALF;
            const d = dist2(plan.pick[0], plan.pick[1], 0, sz);
            const travel = d < 3 ? d / 6 : 0.15 + d / 27;
            const at = Math.max(t + travel, plan.recv[end].t + 0.05);
            if (!best || at < best.at) best = { end, at, sz };
          }
          this.ball = 'thrown';
          this.arrive = best.at;
          this.throwEnd = best.end;
          this.throws.push({ x0: plan.pick[0], z0: plan.pick[1], x1: 0, z1: best.sz, t0: t, t1: best.at, end: best.end });
        }
      }
    } else if (this.ball === 'idle') {
      if (!this.idle) {
        // A run was called after the fielder relaxed: he throws after all.
        this.ball = 'held';
        this.holdUntil = t;
      } else if (t >= this.arrive) {
        this.kill();
      }
    } else if (this.ball === 'thrown') {
      if (t >= this.arrive) {
        if (this.midPitch) {
          const sRunner = this.completed % 2 === 0 ? 'a' : 'b';
          const other = sRunner === 'a' ? 'b' : 'a';
          const nearS = this.p < 0.5 ? sRunner : other;
          const who = this.throwEnd === 'S' ? nearS : nearS === 'a' ? 'b' : 'a';
          this.runOut = { who, end: this.throwEnd };
        }
        this.queued = 0;
        this.wait = 0;
        this.kill();
      }
    }
    if (!this.dead && this.tick > 60 * 40) this.kill();
  }

  static replay(plan, cmds, upto = Infinity, opts) {
    const r = new RunPhase(plan, opts);
    let ci = 0;
    while (!r.dead && r.tick < upto) {
      while (ci < cmds.length && cmds[ci][1] <= r.tick) {
        r.apply(cmds[ci][0] === 1);
        ci++;
      }
      r.step();
    }
    r.cmdLog = cmds.slice(0, ci);
    return r;
  }

  // Who stands at the striker's end once the ball is dead: 'a', 'b' or 'new'.
  result() {
    const odd = this.completed % 2 === 1;
    let endS = odd ? 'b' : 'a';
    if (this.boundary) endS = 'a';
    if (this.caught) endS = 'new';
    if (this.runOut) {
      const survivor = this.runOut.who === 'a' ? 'b' : 'a';
      endS = this.runOut.end === 'S' ? 'new' : survivor;
    }
    return {
      runs: this.boundary ? 0 : this.completed, boundary: this.boundary, caught: this.caught,
      fielder: this.plan.fielder, runOut: this.runOut, endS, deadTick: this.deadTick,
    };
  }
}
