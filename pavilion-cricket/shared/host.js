// MatchHost: the authority for one match. It owns the Match, all randomness and the
// per-ball flow, and talks to two seats (human via a link, or bot).

import { Match } from './match.js';
import { Bot } from './ai.js';
import { makeRng } from './rng.js';
import { makeDelivery, AIM_LIMITS } from './delivery.js';
import { evaluateShot, GUARD_LIMIT } from './batting.js';
import { planFielding, fieldPositions, RunPhase, FIELD_PRESETS } from './fielding.js';
import { clamp, offSignFor } from './constants.js';

const defaultTimers = { set: (fn, ms) => setTimeout(fn, ms), clear: (id) => clearTimeout(id) };

const num = (v, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : d);

export function cleanShot(s) {
  if (!s || (s.kind !== 'stroke' && s.kind !== 'defend')) return { kind: 'none', t: num(s && s.t), guard: clamp(num(s && s.guard), -GUARD_LIMIT, GUARD_LIMIT) };
  let dx = 0;
  let dz = -1;
  if (Array.isArray(s.dir)) {
    dx = num(s.dir[0]);
    dz = num(s.dir[1]);
    const len = Math.sqrt(dx * dx + dz * dz);
    if (Math.abs(len - 1) < 1e-9) {
      // already unit length: leave untouched so client and host see identical values
    } else if (len > 1e-6) {
      dx /= len;
      dz /= len;
    } else {
      dx = 0;
      dz = -1;
    }
  }
  return {
    kind: s.kind, foot: s.foot === 'back' ? 'back' : 'front', loft: !!s.loft, dir: [dx, dz],
    t: clamp(num(s.t), -1, 5), guard: clamp(num(s.guard), -GUARD_LIMIT, GUARD_LIMIT),
  };
}

function cleanCmds(cmds) {
  if (!Array.isArray(cmds)) return [];
  const out = [];
  for (const c of cmds.slice(0, 40)) {
    if (!Array.isArray(c)) continue;
    out.push([c[0] ? 1 : 0, Math.max(0, Math.floor(num(c[1])))]);
  }
  out.sort((a, b) => a[1] - b[1]);
  return out;
}

export class MatchHost {
  constructor({ seats, cfg, seed, timers = defaultTimers, timeouts = false, onDone = null }) {
    this.seats = seats;
    this.cfg = cfg;
    this.rng = makeRng(seed);
    this.match = new Match(cfg);
    this.hasBot = seats.some((s) => s.kind === 'bot');
    this.bots = seats.map((s) => (s.kind === 'bot' ? new Bot(cfg.level, makeRng(this.rng.seed())) : null));
    this.auto = new Bot('medium', makeRng(this.rng.seed()));
    this.timers = timers;
    this.timeouts = timeouts;
    this.onDone = onDone;
    this.ballSeq = 0;
    this.ball = null;
    this.ack = [-1, -1];
    this.cont = [false, false];
    this.preset = 'attacking';
    this.tossWin = 0;
    this.timer = null;
    this.dcTimer = null;
    this.turnKey = null;
    this.queue = [];
    this.draining = false;
    this.disposed = false;
  }

  isBot(seat) {
    return this.seats[seat].kind === 'bot';
  }

  get batSeat() {
    return this.match.I.bat;
  }

  get bowlSeat() {
    return 1 - this.match.I.bat;
  }

  send(seat, msg) {
    const s = this.seats[seat];
    if (s.kind === 'human' && s.on && s.send) s.send(msg);
  }

  broadcast(msg, except = -1) {
    for (let i = 0; i < this.seats.length; i++) if (i !== except) this.send(i, msg);
  }

  post(fn) {
    this.queue.push(fn);
    if (this.draining) return;
    this.draining = true;
    while (this.queue.length && !this.disposed) this.queue.shift()();
    this.draining = false;
  }

  arm(ms, fn) {
    this.disarm();
    if (!this.timeouts) return;
    this.timer = this.timers.set(() => {
      this.timer = null;
      if (!this.disposed) this.post(fn);
    }, ms);
  }

  disarm() {
    if (this.timer != null) this.timers.clear(this.timer);
    this.timer = null;
  }

  dispose() {
    this.disposed = true;
    this.disarm();
    if (this.dcTimer != null) this.timers.clear(this.dcTimer);
  }

  humansReady() {
    for (let i = 0; i < this.seats.length; i++) {
      if (this.isBot(i)) continue;
      if (!this.seats[i].on || this.ack[i] < this.ballSeq) return false;
    }
    return true;
  }

  sendSnap() {
    this.broadcast({ t: 'snap', snap: this.match.snapshot() });
  }

  sendField() {
    this.broadcast({ t: 'field', p: this.preset });
  }

  intro(seat) {
    this.send(seat, { t: 'begin', cfg: this.cfg, names: this.seats.map((s) => s.name), kinds: this.seats.map((s) => s.kind), you: seat });
  }

  start() {
    this.tossWin = this.rng.int(0, 1);
    for (let i = 0; i < this.seats.length; i++) this.intro(i);
    this.broadcast({ t: 'toss', win: this.tossWin });
    if (this.isBot(this.tossWin)) this.post(() => this.applyToss(this.bots[this.tossWin].toss()));
    else this.arm(30000, () => this.applyToss('bat'));
  }

  applyToss(choice) {
    if (this.match.phase !== 'toss') return;
    this.disarm();
    this.match.setToss(this.tossWin, choice);
    this.broadcast({ t: 'toss', win: this.tossWin, choice });
    this.sendField();
    this.sendSnap();
    this.post(() => this.advance());
  }

  // Tell the bowling side it may act. Sent once per decision, and only when every human
  // has finished watching the previous ball, so both screens stay in step.
  issueTurn(phase, ms, fallback) {
    const key = phase + ':' + this.ballSeq + ':' + this.match.cur;
    if (this.turnKey === key) return;
    this.turnKey = key;
    this.broadcast({ t: 'turn', phase, b: this.ballSeq, seat: this.bowlSeat });
    this.arm(ms, fallback);
  }

  turnOpen(phase) {
    return this.turnKey === phase + ':' + this.ballSeq + ':' + this.match.cur;
  }

  advance() {
    const m = this.match;
    if (this.ball || this.disposed) return;
    if (m.phase === 'pickBowler') {
      if (!this.humansReady()) return;
      const seat = this.bowlSeat;
      if (this.isBot(seat)) {
        m.setBowler(this.bots[seat].pickBowler(m));
        this.preset = this.bots[seat].field(m);
        this.sendField();
        this.sendSnap();
        this.post(() => this.advance());
      } else {
        this.issueTurn('pickBowler', 30000, () => {
          if (m.phase !== 'pickBowler') return;
          m.setBowler(this.auto.pickBowler(m));
          this.sendSnap();
          this.advance();
        });
      }
    } else if (m.phase === 'bowl') {
      if (!this.humansReady()) return;
      const seat = this.bowlSeat;
      if (this.isBot(seat)) {
        const preset = this.bots[seat].field(m);
        if (preset !== this.preset) {
          this.preset = preset;
          this.sendField();
        }
        this.doBowl(this.bots[seat].bowl(m));
      } else {
        this.issueTurn('bowl', 45000, () => {
          if (m.phase === 'bowl' && !this.ball) this.doBowl(this.auto.bowl(m));
        });
      }
    } else if (m.phase === 'inningsBreak') {
      const all = this.seats.every((s, i) => s.kind === 'bot' || this.cont[i] || !s.on);
      const go = () => {
        if (m.phase !== 'inningsBreak') return;
        this.disarm();
        m.startSecondInnings();
        this.cont = [false, false];
        this.preset = 'attacking';
        this.sendField();
        this.sendSnap();
        this.advance();
      };
      if (all) go();
      else if (this.timer == null) this.arm(40000, go);
    } else if (m.phase === 'done') {
      this.disarm();
      if (this.onDone) this.onDone(m.result);
    }
  }

  fieldNow(ball) {
    return fieldPositions(ball.preset, offSignFor(ball.striker.hand), ball.d.spin, ball.d.arm);
  }

  doBowl(input) {
    const m = this.match;
    const I = m.I;
    this.disarm();
    const bowler = m.bowlingTeam.players[I.bowler];
    const striker = m.battingTeam.players[I.st];
    const d = makeDelivery({
      bowler, type: String(input.type || 'stock'),
      aim: { x: clamp(num(input.x), -AIM_LIMITS.x, AIM_LIMITS.x), z: clamp(num(input.z, 4), AIM_LIMITS.zMin, AIM_LIMITS.zMax) },
      effort: clamp(num(input.e, 0.7), 0, 1), strikerHand: striker.hand, rng: this.rng,
    });
    const batSeat = this.batSeat;
    const rem = input.rem == null ? null : clamp(num(input.rem, 1.5), 0.55, 4);
    this.turnKey = null;
    const ball = {
      b: ++this.ballSeq, d, seed: this.rng.seed(), striker, bowler, bowlerIdx: I.bowler, freeHit: I.freeHit,
      preset: this.preset, res: null, plan: null, cmds: [], done: false,
      lv: this.isBot(batSeat) ? 'bot' : this.hasBot ? this.cfg.level : 'medium',
      fl: this.isBot(1 - batSeat) ? this.cfg.level : 'medium',
    };
    this.ball = ball;
    this.broadcast({ t: 'delivery', b: ball.b, d, seed: ball.seed, fh: ball.freeHit, f: ball.preset, lv: ball.lv, fl: ball.fl, rem, bw: I.bowler, st: I.st });
    if (this.isBot(batSeat)) this.onShot(this.bots[batSeat].bat(m, d, this.fieldNow(ball)));
    else this.arm(25000, () => this.onShot({ kind: 'none' }));
  }

  onShot(raw) {
    const ball = this.ball;
    if (!ball || ball.res) return;
    this.disarm();
    const shot = cleanShot(raw);
    const res = evaluateShot({ d: ball.d, shot, batter: ball.striker, bowlerSkill: ball.bowler.bowl / 100, level: ball.lv, seed: ball.seed });
    ball.res = res;
    const batSeat = this.batSeat;
    this.broadcast({ t: 'shot', b: ball.b, s: shot });
    this.broadcast({ t: 'contact', b: ball.b, res });
    if (res.kind === 'hit' || res.kind === 'edge') {
      ball.plan = planFielding({ pos: res.pos, vel: res.launch, field: this.fieldNow(ball), seed: (ball.seed ^ 0x5bd1e995) >>> 0, level: ball.fl, edge: res.kind === 'edge' });
      ball.prot = ball.freeHit || ball.d.noBall;
      if (this.isBot(batSeat)) {
        const cmds = this.bots[batSeat].runs(ball.plan, ball.prot);
        this.broadcast({ t: 'runs', b: ball.b, cmds });
        this.finalize(cmds);
      } else {
        this.arm(45000, () => this.finalize(ball.cmds));
      }
    } else {
      this.finalize([]);
    }
  }

  fielderIdx(planIdx, ball) {
    const team = this.match.bowlingTeam;
    if (planIdx === 0) return team.keeper;
    if (planIdx === 1) return ball.bowlerIdx;
    const rest = team.players.filter((p) => p.idx !== team.keeper && p.idx !== ball.bowlerIdx);
    return rest[planIdx - 2] ? rest[planIdx - 2].idx : -1;
  }

  finalize(cmds) {
    const ball = this.ball;
    if (!ball || ball.done) return;
    ball.done = true;
    this.disarm();
    const res = ball.res;
    const prot = ball.freeHit || ball.d.noBall;
    const o = { runs: 0, boundary: 0, wide: false, noBall: ball.d.noBall, wicket: null, endS: 'a', kind: res.kind, name: res.name, timing: res.timing, dropped: false, by: -1 };
    if (ball.plan) {
      const r = RunPhase.replay(ball.plan, cmds, Infinity, { noCatchOut: prot }).result();
      o.runs = r.runs;
      o.boundary = r.boundary;
      o.endS = r.endS;
      o.dropped = ball.plan.dropped;
      o.by = this.fielderIdx(r.fielder, ball);
      o.at = ball.plan.fielder >= 0 ? this.fieldNow(ball)[ball.plan.fielder].name : '';
      if (r.caught) o.wicket = { kind: 'caught', who: 'a', fielder: o.by };
      else if (r.runOut) o.wicket = { kind: 'runout', who: r.runOut.who, fielder: o.by };
    } else if (res.dismissal && !prot) {
      o.wicket = { kind: res.dismissal, who: 'a', fielder: null };
      o.endS = 'new';
    } else if (res.wide && !ball.d.noBall) {
      o.wide = true;
    }
    const ev = this.match.applyBall(o);
    this.ball = null;
    this.broadcast({ t: 'result', b: ball.b, o, ev, cmds, snap: this.match.snapshot() });
    this.post(() => this.advance());
  }

  onMessage(seat, msg) {
    if (this.disposed || this.isBot(seat)) return;
    const m = this.match;
    const ball = this.ball;
    switch (msg.t) {
      case 'tossChoice':
        if (m.phase === 'toss' && seat === this.tossWin) this.post(() => this.applyToss(msg.c === 'bowl' ? 'bowl' : 'bat'));
        break;
      case 'pickBowler':
        if (m.phase === 'pickBowler' && seat === this.bowlSeat && this.turnOpen('pickBowler') && m.setBowler(msg.i | 0)) {
          this.disarm();
          this.turnKey = null;
          this.sendSnap();
          this.post(() => this.advance());
        }
        break;
      case 'field':
        if (!ball && m.cur >= 0 && m.phase !== 'done' && seat === this.bowlSeat && FIELD_PRESETS[msg.p]) {
          this.preset = msg.p;
          this.sendField();
        }
        break;
      case 'bowl':
        if (m.phase === 'bowl' && !ball && seat === this.bowlSeat && this.turnOpen('bowl')) this.post(() => this.doBowl(msg));
        break;
      case 'shot':
        if (ball && !ball.res && seat === this.batSeat && msg.b === ball.b) this.post(() => this.onShot(msg.s));
        break;
      case 'run':
        if (ball && ball.plan && !ball.done && seat === this.batSeat && msg.b === ball.b) {
          const c = [msg.k ? 1 : 0, Math.max(0, Math.floor(num(msg.tick)))];
          ball.cmds.push(c);
          this.broadcast({ t: 'run', b: ball.b, k: c[0], tick: c[1] }, seat);
        }
        break;
      case 'ballDone':
        if (ball && ball.plan && seat === this.batSeat && msg.b === ball.b) this.post(() => this.finalize(cleanCmds(msg.cmds)));
        break;
      case 'ready':
        this.ack[seat] = Math.max(this.ack[seat], msg.b | 0);
        this.post(() => this.advance());
        break;
      case 'continue':
        this.cont[seat] = true;
        this.post(() => this.advance());
        break;
      case 'emote':
        this.broadcast({ t: 'emote', seat, id: msg.id | 0 }, seat);
        break;
      default:
        break;
    }
  }

  voidBall() {
    if (!this.ball) return;
    const b = this.ball.b;
    this.ball = null;
    this.disarm();
    this.broadcast({ t: 'void', b });
  }

  disconnect(seat) {
    if (this.match.phase === 'done') return;
    this.broadcast({ t: 'peer', seat, on: false }, seat);
    if (this.dcTimer != null) this.timers.clear(this.dcTimer);
    this.dcTimer = this.timers.set(() => {
      this.dcTimer = null;
      this.forfeit(seat);
    }, 60000);
  }

  reconnect(seat) {
    if (this.dcTimer != null) this.timers.clear(this.dcTimer);
    this.dcTimer = null;
    this.voidBall();
    this.ack = this.ack.map(() => this.ballSeq);
    this.turnKey = null;
    this.intro(seat);
    if (this.match.toss) this.send(seat, { t: 'toss', win: this.tossWin, choice: this.match.toss.choice, quiet: true });
    else this.send(seat, { t: 'toss', win: this.tossWin });
    this.send(seat, { t: 'field', p: this.preset });
    this.send(seat, { t: 'snap', snap: this.match.snapshot(), sync: true });
    this.broadcast({ t: 'peer', seat, on: true }, seat);
    this.post(() => this.advance());
  }

  forfeit(seat) {
    if (this.match.phase === 'done' || this.disposed) return;
    this.ball = null;
    this.disarm();
    this.match.forfeit(seat);
    this.broadcast({ t: 'void', b: this.ballSeq });
    this.sendSnap();
    this.post(() => this.advance());
  }
}
