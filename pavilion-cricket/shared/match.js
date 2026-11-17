// Rules and scorecard. Holds only plain data so a snapshot is a deep copy.

import { getTeam } from './teams.js';

export const HOW = { DNB: 0, IN: 1, BOWLED: 2, CAUGHT: 3, LBW: 4, RUNOUT: 5 };
const HOW_CODE = { bowled: HOW.BOWLED, caught: HOW.CAUGHT, lbw: HOW.LBW, runout: HOW.RUNOUT };

// Batter row: [runs, balls, fours, sixes, how, bowlerIdx, fielderIdx]
// Bowler row: [balls, runs, wickets, wides, noBalls]
function newInnings(side) {
  const bt = [];
  const bw = [];
  for (let i = 0; i < 11; i++) {
    bt.push([0, 0, 0, 0, HOW.DNB, -1, -1]);
    bw.push([0, 0, 0, 0, 0]);
  }
  bt[0][4] = HOW.IN;
  bt[1][4] = HOW.IN;
  return {
    bat: side, runs: 0, wkts: 0, balls: 0, ex: { wd: 0, nb: 0 }, bt, bw, fow: [], ov: [], overRuns: 0,
    thisOver: [], lastOver: [], st: 0, ns: 1, next: 2, bowler: -1, last: -1, freeHit: false, target: null,
  };
}

export class Match {
  constructor({ overs, teams, level = 'medium' }) {
    this.cfg = { overs, teams: [teams[0], teams[1]], level };
    this.teams = [getTeam(teams[0]), getTeam(teams[1])];
    this.phase = 'toss';
    this.toss = null;
    this.cur = -1;
    this.inn = [];
    this.result = null;
  }

  get I() {
    return this.inn[this.cur];
  }

  get battingTeam() {
    return this.teams[this.I.bat];
  }

  get bowlingTeam() {
    return this.teams[1 - this.I.bat];
  }

  get maxOversPerBowler() {
    return Math.ceil(this.cfg.overs / 5);
  }

  setToss(win, choice) {
    this.toss = { win, choice };
    this.startInnings(choice === 'bat' ? win : 1 - win);
  }

  startInnings(side) {
    const inn = newInnings(side);
    if (this.inn.length === 1) inn.target = this.inn[0].runs + 1;
    this.inn.push(inn);
    this.cur = this.inn.length - 1;
    this.phase = 'pickBowler';
  }

  startSecondInnings() {
    if (this.phase !== 'inningsBreak') return;
    this.startInnings(1 - this.inn[0].bat);
  }

  eligibleBowlers() {
    const I = this.I;
    const cap = this.maxOversPerBowler * 6;
    const out = [];
    const team = this.bowlingTeam;
    for (const p of team.players) {
      if (p.idx === I.last || p.idx === team.keeper) continue;
      if (I.bw[p.idx][0] >= cap) continue;
      out.push(p.idx);
    }
    if (!out.length) {
      for (const p of team.players) if (p.idx !== I.last && p.idx !== team.keeper) out.push(p.idx);
    }
    return out;
  }

  setBowler(idx) {
    if (this.phase !== 'pickBowler') return false;
    if (!this.eligibleBowlers().includes(idx)) return false;
    const I = this.I;
    I.bowler = idx;
    I.thisOver = [];
    this.phase = 'bowl';
    return true;
  }

  // o: { runs, boundary, wide, noBall, wicket: {kind, who:'a'|'b', fielder}|null, endS:'a'|'b'|'new' }
  applyBall(o) {
    const I = this.I;
    const B = I.bw[I.bowler];
    const events = [];
    const a = I.st;
    const b = I.ns;
    let total = 0;
    let label;

    if (o.wide) {
      I.ex.wd++;
      B[3]++;
      B[1]++;
      total = 1;
      label = 'Wd';
    } else {
      const S = I.bt[a];
      const r = o.boundary || o.runs || 0;
      if (o.noBall) {
        I.ex.nb++;
        B[4]++;
        B[1]++;
        total++;
      } else {
        I.balls++;
        B[0]++;
      }
      const before = S[0];
      S[0] += r;
      S[1]++;
      if (o.boundary === 4) S[2]++;
      if (o.boundary === 6) S[3]++;
      B[1] += r;
      total += r;
      label = o.noBall ? 'Nb' + (r || '') : r === 0 ? '0' : String(r);
      for (const mark of [50, 100]) if (before < mark && S[0] >= mark) events.push({ e: 'milestone', who: a, mark });
    }
    I.runs += total;
    I.overRuns += total;

    let newIdx = -1;
    if (o.wicket) {
      const who = o.wicket.who === 'b' ? b : a;
      const row = I.bt[who];
      const kind = o.wicket.kind;
      row[4] = HOW_CODE[kind] || HOW.BOWLED;
      row[5] = kind === 'runout' ? -1 : I.bowler;
      row[6] = o.wicket.fielder == null ? -1 : o.wicket.fielder;
      I.wkts++;
      if (kind !== 'runout') B[2]++;
      I.fow.push([I.runs, I.wkts, who, I.balls]);
      label = o.wide || o.noBall ? label + 'W' : 'W';
      events.push({ e: 'wicket', who, kind });
      if (I.wkts < 10 && I.next < 11) {
        newIdx = I.next++;
        I.bt[newIdx][4] = HOW.IN;
      }
    }

    // Who ends up at which end.
    const outIdx = o.wicket ? (o.wicket.who === 'b' ? b : a) : -1;
    const pool = [a, b, newIdx].filter((i) => i >= 0 && i !== outIdx);
    let st = o.endS === 'b' ? b : o.endS === 'new' ? newIdx : a;
    if (st < 0 || st === outIdx) st = pool[0];
    const ns = pool.find((i) => i !== st);
    I.st = st;
    I.ns = ns === undefined ? -1 : ns;

    if (o.noBall) I.freeHit = true;
    else if (!o.wide) I.freeHit = false;
    I.thisOver.push(label);
    if (o.boundary) events.push({ e: 'boundary', runs: o.boundary });

    const legal = !o.wide && !o.noBall;
    let overEnded = false;
    if (legal && I.balls % 6 === 0) {
      overEnded = true;
      I.ov.push(I.overRuns);
      I.overRuns = 0;
      I.lastOver = I.thisOver;
      const s = I.st;
      I.st = I.ns;
      I.ns = s;
      I.last = I.bowler;
      events.push({ e: 'over', n: I.balls / 6 });
    }

    const allOut = I.wkts >= 10;
    const oversDone = I.balls >= this.cfg.overs * 6;
    const chased = I.target != null && I.runs >= I.target;
    if (allOut || oversDone || chased) {
      if (I.balls % 6 !== 0) I.ov.push(I.overRuns);
      if (this.cur === 0) {
        this.phase = 'inningsBreak';
        events.push({ e: 'innings' });
      } else {
        this.finish();
        events.push({ e: 'done' });
      }
    } else if (overEnded) {
      I.bowler = -1;
      this.phase = 'pickBowler';
    } else {
      this.phase = 'bowl';
    }
    return events;
  }

  finish() {
    const first = this.inn[0];
    const second = this.inn[1];
    const t1 = this.teams[first.bat];
    const t2 = this.teams[second.bat];
    if (second.runs >= second.target) {
      const w = 10 - second.wkts;
      this.result = { winner: second.bat, text: `${t2.name} won by ${w} wicket${w === 1 ? '' : 's'}` };
    } else if (second.runs === first.runs) {
      this.result = { winner: -1, text: 'Match tied' };
    } else {
      const r = first.runs - second.runs;
      this.result = { winner: first.bat, text: `${t1.name} won by ${r} run${r === 1 ? '' : 's'}` };
    }
    this.phase = 'done';
  }

  forfeit(side) {
    if (this.phase === 'done') return;
    this.result = { winner: 1 - side, text: `${this.teams[1 - side].name} won — opponent left the match` };
    this.phase = 'done';
  }

  snapshot() {
    return JSON.parse(JSON.stringify({
      phase: this.phase, cfg: this.cfg, toss: this.toss, cur: this.cur, inn: this.inn, result: this.result,
      elig: this.phase === 'pickBowler' ? this.eligibleBowlers() : null,
    }));
  }
}

export function oversText(balls) {
  return Math.floor(balls / 6) + '.' + (balls % 6);
}
