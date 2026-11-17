// Plays computer-v-computer matches through the real host and checks the rules add up.
// Usage: node test/sim.mjs [matches] [overs] [level]

import { MatchHost } from '../shared/host.js';
import { HOW } from '../shared/match.js';

const N = +(process.argv[2] || 200);
const OVERS = +(process.argv[3] || 20);
const LEVEL = process.argv[4] || 'medium';
const TEAM_IDS = ['IND', 'AUS', 'ENG', 'PAK', 'RSA', 'NZL', 'WIN', 'SRL'];

let failures = 0;
const check = (cond, msg) => {
  if (!cond) {
    failures++;
    if (failures < 20) console.error('FAIL:', msg);
  }
};

const agg = { runs: 0, wkts: 0, balls: 0, inns: 0, fours: 0, sixes: 0, wd: 0, nb: 0, how: {}, ties: 0, chases: 0, allOut: 0, totals: [], dots: 0, timings: {}, kinds: {} };

for (let n = 0; n < N; n++) {
  const a = TEAM_IDS[n % 8];
  const b = TEAM_IDS[(n * 3 + 1) % 8 === n % 8 ? (n + 1) % 8 : (n * 3 + 1) % 8];
  const seats = [{ kind: 'bot', name: 'A', on: true }, { kind: 'bot', name: 'B', on: true }];
  const host = new MatchHost({ seats, cfg: { overs: OVERS, teams: [a, b], level: LEVEL }, seed: 1000 + n * 7919 });
  const origApply = host.match.applyBall.bind(host.match);
  host.match.applyBall = (o) => {
    agg.timings[o.timing] = (agg.timings[o.timing] || 0) + 1;
    agg.kinds[o.kind] = (agg.kinds[o.kind] || 0) + 1;
    if (!o.wide && !o.noBall && !o.runs && !o.boundary) agg.dots++;
    return origApply(o);
  };
  host.start();
  const m = host.match;
  check(m.phase === 'done', `match ${n} did not finish (phase ${m.phase})`);
  check(m.inn.length === 2, `match ${n} innings count ${m.inn.length}`);
  for (const I of m.inn) {
    const batRuns = I.bt.reduce((s, r) => s + r[0], 0);
    check(batRuns + I.ex.wd + I.ex.nb === I.runs, `match ${n}: batter runs ${batRuns} + extras ${I.ex.wd + I.ex.nb} != total ${I.runs}`);
    const bowlBalls = I.bw.reduce((s, r) => s + r[0], 0);
    check(bowlBalls === I.balls, `match ${n}: bowler balls ${bowlBalls} != ${I.balls}`);
    const bowlRuns = I.bw.reduce((s, r) => s + r[1], 0);
    check(bowlRuns === I.runs, `match ${n}: bowler runs ${bowlRuns} != ${I.runs}`);
    const outs = I.bt.filter((r) => r[4] >= HOW.BOWLED).length;
    check(outs === I.wkts, `match ${n}: dismissals ${outs} != wkts ${I.wkts}`);
    check(I.wkts <= 10 && I.balls <= OVERS * 6, `match ${n}: limits exceeded`);
    const cap = Math.ceil(OVERS / 5) * 6;
    check(I.bw.every((r) => r[0] <= cap), `match ${n}: bowler over cap`);
    check(I.fow.length === I.wkts, `match ${n}: fow length`);
    const inCount = I.bt.filter((r) => r[4] === HOW.IN).length;
    check(I.wkts === 10 ? inCount === 1 : inCount === 2, `match ${n}: not-out batters ${inCount} with ${I.wkts} wickets`);
    agg.runs += I.runs;
    agg.wkts += I.wkts;
    agg.balls += I.balls;
    agg.inns++;
    agg.wd += I.ex.wd;
    agg.nb += I.ex.nb;
    agg.totals.push(I.runs);
    if (I.wkts === 10) agg.allOut++;
    for (const r of I.bt) {
      agg.fours += r[2];
      agg.sixes += r[3];
      if (r[4] >= HOW.BOWLED) agg.how[r[4]] = (agg.how[r[4]] || 0) + 1;
    }
  }
  const first = m.inn[0];
  const second = m.inn[1];
  if (m.result.winner === -1) agg.ties++;
  else if (m.result.winner === second.bat) agg.chases++;
  check(second.target === first.runs + 1, `match ${n}: target`);
  if (second.runs >= second.target) check(m.result.winner === second.bat, `match ${n}: chase winner`);
}

const names = { 2: 'bowled', 3: 'caught', 4: 'lbw', 5: 'run out' };
agg.totals.sort((x, y) => x - y);
const pct = (p) => agg.totals[Math.floor(agg.totals.length * p)];
console.log(`${N} matches, ${OVERS} overs, level ${LEVEL}`);
console.log(`avg innings: ${(agg.runs / agg.inns).toFixed(1)} for ${(agg.wkts / agg.inns).toFixed(1)} in ${(agg.balls / agg.inns / 6).toFixed(1)} overs  (run rate ${(agg.runs / agg.balls * 6).toFixed(2)})`);
console.log(`totals p10/p50/p90: ${pct(0.1)}/${pct(0.5)}/${pct(0.9)}   all-out innings: ${(agg.allOut / agg.inns * 100).toFixed(0)}%`);
console.log(`per innings: fours ${(agg.fours / agg.inns).toFixed(1)}, sixes ${(agg.sixes / agg.inns).toFixed(1)}, wides ${(agg.wd / agg.inns).toFixed(1)}, no-balls ${(agg.nb / agg.inns).toFixed(1)}, dot balls ${(agg.dots / agg.balls * 100).toFixed(0)}%`);
console.log('dismissals:', Object.entries(agg.how).map(([k, v]) => `${names[k]} ${(v / agg.wkts * 100).toFixed(0)}%`).join(', '));
console.log('timing:', JSON.stringify(agg.timings), 'contact:', JSON.stringify(agg.kinds));
console.log(`chasing side won ${(agg.chases / N * 100).toFixed(0)}%, ties ${agg.ties}`);
console.log(failures ? `${failures} invariant failures` : 'all invariants hold');
process.exit(failures ? 1 : 0);
