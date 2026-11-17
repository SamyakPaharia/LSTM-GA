// One-line commentary built from the ball outcome.

import { lengthName, typeName } from '../../shared/delivery.js';

const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

export function describeBall({ o, d, striker, bowler, fielderName }) {
  const who = striker.short;
  const shot = o.name || 'shot';
  const where = o.at ? o.at : '';

  if (o.wicket) {
    const k = o.wicket.kind;
    if (k === 'bowled') return pick([`Bowled him! ${bowler.short} hits the stumps.`, `Timber! ${who} is bowled by ${bowler.short}.`, `Through the gate — ${who} has to go.`]);
    if (k === 'lbw') return pick([`Up goes the finger — ${who} is lbw to ${bowler.short}.`, `Trapped in front! ${who} lbw.`]);
    if (k === 'caught') {
      if (o.kind === 'edge') return pick([`Edged and taken at ${where}! ${who} walks.`, `Thin edge, ${fielderName} holds on. ${who} is out.`]);
      return pick([`In the air… and taken at ${where}! ${who} is caught.`, `${cap(shot)}, straight to ${where}. Caught by ${fielderName}.`, `${who} holes out to ${where}.`]);
    }
    return pick([`Run out! The throw from ${where} beats the batter.`, `Direct to the stumps — that's a run out.`]);
  }
  if (o.wide) return pick([`Wide. ${cap(lengthName(d))} ball slips past the tramline.`, 'Called wide by the umpire.']);

  const nb = o.noBall ? 'No-ball! Free hit coming. ' : '';
  if (o.boundary === 6) return nb + pick([`SIX! ${who} launches the ${shot} into the stands.`, `That is huge — ${shot} for six.`, `Clean as a whistle. Six runs.`]);
  if (o.boundary === 4) return nb + pick([`FOUR! ${cap(shot)} races away.`, `Beats the field — ${shot} for four.`, `${who} finds the gap. Four.`]);

  if (o.kind === 'miss') return nb + pick([`Beaten! ${typeName(d)} from ${bowler.short} goes past the bat.`, `Swing and a miss from ${who}.`, `Past the edge. Good ball.`]);
  if (o.kind === 'none') return nb + pick([`Left alone outside off.`, `${who} shoulders arms.`, `No shot offered.`]);
  if (o.dropped) return nb + `Put down at ${where}! ${o.runs ? `They take ${o.runs}.` : 'A let-off.'}`;
  if (o.kind === 'edge') return nb + (o.runs ? `Off the edge past ${where} for ${o.runs}.` : `Edged, but it falls short of ${where}.`);
  if (o.runs === 0) return nb + pick([`${cap(shot)}, straight to ${where || 'the fielder'}.`, `${cap(shot)} — no run.`, `Well fielded at ${where || 'the ring'}.`]);
  if (o.runs === 1) return nb + pick([`${cap(shot)} to ${where || 'the deep'} for a single.`, `Worked away for one.`, `Quick single taken.`]);
  return nb + pick([`${cap(shot)} into the gap, they come back for ${o.runs}.`, `Good running — ${o.runs} runs.`]);
}

export function overSummary(n, runs, team) {
  return `End of over ${n}: ${runs} from it. ${team}`;
}
