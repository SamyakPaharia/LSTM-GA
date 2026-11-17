// Plays complete two-human matches with scripted clients over each transport and checks
// that both screens end with the same scorecard.
// Usage: node test/net.mjs

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { RoomCore } from '../shared/room.js';
import { LocalLink, WsLink, PresenceHost, PresenceGuest } from '../client/src/links.js';
import { idealPressTime } from '../shared/batting.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// A client that answers every prompt the host sends, with small random delays.
function scripted(link, label, { autostart = false, overs = 2 } = {}) {
  const st = { you: -1, snap: null, done: false, final: null, balls: 0, errors: [], turns: 0 };
  const later = (fn) => setTimeout(fn, Math.random() * 15);
  let configured = false;
  link.onMessage = (m) => {
    switch (m.t) {
      case 'lobby':
        if (m.you === 0 && !configured) {
          configured = true;
          link.send({ t: 'cfg', overs, team: 'ENG' });
        } else if (m.you === 1 && !configured) {
          configured = true;
          link.send({ t: 'cfg', team: 'NZL' });
        }
        if (autostart && m.you === 0 && m.state === 'lobby' && m.players[1] && m.players[1].on && m.cfg.teams[1] === 'NZL') {
          link.send({ t: 'start' });
          autostart = false;
        }
        break;
      case 'begin':
        st.you = m.you;
        break;
      case 'toss':
        if (m.choice === undefined && m.win === st.you) later(() => link.send({ t: 'tossChoice', c: 'bat' }));
        if (m.choice !== undefined) later(() => link.send({ t: 'ready', b: 0 }));
        break;
      case 'snap':
        st.snap = m.snap;
        if (m.snap.phase === 'inningsBreak') later(() => link.send({ t: 'continue' }));
        if (m.snap.phase === 'done') {
          st.done = true;
          st.final = m.snap;
        }
        break;
      case 'turn':
        if (m.seat !== st.you) break;
        st.turns++;
        if (m.phase === 'pickBowler') later(() => link.send({ t: 'pickBowler', i: st.snap.elig[0] }));
        if (m.phase === 'bowl') later(() => link.send({ t: 'bowl', type: 'stock', x: 0.1, z: 3 + Math.random() * 3, e: 0.82 + Math.random() * 0.12, rem: 1.1 }));
        break;
      case 'delivery': {
        const I = st.snap.inn[st.snap.cur];
        if (I.bat !== st.you) break;
        const foot = Math.random() < 0.5 ? 'front' : 'back';
        const t = idealPressTime(m.d, foot) + (Math.random() - 0.5) * 0.12;
        later(() => link.send({ t: 'shot', b: m.b, s: { kind: 'stroke', foot, loft: Math.random() < 0.3, dir: [Math.random() - 0.5, -1], t, guard: 0 } }));
        break;
      }
      case 'contact': {
        const I = st.snap.inn[st.snap.cur];
        if (I.bat !== st.you) break;
        if (m.res.kind === 'hit' || m.res.kind === 'edge') {
          later(() => link.send({ t: 'run', b: m.b, k: 1, tick: 12 }));
          later(() => link.send({ t: 'ballDone', b: m.b, cmds: [[1, 12]] }));
        }
        break;
      }
      case 'result':
        st.balls++;
        st.snap = m.snap;
        if (m.snap.phase === 'inningsBreak') later(() => link.send({ t: 'continue' }));
        if (m.snap.phase === 'done') {
          st.done = true;
          st.final = m.snap;
        }
        later(() => link.send({ t: 'ready', b: m.b }));
        break;
      case 'error':
        st.errors.push(m.m);
        break;
      default:
        break;
    }
  };
  return st;
}

async function waitFor(cond, ms, what) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    if (cond()) return;
    await sleep(20);
  }
  throw new Error('timed out waiting for ' + what);
}

function compare(a, b, label) {
  const ja = JSON.stringify(a.final.inn);
  const jb = JSON.stringify(b.final.inn);
  if (ja !== jb) throw new Error(label + ': scorecards differ');
  const I0 = a.final.inn[0];
  const I1 = a.final.inn[1];
  console.log(`  ${label}: ${a.final.result.text} — ${I0.runs}/${I0.wkts} v ${I1.runs}/${I1.wkts}, ${a.balls} balls seen by each, turns ${a.turns}+${b.turns}`);
}

async function testLocal() {
  const room = new RoomCore({ code: 'LOCAL' });
  const a = new LocalLink(room, 0, 'Asha');
  const sa = scripted(a, 'A', { autostart: true });
  const b = new LocalLink(room, 1, 'Ben');
  const sb = scripted(b, 'B');
  await waitFor(() => sa.done && sb.done, 20000, 'local match');
  compare(sa, sb, 'in-page link');
}

async function testWs() {
  const port = 3900 + Math.floor(Math.random() * 90);
  const proc = spawn(process.execPath, [path.join(ROOT, 'server/index.js')], { env: { ...process.env, PORT: String(port) }, stdio: ['ignore', 'pipe', 'inherit'] });
  try {
    await new Promise((resolve, reject) => {
      proc.stdout.on('data', (d) => d.toString().includes('running') && resolve());
      proc.on('exit', () => reject(new Error('server exited')));
    });
    const url = `ws://127.0.0.1:${port}/ws`;
    const a = new WsLink(url);
    const sa = scripted(a, 'A', { autostart: true });
    a.open({ t: 'create', name: 'Asha' });
    await waitFor(() => a.code, 3000, 'room code');
    const b = new WsLink(url);
    const sb = scripted(b, 'B');
    b.open({ t: 'join', code: a.code, name: 'Ben' });

    // Drop player B's socket mid-match; it should rejoin with its token and carry on.
    await waitFor(() => sb.balls >= 5, 20000, 'five balls');
    b.ws.close();
    await waitFor(() => sa.done && sb.done, 30000, 'ws match');
    compare(sa, sb, 'WebSocket server (with a reconnect)');

    const res = await fetch(`http://127.0.0.1:${port}/health`).then((r) => r.json());
    if (!res.ok) throw new Error('health check failed');
    const bad = new WsLink(url);
    const sbad = scripted(bad, 'X');
    bad.open({ t: 'join', code: 'ZZZZZ', name: 'Nobody' });
    await waitFor(() => sbad.errors.length, 3000, 'join error');
    console.log(`  bad code rejected: "${sbad.errors[0]}"`);
    a.close();
    b.close();
    bad.close();
  } finally {
    proc.kill();
  }
}

// In-memory stand-in for the artifact room capability: presence only, coalesced, delayed.
function mockRoomHub() {
  const rooms = new Map();
  let seq = 0;
  const flush = (room) => {
    room.timer = null;
    for (const [id, member] of room.members) {
      const peers = Object.freeze([...room.members.entries()].map(([pid, m]) => Object.freeze({
        peer: pid, by: null, isMe: pid === id, sameTab: pid === id, kind: 'viewer', guest: false, presence: m.presence, updatedAt: Date.now(),
      })));
      for (const fn of member.listeners) fn({ peers, joined: [], left: [], updated: [] });
    }
  };
  const schedule = (room) => {
    if (!room.timer) room.timer = setTimeout(() => flush(room), 5 + Math.random() * 40);
  };
  return {
    api() {
      const id = 'peer' + ++seq;
      return {
        async join(name) {
          if (!rooms.has(name)) rooms.set(name, { members: new Map(), timer: null });
          const room = rooms.get(name);
          const member = { presence: Object.freeze({}), listeners: new Set() };
          room.members.set(id, member);
          schedule(room);
          return {
            name,
            presence: async (patch) => {
              const next = { ...member.presence };
              for (const k in patch) {
                if (patch[k] === null) delete next[k];
                else next[k] = patch[k];
              }
              const size = JSON.stringify(next).length;
              if (size > 4096) throw new Error('presence over 4 KiB: ' + size);
              member.presence = Object.freeze(next);
              schedule(room);
            },
            onPeers: (fn) => {
              member.listeners.add(fn);
              schedule(room);
              return () => member.listeners.delete(fn);
            },
            peers: () => [],
            leave: async () => {
              room.members.delete(id);
              schedule(room);
            },
          };
        },
      };
    },
  };
}

async function testPresence() {
  const hub = mockRoomHub();
  const host = new PresenceHost(hub.api(), 'Asha');
  const sa = scripted(host.local, 'A', { autostart: true });
  await host.open();
  const guest = new PresenceGuest(hub.api(), host.code, 'Ben');
  const sb = scripted(guest, 'B');
  let status = '';
  guest.onStatus = (s) => {
    status = s;
  };
  await guest.open();
  await waitFor(() => status === 'open', 5000, 'guest to connect');
  await waitFor(() => sa.done && sb.done, 90000, 'presence match');
  compare(sa, sb, 'artifact room (simulated)');
  host.close();
  guest.close();
}

const tests = [['in-page', testLocal], ['websocket', testWs], ['presence', testPresence]];
let failed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
  } catch (e) {
    failed++;
    console.error(`  ${name} FAILED:`, e.message);
  }
}
console.log(failed ? `${failed} transport test(s) failed` : 'all transport tests passed');
process.exit(failed ? 1 : 0);
