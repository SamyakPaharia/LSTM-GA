// Tests dist/artifact.html the way a Claude artifact runs it: wrapped in a page skeleton,
// three.js from the CDN address (served here from node_modules), and the artifact's live
// room replaced by a BroadcastChannel stand-in shared by two pages. Plays a full online match.
// Usage: npm run build && node test/artifact-e2e.mjs [outDir]

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { AUTOPLAY } from './autoplay.mjs';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = process.argv[2] || path.join(ROOT, 'test-output');
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const body = fs.readFileSync(path.join(ROOT, 'dist/artifact.html'), 'utf8');
const page = `<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style></head><body>${body}</body></html>`;
const three = fs.readFileSync(path.join(ROOT, 'node_modules/three/build/three.min.js'));

// Presence-only stand-in for claude.use('room'), shared across pages of one browser context.
const MOCK_ROOM = `(() => {
  const me = Math.random().toString(36).slice(2, 12);
  const room = {
    async join(name) {
      const ch = new BroadcastChannel('mock-room-' + name);
      const members = new Map([[me, Object.freeze({})]]);
      const listeners = new Set();
      let timer = null;
      const snapshot = () => Object.freeze([...members.entries()].map(([peer, presence]) => Object.freeze({ peer, presence, by: null, isMe: peer === me, sameTab: peer === me, kind: 'viewer', guest: false, updatedAt: Date.now() })));
      const deliver = () => { if (timer) return; timer = setTimeout(() => { timer = null; const peers = snapshot(); for (const l of listeners) l({ peers, joined: [], left: [], updated: [] }); }, 10 + Math.random() * 30); };
      ch.onmessage = (e) => {
        const m = e.data;
        if (m.t === 'pres') members.set(m.peer, Object.freeze(m.pres));
        if (m.t === 'hello') ch.postMessage({ t: 'pres', peer: me, pres: members.get(me) });
        if (m.t === 'bye') members.delete(m.peer);
        deliver();
      };
      ch.postMessage({ t: 'hello' });
      return {
        name,
        async presence(patch) {
          const next = { ...members.get(me) };
          for (const k in patch) { if (patch[k] === null) delete next[k]; else next[k] = patch[k]; }
          if (new TextEncoder().encode(JSON.stringify(next)).length > 4096) throw { code: 'invalid_argument', message: 'presence over 4 KiB' };
          members.set(me, Object.freeze(next));
          ch.postMessage({ t: 'pres', peer: me, pres: next });
          deliver();
        },
        onPeers(fn) { listeners.add(fn); deliver(); return () => listeners.delete(fn); },
        peers: snapshot,
        connected: () => true,
        async leave() { ch.postMessage({ t: 'bye', peer: me }); ch.close(); },
      };
    },
  };
  window.claude = { use: async (n) => (n === 'room' ? room : null) };
})();`;

async function open(ctx, errors, label) {
  const p = await ctx.newPage();
  p.on('pageerror', (e) => errors.push(`${label}: ${e.message}`));
  p.on('console', (m) => {
    if (m.type() === 'error' && !/fonts\.g|net::|Failed to load/.test(m.text())) errors.push(`${label}: ${m.text()}`);
  });
  await p.route('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js', (r) => r.fulfill({ body: three, contentType: 'text/javascript' }));
  await p.route('https://fonts.googleapis.com/**', (r) => r.fulfill({ body: '', contentType: 'text/css' }));
  await p.route('http://artifact.test/', (r) => r.fulfill({ body: page, contentType: 'text/html' }));
  await p.addInitScript(MOCK_ROOM);
  await p.goto('http://artifact.test/');
  await p.waitForFunction(() => window.__pavilion, null, { timeout: 10000 });
  return p;
}

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 640, height: 360 } });
const errors = [];
let failed = false;
try {
  const a = await open(ctx, errors, 'host');
  const b = await open(ctx, errors, 'guest');
  await a.fill('#in-name', 'Asha');
  await b.fill('#in-name', 'Ben');
  await a.click('#go-online');
  await a.waitForFunction(() => !document.getElementById('online-actions').hidden, null, { timeout: 5000 });
  await a.click('#on-create');
  await a.waitForFunction(() => /^[A-Z]{5}$/.test(document.getElementById('lobby-code').textContent), null, { timeout: 5000 });
  const code = await a.textContent('#lobby-code');
  await b.click('#go-online');
  await b.waitForFunction(() => !document.getElementById('online-actions').hidden, null, { timeout: 5000 });
  await b.fill('#in-code', code);
  await b.click('#on-join');
  await a.waitForFunction(() => !document.getElementById('lobby-start').disabled, null, { timeout: 15000 });
  await a.setViewportSize({ width: 1280, height: 720 });
  await sleep(400);
  await a.screenshot({ path: path.join(OUT, 'artifact-lobby.png') });
  await a.setViewportSize({ width: 640, height: 360 });
  await a.click('#lobby-overs button[data-v="2"]');
  await sleep(300);
  await a.click('#lobby-start');
  await a.waitForFunction(() => window.__pavilion.state.game, null, { timeout: 15000 });
  await b.waitForFunction(() => window.__pavilion.state.game, null, { timeout: 15000 });
  await a.evaluate(AUTOPLAY);
  await b.evaluate(AUTOPLAY);
  const start = Date.now();
  let done = false;
  while (Date.now() - start < 8 * 60000) {
    const [sa, sb] = await Promise.all([a, b].map((p) => p.evaluate(() => {
      const g = window.__pavilion.state.game;
      return g && g.snap ? { phase: g.snap.phase, inn: JSON.stringify(g.snap.inn), result: g.snap.result && g.snap.result.text } : null;
    })));
    if (sa && sb && sa.phase === 'done' && sb.phase === 'done') {
      if (sa.inn !== sb.inn) throw new Error('the two screens disagree on the scorecard');
      console.log(`  artifact build, simulated live room: ${sa.result} — both screens agree`);
      done = true;
      break;
    }
    await sleep(250);
  }
  if (!done) throw new Error('match did not finish in time');
} catch (e) {
  failed = true;
  console.error('  FAILED:', e.message);
} finally {
  await browser.close();
}
if (errors.length) for (const e of [...new Set(errors)].slice(0, 10)) console.error('   ', e);
console.log(failed || errors.length ? 'artifact test FAILED' : 'artifact test passed');
process.exit(failed || errors.length ? 1 : 0);
