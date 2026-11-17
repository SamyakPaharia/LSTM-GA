// Browser test: plays a match against the computer with an in-page autoplayer, and an online
// match between two pages through the game server. Saves screenshots.
// Usage: node test/e2e.mjs [outDir] [--solo-only|--online-only] [--overs N]

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const OUT = args.find((a) => !a.startsWith('--') && !/^\d+$/.test(a)) || path.join(ROOT, 'test-output');
const OVERS = +(args[args.indexOf('--overs') + 1] || 2) || 2;
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const SMALL = { width: 640, height: 360 };
const BIG = { width: 1280, height: 720 };

// Software rendering is slow, so play at a small size and only render large for screenshots.
async function snap(page, name) {
  await page.setViewportSize(BIG);
  await sleep(450);
  await page.screenshot({ path: path.join(OUT, name) });
  await page.setViewportSize(SMALL);
}

// Runs inside the page: plays sensibly so the match moves along without a person.
const AUTOPLAY = `(() => {
  const P = window.__pavilion;
  const key = (code, type = 'keydown') => window.dispatchEvent(new KeyboardEvent(type, { code, bubbles: true }));
  const tap = (code) => { key(code); setTimeout(() => key(code, 'keyup'), 30); };
  window.__auto = { balls: 0, shots: 0, runs: 0, bowled: 0, log: [] };
  let lastBall = null;
  function tick() {
    const g = P.state.game;
    if (g) {
      const ov = document.getElementById('ov');
      if (!ov.hidden && !g.ball) {
        const pick = ov.querySelector('.pick.sel');
        const primary = ov.querySelector('.btn.primary');
        const title = (ov.querySelector('h2') || {}).textContent || '';
        if (pick) pick.click();
        else if (primary && !/rematch/i.test(primary.textContent) && !window.__autoStopAtResult) primary.click();
        else if (primary && !/rematch/i.test(primary.textContent)) primary.click();
      }
      const b = g.ball;
      if (g.mode === 'aim' && !b && ov.hidden) {
        if (Math.random() < 0.5) tap('KeyE');
        tap('Space');
        window.__auto.bowled++;
      }
      if (b && b.local && !b.released && b.effort == null && b.time / b.runT >= 0.86) tap('Space');
      if (b && g.amBatting() && b.released && !b.shot && b.d && b.clock >= b.tFront - 0.14 - 0.01) {
        const dirs = ['ArrowUp', 'ArrowLeft', 'ArrowRight', null];
        const d = dirs[Math.floor(Math.random() * dirs.length)];
        if (d) key(d);
        tap(Math.random() < 0.7 ? 'KeyS' : 'KeyA');
        if (d) setTimeout(() => key(d, 'keyup'), 60);
        window.__auto.shots++;
      }
      if (b && g.amBatting() && b.rp && !b.rp.dead && b.rp.idle && b.advice > b.rp.completed && b.rpTime > 0.2) tap('Space');
      if (b && b.b != null && b.b !== lastBall) { lastBall = b.b; window.__auto.balls++; }
    }
    requestAnimationFrame(tick);
  }
  tick();
})();`;

async function startServer() {
  const port = 4100 + Math.floor(Math.random() * 300);
  const proc = spawn(process.execPath, [path.join(ROOT, 'server/index.js')], { env: { ...process.env, PORT: String(port) }, stdio: ['ignore', 'pipe', 'inherit'] });
  await new Promise((resolve, reject) => {
    proc.stdout.on('data', (d) => d.toString().includes('running') && resolve());
    proc.on('exit', () => reject(new Error('server exited')));
  });
  return { proc, url: `http://127.0.0.1:${port}/` };
}

function watch(page, label, errors) {
  page.on('pageerror', (e) => errors.push(`${label} pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' && !/fonts\.g|ERR_TUNNEL|ERR_PROXY|net::|Failed to load resource/.test(m.text())) errors.push(`${label} console: ${m.text()}`);
  });
}

async function waitFor(page, fn, ms, what, arg) {
  try {
    await page.waitForFunction(fn, arg, { timeout: ms, polling: 100 });
  } catch {
    throw new Error('timed out: ' + what);
  }
}

async function solo(browser, base, errors) {
  const page = await browser.newPage({ viewport: SMALL });
  watch(page, 'solo', errors);
  await page.goto(base);
  await waitFor(page, () => window.__pavilion, 10000, 'app start');
  await sleep(800);
  await snap(page, '01-title.png');
  await page.fill('#in-name', 'Tester');
  await page.click('#go-solo');
  await page.click(`#pick-overs button[data-v="${OVERS}"]`);
  await snap(page, '02-setup.png');
  await page.click('#solo-start');
  await waitFor(page, () => !document.getElementById('ov').hidden, 5000, 'toss overlay');
  await sleep(300);
  await snap(page, '03-toss.png');
  // Bat first if we won the toss.
  const bat = await page.$('#ov .btn.primary');
  if (bat && /bat/i.test(await bat.textContent())) await bat.click();
  await page.evaluate(AUTOPLAY);

  const shots = { batRunup: false, batFollow: false, bowlAim: false, bowlRunup: false, card: false };
  const start = Date.now();
  const limit = (OVERS * 2 * 6 + 6) * 11000;
  while (Date.now() - start < limit) {
    const s = await page.evaluate(() => {
      const g = window.__pavilion.state.game;
      if (!g) return { none: true };
      const b = g.ball;
      return {
        phase: g.snap && g.snap.phase, batting: g.amBatting(), mode: g.mode, cur: g.snap ? g.snap.cur : -1,
        released: !!(b && b.released), clock: b ? b.clock : 0, plan: !!(b && b.plan), rpTime: b ? b.rpTime : 0,
        runup: !!(b && !b.released && b.relAt - b.time < 0.5), result: g.snap && g.snap.result ? g.snap.result.text : null,
      };
    });
    if (s.none) break;
    if (s.batting && s.runup && !shots.batRunup) {
      shots.batRunup = true;
      await snap(page, '04-batting-delivery.png');
    }
    if (s.plan && s.rpTime > 0.7 && !shots.batFollow) {
      shots.batFollow = true;
      await snap(page, '05-shot-follow.png');
    }
    if (!s.batting && s.mode === 'aim' && !shots.bowlAim) {
      await sleep(200);
      shots.bowlAim = true;
      await snap(page, '06-bowling-aim.png');
    }
    if (!s.batting && s.runup && !shots.bowlRunup) {
      shots.bowlRunup = true;
      await snap(page, '07-bowling-runup.png');
    }
    if (s.cur === 1 && !shots.card && !s.released) {
      shots.card = true;
      await page.keyboard.press('Tab');
      await sleep(250);
      await snap(page, '08-scorecard.png');
      await page.keyboard.press('Tab');
    }
    if (s.phase === 'done') {
      await sleep(1500);
      await snap(page, '09-result.png');
      const auto = await page.evaluate(() => window.__auto);
      const final = await page.evaluate(() => window.__pavilion.state.game.snap);
      console.log(`  solo: ${s.result} — ${final.inn.map((I) => `${I.runs}/${I.wkts} (${Math.floor(I.balls / 6)}.${I.balls % 6})`).join(' v ')}; autoplayer saw ${auto.balls} balls, played ${auto.shots} shots`);
      await page.close();
      return;
    }
    await sleep(60);
  }
  await snap(page, 'solo-timeout.png');
  throw new Error('solo match did not finish in time');
}

async function online(browser, base, errors) {
  const ctxA = await browser.newContext({ viewport: SMALL });
  const ctxB = await browser.newContext({ viewport: SMALL });
  const a = await ctxA.newPage();
  const b = await ctxB.newPage();
  watch(a, 'host', errors);
  watch(b, 'guest', errors);
  await a.goto(base);
  await b.goto(base);
  await waitFor(a, () => window.__pavilion, 10000, 'host app');
  await waitFor(b, () => window.__pavilion, 10000, 'guest app');
  await a.fill('#in-name', 'Asha');
  await b.fill('#in-name', 'Ben');
  await a.click('#go-online');
  await a.click('#on-create');
  await waitFor(a, () => /^[A-Z]{5}$/.test(document.getElementById('lobby-code').textContent), 5000, 'room code');
  const code = await a.textContent('#lobby-code');
  await b.click('#go-online');
  await b.fill('#in-code', code);
  await b.click('#on-join');
  await waitFor(a, () => !document.getElementById('lobby-start').disabled, 5000, 'guest to arrive');
  await b.click('#lobby-team button[data-id="ENG"]');
  await a.click('#lobby-overs button[data-v="2"]');
  await sleep(300);
  await snap(a, '10-lobby.png');
  await a.click('#lobby-start');
  await waitFor(a, () => window.__pavilion.state.game, 5000, 'host game');
  await waitFor(b, () => window.__pavilion.state.game, 5000, 'guest game');
  await sleep(300);
  for (const p of [a, b]) {
    const btn = await p.$('#ov .btn.primary');
    if (btn && /bat/i.test(await btn.textContent())) await btn.click();
  }
  await a.evaluate(AUTOPLAY);
  await b.evaluate(AUTOPLAY);
  const start = Date.now();
  let shot = false;
  while (Date.now() - start < 6 * 60000) {
    const [sa, sb] = await Promise.all([a, b].map((p) => p.evaluate(() => {
      const g = window.__pavilion.state.game;
      return g && g.snap ? { phase: g.snap.phase, inn: JSON.stringify(g.snap.inn), result: g.snap.result && g.snap.result.text, ball: !!g.ball, cur: g.snap.cur } : null;
    })));
    if (!shot && sa && sa.cur >= 0 && sa.ball) {
      await sleep(1200);
      shot = true;
      await snap(a, '11-online-host.png');
      await snap(b, '12-online-guest.png');
    }
    if (sa && sb && sa.phase === 'done' && sb.phase === 'done') {
      if (sa.inn !== sb.inn) throw new Error('online: the two screens disagree on the scorecard');
      console.log(`  online: ${sa.result} — both screens show the same scorecard`);
      await ctxA.close();
      await ctxB.close();
      return;
    }
    await sleep(200);
  }
  throw new Error('online match did not finish in time');
}

const { proc, url } = await startServer();
const browser = await chromium.launch();
const errors = [];
let failed = false;
try {
  if (!args.includes('--online-only')) await solo(browser, url, errors);
  if (!args.includes('--solo-only')) await online(browser, url, errors);
} catch (e) {
  failed = true;
  console.error('  FAILED:', e.message);
} finally {
  await browser.close();
  proc.kill();
}
if (errors.length) {
  console.error(`  ${errors.length} page error(s):`);
  for (const e of [...new Set(errors)].slice(0, 15)) console.error('   ', e);
}
console.log(failed || errors.length ? 'browser test FAILED' : 'browser test passed');
console.log('screenshots in', OUT);
process.exit(failed || errors.length ? 1 : 0);
