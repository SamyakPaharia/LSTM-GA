// Shared by the browser tests.
// Runs inside the page: plays sensibly so the match moves along without a person.
export const AUTOPLAY = `(() => {
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
