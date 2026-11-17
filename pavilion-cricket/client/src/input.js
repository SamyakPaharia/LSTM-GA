// Keyboard and touch input. Keys are tracked by physical code (KeyS, ArrowLeft, Space...).

const GAME_KEYS = new Set([
  'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Space', 'Tab', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyF',
  'KeyX', 'KeyZ', 'KeyC', 'KeyV', 'KeyM', 'KeyW', 'ShiftLeft', 'ShiftRight', 'Enter', 'NumpadEnter', 'Escape',
]);

const typing = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);

export class Input {
  constructor() {
    this.down = new Set();
    this.pressed = new Set();
    this.stick = { x: 0, y: 0 };
    this.loftHold = false;
    this.enabled = false;
    this.touchMode = false;

    window.addEventListener('keydown', (e) => {
      if (typing(e.target)) return;
      if (this.enabled && GAME_KEYS.has(e.code)) e.preventDefault();
      if (e.repeat) return;
      this.pressed.add(e.code);
      this.down.add(e.code);
    });
    window.addEventListener('keyup', (e) => this.down.delete(e.code));
    window.addEventListener('blur', () => this.down.clear());
  }

  isDown(code) {
    return this.down.has(code);
  }

  hit(code) {
    return this.pressed.has(code);
  }

  get shift() {
    return this.down.has('ShiftLeft') || this.down.has('ShiftRight') || this.loftHold;
  }

  // -1..1 on each axis; y is +1 for up.
  axis() {
    let x = (this.isDown('ArrowRight') ? 1 : 0) - (this.isDown('ArrowLeft') ? 1 : 0);
    let y = (this.isDown('ArrowUp') ? 1 : 0) - (this.isDown('ArrowDown') ? 1 : 0);
    if (x === 0 && y === 0) {
      x = this.stick.x;
      y = this.stick.y;
    }
    return { x, y };
  }

  // Eight-way direction for shots.
  dir8() {
    const a = this.axis();
    const dead = 0.35;
    return { x: a.x > dead ? 1 : a.x < -dead ? -1 : 0, y: a.y > dead ? 1 : a.y < -dead ? -1 : 0 };
  }

  virtualPress(code) {
    this.pressed.add(code);
  }

  endFrame() {
    this.pressed.clear();
  }

  // On-screen stick and buttons for touch devices.
  bindTouch(root, stickEl, knobEl, buttonsEl) {
    this.touchRoot = root;
    this.buttonsEl = buttonsEl;
    let id = null;
    const move = (t) => {
      const r = stickEl.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      let dx = (t.clientX - cx) / (r.width / 2);
      let dy = (t.clientY - cy) / (r.height / 2);
      const len = Math.hypot(dx, dy);
      if (len > 1) {
        dx /= len;
        dy /= len;
      }
      this.stick.x = dx;
      this.stick.y = -dy;
      knobEl.style.transform = `translate(${dx * 34}px, ${dy * 34}px)`;
    };
    const end = () => {
      id = null;
      this.stick.x = 0;
      this.stick.y = 0;
      knobEl.style.transform = '';
    };
    stickEl.addEventListener('pointerdown', (e) => {
      id = e.pointerId;
      stickEl.setPointerCapture(id);
      move(e);
      e.preventDefault();
    });
    stickEl.addEventListener('pointermove', (e) => e.pointerId === id && move(e));
    stickEl.addEventListener('pointerup', end);
    stickEl.addEventListener('pointercancel', end);
    window.addEventListener('touchstart', () => this.setTouchMode(true), { once: true, passive: true });
  }

  setTouchMode(on) {
    this.touchMode = on;
    document.body.classList.toggle('touch-mode', on);
    if (this.touchRoot) this.touchRoot.hidden = !on;
  }

  // buttons: [{label, code, hold?: 'loft', wide?: bool}]
  setButtons(buttons) {
    const el = this.buttonsEl;
    if (!el) return;
    const key = buttons.map((b) => b.label).join('|');
    if (el.dataset.key === key) return;
    el.dataset.key = key;
    el.textContent = '';
    this.loftHold = false;
    for (const b of buttons) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tbtn' + (b.wide ? ' wide' : '') + (b.accent ? ' accent' : '');
      btn.textContent = b.label;
      btn.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        if (b.hold === 'loft') {
          this.loftHold = !this.loftHold;
          btn.classList.toggle('on', this.loftHold);
        } else {
          this.virtualPress(b.code);
        }
      });
      el.appendChild(btn);
    }
  }
}
