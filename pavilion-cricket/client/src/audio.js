// All sound is synthesised with WebAudio; there are no audio files.

export class Sound {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.master = null;
    this.crowdGain = null;
    this.crowdBase = 0.05;
  }

  // Must be called from a user gesture.
  unlock() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    try {
      this.ctx = new AC();
    } catch {
      return;
    }
    this.master = this.ctx.createGain();
    this.master.gain.value = this.muted ? 0 : 0.9;
    this.master.connect(this.ctx.destination);
    this.noise = this.makeNoise(2);
    this.startCrowd();
  }

  makeNoise(seconds) {
    const n = Math.floor(this.ctx.sampleRate * seconds);
    const buf = this.ctx.createBuffer(1, n, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    let last = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.random() * 2 - 1;
      last = (last + 0.04 * w) / 1.04;
      d[i] = last * 3.2 + w * 0.12;
    }
    return buf;
  }

  startCrowd() {
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    src.loop = true;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 700;
    bp.Q.value = 0.4;
    this.crowdGain = this.ctx.createGain();
    this.crowdGain.gain.value = this.crowdBase;
    src.connect(bp).connect(this.crowdGain).connect(this.master);
    src.start();
  }

  setMuted(m) {
    this.muted = m;
    if (this.master) this.master.gain.value = m ? 0 : 0.9;
  }

  burst({ dur = 0.1, freq = 1200, q = 1, gain = 0.5, type = 'bandpass', decay = 0.08, rate = 1 }) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = rate;
    const f = this.ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + decay);
    src.connect(f).connect(g).connect(this.master);
    src.start(t, Math.random() * 1.5, dur);
  }

  tone({ freq = 200, to = 80, dur = 0.12, gain = 0.4, type = 'sine' }) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    o.frequency.exponentialRampToValueAtTime(Math.max(20, to), t + dur);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g).connect(this.master);
    o.start(t);
    o.stop(t + dur + 0.02);
  }

  bat(power = 1) {
    this.tone({ freq: 520 + 260 * power, to: 150, dur: 0.07, gain: 0.25 + 0.35 * power, type: 'triangle' });
    this.burst({ freq: 2400, q: 0.8, gain: 0.25 + 0.4 * power, decay: 0.05 });
  }

  edge() {
    this.burst({ freq: 3800, q: 2, gain: 0.3, decay: 0.03 });
  }

  bounce() {
    this.tone({ freq: 160, to: 70, dur: 0.06, gain: 0.25 });
  }

  stumps() {
    for (let i = 0; i < 4; i++) setTimeout(() => this.tone({ freq: 700 + Math.random() * 500, to: 260, dur: 0.06, gain: 0.3, type: 'square' }), i * 35);
    this.burst({ freq: 1500, q: 0.6, gain: 0.4, decay: 0.12 });
  }

  glove() {
    this.tone({ freq: 240, to: 110, dur: 0.07, gain: 0.3 });
    this.burst({ freq: 900, q: 0.7, gain: 0.2, decay: 0.05 });
  }

  click() {
    this.tone({ freq: 880, to: 660, dur: 0.04, gain: 0.12, type: 'triangle' });
  }

  coin() {
    this.tone({ freq: 1800, to: 2400, dur: 0.25, gain: 0.12, type: 'sine' });
  }

  // Crowd swell: size 0..1.
  cheer(size = 0.6, dur = 2.4) {
    if (!this.ctx || !this.crowdGain) return;
    const t = this.ctx.currentTime;
    const g = this.crowdGain.gain;
    g.cancelScheduledValues(t);
    g.setValueAtTime(Math.max(g.value, this.crowdBase), t);
    g.linearRampToValueAtTime(this.crowdBase + 0.5 * size, t + 0.25);
    g.linearRampToValueAtTime(this.crowdBase, t + dur);
    this.burst({ dur: dur, freq: 1500, q: 0.3, gain: 0.12 * size, decay: dur, rate: 1.3 });
  }

  groan() {
    this.cheer(0.25, 1.4);
  }
}
