// GameClient: plays each ball from the host's messages, reads the player's input, and drives
// the 3D scene and HUD. It never decides an outcome; the host does.

import * as THREE from 'three';
import {
  HALF, DELIVERY_TIME_SCALE, FRONT_FOOT_Z, BACK_FOOT_Z, SWING_TIME, TICK, BALL_R, offSignFor, clamp, lerp,
} from '../../shared/constants.js';
import { ballAt, timeAtZ, typesFor, AIM_LIMITS, METER, isSpin, bowlingArm, styleName, typeName, lengthName } from '../../shared/delivery.js';
import { evaluateShot, bodyXFor, GUARD_LIMIT } from '../../shared/batting.js';
import { planFielding, fieldPositions, RunPhase, ballOnPath, FIELD_ORDER, FIELD_NAMES } from '../../shared/fielding.js';
import { cleanShot } from '../../shared/host.js';
import { Bot } from '../../shared/ai.js';
import { makeRng } from '../../shared/rng.js';
import { getTeam, hex } from '../../shared/teams.js';
import { oversText } from '../../shared/match.js';
import { Rig, Actor, makeKit, setKit, swingFor, BOWL_RELEASE } from './players.js';
import { describeBall } from './commentary.js';
import { esc, teamsOf } from './hud.js';

const PI = Math.PI;
const CLIP_T = 0.85;
const REL_IN_CLIP = BOWL_RELEASE * CLIP_T;
const Z_CLIP = -11.7;
const Z_REL = -9.15;
const Z_END = -7.3;
const STANCE_Z = 9.5;
const RUN_Z = 9.25;
const smooth = (t) => t * t * (3 - 2 * t);

const BAT_CAMS = [
  { pos: [0, 2.35, 16.6], look: [0, 0.85, -2] },
  { pos: [0, 5.2, 23], look: [0, 0.3, -1] },
];
const BOWL_CAMS = [
  { pos: [0, 3.2, -21.5], look: [0, 0.35, 6] },
  { pos: [0, 7, -30], look: [0, 0, 5] },
];

function runSpec(style) {
  if (isSpin(style)) return { len: 6.5, speed: 4.4 };
  if (style === 'RM' || !style) return { len: 11, speed: 6.3 };
  return { len: 15, speed: 7.3 };
}

function bowlerAt(tau, spec) {
  const tc = REL_IN_CLIP - tau;
  if (tc < 0) {
    const dist = Math.min(spec.len, -tc * spec.speed);
    return { z: Z_CLIP - dist, anim: dist >= spec.len - 0.01 ? 'stand' : 'jog', s: 0 };
  }
  if (tc <= CLIP_T) {
    const u = tc / CLIP_T;
    const z = u < BOWL_RELEASE ? lerp(Z_CLIP, Z_REL, u / BOWL_RELEASE) : lerp(Z_REL, Z_END, smooth((u - BOWL_RELEASE) / (1 - BOWL_RELEASE)));
    return { z, anim: 'bowl', s: u };
  }
  return { z: Z_END, anim: 'stand', s: 1 };
}

const TIMING_LABEL = { perfect: 'Perfect', good: 'Good', early: 'Early', late: 'Late', missed: 'Missed' };
const v3 = new THREE.Vector3();

export class GameClient {
  constructor({ scene, hud, input, sound, link, vsBot, onExit, onMute }) {
    this.S = scene;
    this.hud = hud;
    this.input = input;
    this.sound = sound;
    this.link = link;
    this.vsBot = vsBot;
    this.onExit = onExit;
    this.onMute = onMute;

    this.you = 0;
    this.names = ['', ''];
    this.kinds = ['human', 'human'];
    this.cfg = null;
    this.snap = null;
    this.latestSnap = null;
    this.teams = null;
    this.preset = 'attacking';
    this.buffer = new Map();
    this.ball = null;
    this.turn = null;
    this.mode = 'wait';
    this.camMode = 0;
    this.camCut = true;
    this.aim = { x: 0.15, z: 4.2 };
    this.typeIdx = 0;
    this.guard = 0;
    this.lastBall = 0;
    this.paused = false;
    this.breakShown = -1;
    this.resultShown = false;
    this.peerAway = false;
    this.field = [];
    this.orbit = 0;
    this.advisor = new Bot('medium', makeRng(99));

    this.group = new THREE.Group();
    this.group.visible = false;
    scene.scene.add(this.group);
    this.kits = { field: makeKit(), bat: makeKit() };
    this.fielders = [];
    for (let i = 0; i < 11; i++) {
      const a = new Actor(new Rig({ kit: this.kits.field, skin: i * 3 + 1, gloves: i === 0, pads: i === 0 }));
      this.fielders.push(a);
      this.group.add(a.rig.root);
    }
    this.batters = [0, 1].map((i) => {
      const a = new Actor(new Rig({ kit: this.kits.bat, skin: i * 2 + 2, bat: true, pads: true, helmet: true, gloves: true }), true);
      this.group.add(a.rig.root);
      return a;
    });
    this.umpires = [0, 1].map((i) => {
      const a = new Actor(new Rig({ kit: this.kits.field, skin: 3 + i * 2, umpire: true }));
      this.group.add(a.rig.root);
      return a;
    });
    this.actors = [...this.fielders, ...this.batters, ...this.umpires];
    this.camPos = scene.camera.position.clone();
    this.camLook = new THREE.Vector3(0, 0.5, 0);
    this.followDir = new THREE.Vector2(0, 1);
    hud.show(true);
    hud.comm('');
    hud.speed('');
    hud.hint([]);
    hud.status('Starting match…');
    input.enabled = true;
  }

  destroy() {
    this.S.scene.remove(this.group);
    this.S.setBall(null, false);
    this.S.aimMarker.visible = false;
    this.S.bounceMarker.visible = false;
    this.S.stumpsS.reset();
    this.S.stumpsN.reset();
    this.hud.show(false);
    this.hud.banner('');
    this.hud.meter(false);
    this.input.enabled = false;
    this.input.setButtons([]);
  }

  // ---- helpers -------------------------------------------------------------------------

  get I() {
    return this.snap && this.snap.cur >= 0 ? this.snap.inn[this.snap.cur] : null;
  }

  amBatting() {
    const I = this.I;
    return !!I && I.bat === this.you;
  }

  battingTeam() {
    return this.teams[this.I.bat];
  }

  bowlingTeam() {
    return this.teams[1 - this.I.bat];
  }

  seatName(seat) {
    if (this.kinds[seat] === 'bot') return 'The computer';
    return seat === this.you ? 'You' : this.names[seat] || 'Your opponent';
  }

  slot(b) {
    let s = this.buffer.get(b);
    if (!s) {
      s = { shot: null, res: null, cmds: [], dirty: false, result: null };
      this.buffer.set(b, s);
    }
    return s;
  }

  send(msg) {
    this.link.send(msg);
  }

  // ---- messages ------------------------------------------------------------------------

  onMessage(m) {
    switch (m.t) {
      case 'begin':
        this.you = m.you;
        this.names = m.names;
        this.kinds = m.kinds;
        this.cfg = m.cfg;
        break;
      case 'toss':
        this.onToss(m);
        break;
      case 'snap':
        this.latestSnap = m.snap;
        if (!this.ball) this.applySnap(m.snap);
        break;
      case 'field':
        this.preset = m.p;
        if (!this.ball && this.snap && this.snap.cur >= 0) this.placeField(true);
        break;
      case 'turn':
        this.turn = m;
        if (!this.ball) this.takeTurn();
        break;
      case 'delivery':
        this.onDelivery(m);
        break;
      case 'shot':
        this.slot(m.b).shot = m.s;
        break;
      case 'contact':
        this.slot(m.b).res = m.res;
        break;
      case 'run': {
        const s = this.slot(m.b);
        s.cmds.push([m.k, m.tick]);
        s.cmds.sort((a, b) => a[1] - b[1]);
        s.dirty = true;
        break;
      }
      case 'runs': {
        const s = this.slot(m.b);
        s.cmds = m.cmds.slice();
        s.dirty = true;
        break;
      }
      case 'result':
        this.slot(m.b).result = m;
        this.latestSnap = m.snap;
        if (!this.ball || this.ball.b !== m.b) {
          // A result for a ball we are not playing (e.g. after a resync): just take the state.
          if (!this.ball) this.applySnap(m.snap);
        }
        break;
      case 'void':
        if (this.ball && (this.ball.b === m.b || this.ball.b == null)) this.abortBall();
        break;
      case 'peer':
        this.peerAway = !m.on;
        this.hud.banner(m.on ? '' : `${this.names[m.seat] || 'Your opponent'} lost connection. Holding the match for 60 seconds…`);
        if (m.on) this.hud.flash('Opponent is back', 'small', 1400);
        break;
      default:
        break;
    }
  }

  onToss(m) {
    const teamOf = (s) => getTeam(this.cfg.teams[s]);
    if (m.quiet) return;
    if (m.choice === undefined) {
      this.mode = 'toss';
      this.hud.status('');
      this.sound.coin();
      if (m.win === this.you) {
        const pick = (c) => {
          this.sound.click();
          this.send({ t: 'tossChoice', c });
          this.hud.overlay({ title: 'You won the toss', sub: 'Sending your decision…' });
        };
        this.hud.overlay({
          title: 'You won the toss', sub: `${esc(teamOf(this.you).name)} v ${esc(teamOf(1 - this.you).name)} · ${this.cfg.overs} overs a side`,
          buttons: [{ label: 'Bat first', primary: true, onClick: () => pick('bat') }, { label: 'Bowl first', onClick: () => pick('bowl') }],
        });
      } else {
        this.hud.overlay({ title: `${esc(this.seatName(m.win))} won the toss`, sub: 'Waiting for their decision…' });
      }
      return;
    }
    const team = teamOf(m.win);
    this.hud.overlay({ title: `${esc(team.name)} won the toss`, sub: `and chose to ${m.choice === 'bat' ? 'bat' : 'bowl'} first.`, cls: 'toss' });
    setTimeout(() => {
      if (this.mode === 'toss') {
        this.hud.closeOverlay();
        this.mode = 'wait';
        this.send({ t: 'ready', b: 0 });
      }
    }, 2200);
  }

  applySnap(snap) {
    const prevCur = this.snap ? this.snap.cur : -1;
    this.snap = snap;
    this.teams = teamsOf(snap);
    if (snap.cur >= 0) {
      this.group.visible = true;
      if (snap.cur !== prevCur) {
        setKit(this.kits.field, this.bowlingTeam());
        setKit(this.kits.bat, this.battingTeam());
        this.preset = this.preset || 'attacking';
      }
      this.resetPositions();
    }
    this.hud.score(snap);
    this.checkPhase();
    this.refreshStatus();
  }

  checkPhase() {
    const snap = this.snap;
    if (!snap) return;
    if (snap.phase === 'inningsBreak' && this.breakShown !== snap.cur) {
      this.breakShown = snap.cur;
      this.mode = 'break';
      this.showBreak();
    } else if (snap.phase === 'done' && !this.resultShown) {
      this.resultShown = true;
      this.mode = 'done';
      this.showResult();
    } else if (snap.phase !== 'inningsBreak' && this.mode === 'break') {
      this.hud.closeOverlay();
      this.mode = 'wait';
    }
  }

  takeTurn() {
    const t = this.turn;
    if (!t || !this.snap || this.ball) return;
    if (t.seat !== this.you) {
      this.refreshStatus();
      return;
    }
    if (t.phase === 'pickBowler' && this.snap.phase === 'pickBowler') {
      this.turn = null;
      this.showPicker();
    } else if (t.phase === 'bowl' && this.snap.phase === 'bowl') {
      this.turn = null;
      this.enterAim();
    }
  }

  refreshStatus() {
    const snap = this.snap;
    if (!snap || snap.cur < 0 || this.ball) {
      if (this.ball) this.hud.status('');
      return;
    }
    if (snap.phase === 'pickBowler' || (snap.phase === 'bowl' && this.mode !== 'aim')) {
      const bowlSeat = 1 - snap.inn[snap.cur].bat;
      if (bowlSeat !== this.you) {
        const who = this.seatName(bowlSeat);
        this.hud.status(snap.phase === 'pickBowler' ? `${who} ${who === 'The computer' ? 'is' : 'is'} choosing a bowler…` : `${who} is about to bowl`);
        this.updateHints();
        return;
      }
    }
    this.hud.status('');
    this.updateHints();
  }

  // ---- overlays -----------------------------------------------------------------------

  showPicker() {
    const I = this.I;
    const team = this.bowlingTeam();
    const elig = this.snap.elig || [];
    const cap = Math.ceil(this.snap.cfg.overs / 5) * 6;
    const list = team.players.filter((p) => p.idx !== team.keeper).sort((a, b) => {
      const ea = elig.includes(a.idx) ? 1 : 0;
      const eb = elig.includes(b.idx) ? 1 : 0;
      return eb - ea || b.bowl - a.bowl;
    });
    const items = list.map((p) => {
      const f = I.bw[p.idx];
      const ok = elig.includes(p.idx);
      const why = !ok ? (p.idx === I.last ? 'bowled last over' : f[0] >= cap ? 'overs used up' : '') : '';
      return {
        disabled: !ok,
        html: `<span class="pk-name">${esc(p.name)}</span><span class="pk-style">${esc(styleName(p.style))}</span><span class="pk-bar"><i style="width:${p.bowl}%"></i></span><span class="pk-fig">${oversText(f[0])}-${f[1]}-${f[2]}${why ? ` · ${why}` : ''}</span>`,
      };
    });
    this.mode = 'pick';
    this.hud.listOverlay({
      title: 'Choose your bowler', sub: `Over ${Math.floor(I.balls / 6) + 1} of ${this.snap.cfg.overs} · each bowler can bowl ${cap / 6} over${cap > 6 ? 's' : ''}`,
      items,
      onPick: (i) => {
        this.sound.click();
        this.send({ t: 'pickBowler', i: list[i].idx });
        this.hud.closeOverlay();
        this.mode = 'wait';
        this.typeIdx = 0;
      },
    });
  }

  showBreak() {
    const snap = this.snap;
    const I0 = snap.inn[0];
    const t1 = this.teams[I0.bat];
    const t2 = this.teams[1 - I0.bat];
    const top = this.topPerformers(0);
    const body = `<div class="summary"><div class="sum-row"><span>${esc(t1.name)}</span><b>${I0.runs}/${I0.wkts}</b><span>${oversText(I0.balls)} ov</span></div>${top}</div>`;
    const onGo = () => {
      this.sound.click();
      this.send({ t: 'continue' });
      this.hud.overlay({ title: 'Innings break', sub: this.vsBot ? 'Starting the chase…' : 'Waiting for your opponent…' });
    };
    this.hud.overlay({
      title: 'Innings break', sub: `${esc(t2.name)} need <b>${I0.runs + 1}</b> to win from ${snap.cfg.overs * 6} balls.`, body,
      buttons: [{ label: 'Start the chase', primary: true, onClick: onGo }, { label: 'Scorecard', onClick: () => this.hud.showCard(true, this.snap) }],
    });
  }

  topPerformers(idx) {
    const I = this.snap.inn[idx];
    const bat = this.teams[I.bat];
    const bowl = this.teams[1 - I.bat];
    const bats = I.bt.map((r, i) => [i, r]).filter(([, r]) => r[1] > 0).sort((a, b) => b[1][0] - a[1][0]).slice(0, 2);
    const bowls = I.bw.map((r, i) => [i, r]).filter(([, r]) => r[0] > 0).sort((a, b) => b[1][2] - a[1][2] || a[1][1] - b[1][1]).slice(0, 2);
    return `<div class="sum-cols"><div>${bats.map(([i, r]) => `<p>${esc(bat.players[i].short)} <b>${r[0]}</b> <small>(${r[1]})</small></p>`).join('')}</div><div>${bowls.map(([i, r]) => `<p>${esc(bowl.players[i].short)} <b>${r[2]}-${r[1]}</b> <small>(${oversText(r[0])})</small></p>`).join('')}</div></div>`;
  }

  showResult() {
    const snap = this.snap;
    const rows = snap.inn.map((I, i) => `<div class="sum-row"><span>${esc(this.teams[I.bat].name)}</span><b>${I.runs}/${I.wkts}</b><span>${oversText(I.balls)} ov</span></div>${this.topPerformers(i)}`).join('');
    const won = snap.result.winner === this.you;
    const tie = snap.result.winner === -1;
    if (!tie) this.sound.cheer(won ? 1 : 0.4, 3);
    this.hud.overlay({
      title: esc(snap.result.text), sub: tie ? 'Honours even.' : won ? 'Well played — that one is yours.' : 'Not your day. Run it back?', body: `<div class="summary">${rows}</div>`,
      cls: 'result',
      buttons: [
        { label: 'Rematch', primary: true, onClick: () => this.send({ t: 'rematch' }) },
        { label: 'Scorecard', onClick: () => this.hud.showCard(true, this.snap) },
        { label: 'Main menu', onClick: () => this.onExit() },
      ],
    });
  }

  togglePause() {
    if (this.hud.overlayOpen && !this.paused) return;
    if (this.paused) {
      this.paused = false;
      this.hud.closeOverlay();
      return;
    }
    this.paused = this.vsBot;
    const controls = () => this.hud.overlay({
      title: 'Controls', body: controlsHtml(),
      buttons: [{ label: 'Back', primary: true, onClick: () => { this.paused = false; this.togglePause(); } }],
    });
    const quit = () => this.hud.overlay({
      title: 'Leave this match?', sub: this.vsBot ? 'Progress in this match will be lost.' : 'Your opponent will be awarded the match.',
      buttons: [
        { label: 'Leave match', onClick: () => { if (!this.vsBot) this.send({ t: 'leave' }); this.onExit(); } },
        { label: 'Keep playing', primary: true, onClick: () => { this.paused = false; this.hud.closeOverlay(); } },
      ],
    });
    this.hud.overlay({
      title: this.vsBot ? 'Paused' : 'Menu', sub: this.vsBot ? '' : 'The match keeps running while this menu is open.',
      buttons: [
        { label: 'Resume', primary: true, onClick: () => { this.paused = false; this.hud.closeOverlay(); } },
        { label: this.sound.muted ? 'Sound: off' : 'Sound: on', onClick: () => { this.onMute(); this.paused = false; this.togglePause(); } },
        { label: 'Controls', onClick: controls },
        { label: 'Leave match', onClick: quit },
      ],
    });
  }

  // ---- positions -----------------------------------------------------------------------

  currentBowler() {
    const I = this.I;
    const team = this.bowlingTeam();
    let idx = I.bowler >= 0 ? I.bowler : I.last;
    if (idx < 0) idx = team.players.reduce((best, p) => (p.bowl > team.players[best].bowl ? p.idx : best), 0);
    return team.players[idx];
  }

  resetPositions() {
    const I = this.I;
    if (!I) return;
    const bat = this.battingTeam();
    const striker = bat.players[I.st];
    const non = I.ns >= 0 ? bat.players[I.ns] : null;
    const bowler = this.currentBowler();
    const style = bowler.style || 'RM';
    const off = offSignFor(striker.hand);
    const arm = bowlingArm(style);
    const armSide = arm === 'R' ? 1 : -1;
    const spec = runSpec(style);
    this.field = fieldPositions(this.preset, off, isSpin(style), arm);

    const k = this.fielders[0];
    k.reach = null;
    k.place(this.field[0].x, this.field[0].z, PI);
    k.set('crouch');
    const bw = this.fielders[1];
    bw.reach = null;
    bw.look = null;
    bw.rig.setMirror(arm === 'L');
    bw.place(armSide * 0.95, Z_CLIP - spec.len, 0);
    bw.set('stand');
    for (let i = 2; i < 11; i++) {
      const f = this.fielders[i];
      const p = this.field[i];
      f.reach = null;
      f.place(p.x, p.z, Math.atan2(-p.x, STANCE_Z - p.z));
      f.look = [0, STANCE_Z];
      f.set('stand');
    }
    const [s, n] = this.batters;
    s.rig.setMirror(striker.hand === 'L');
    s.swing = null;
    s.place(bodyXFor(off, 0) - off * 0.07, STANCE_Z, (off * PI) / 2);
    s.set('stance');
    n.rig.root.visible = !!non;
    if (non) {
      n.rig.setMirror(non.hand === 'L');
      n.place(-armSide * 1.3, -9.7, 0);
      n.set('batIdle');
    }
    const [u1, u2] = this.umpires;
    u1.place(-armSide * 0.5, -12.1, 0);
    u1.set('umpire');
    u2.place(-off * 26, HALF - 1.22, off > 0 ? PI / 2 : -PI / 2);
    u2.set('umpire');
    this.S.stumpsS.reset();
    this.S.stumpsN.reset();
    this.S.setBall(null, false);
    this.S.aimMarker.visible = this.mode === 'aim';
    this.S.bounceMarker.visible = false;
    this.guard = 0;
  }

  placeField(walk) {
    const I = this.I;
    const striker = this.battingTeam().players[I.st];
    const bowler = this.currentBowler();
    const off = offSignFor(striker.hand);
    this.field = fieldPositions(this.preset, off, isSpin(bowler.style), bowlingArm(bowler.style || 'RM'));
    for (let i = 2; i < 11; i++) {
      const p = this.field[i];
      if (walk) this.fielders[i].goTo(p.x, p.z, 5);
      else this.fielders[i].place(p.x, p.z);
    }
    this.fielders[0].place(this.field[0].x, this.field[0].z, PI);
    this.updateHints();
  }

  // ---- bowling -------------------------------------------------------------------------

  enterAim() {
    this.mode = 'aim';
    const bowler = this.currentBowler();
    const types = typesFor(bowler.style || 'RM');
    if (this.typeIdx >= types.length) this.typeIdx = 0;
    if (isSpin(bowler.style)) this.aim.z = clamp(this.aim.z, 4.5, 8);
    this.S.aimMarker.visible = true;
    this.hud.status('');
    this.updateHints();
  }

  updateAim(dt) {
    const inp = this.input;
    const a = inp.axis();
    this.aim.z = clamp(this.aim.z + a.y * dt * 3.4, AIM_LIMITS.zMin, AIM_LIMITS.zMax);
    this.aim.x = clamp(this.aim.x - a.x * dt * 1.1, -AIM_LIMITS.x, AIM_LIMITS.x);
    this.S.aimMarker.position.set(this.aim.x, 0.015, this.aim.z);
    this.S.aimMarker.visible = true;
    const bowler = this.currentBowler();
    const types = typesFor(bowler.style || 'RM');
    if (inp.hit('KeyQ')) {
      this.typeIdx = (this.typeIdx + types.length - 1) % types.length;
      this.sound.click();
      this.updateHints();
    }
    if (inp.hit('KeyE')) {
      this.typeIdx = (this.typeIdx + 1) % types.length;
      this.sound.click();
      this.updateHints();
    }
    if (inp.hit('KeyF')) {
      const next = FIELD_ORDER[(FIELD_ORDER.indexOf(this.preset) + 1) % FIELD_ORDER.length];
      this.sound.click();
      this.send({ t: 'field', p: next });
    }
    if (inp.hit('Space')) this.startRunup();
  }

  startRunup() {
    const I = this.I;
    const bowler = this.currentBowler();
    const spec = runSpec(bowler.style || 'RM');
    const runT = spec.len / spec.speed;
    this.mode = 'ball';
    this.ball = this.newBall({ local: true, relAt: runT + REL_IN_CLIP, runT, bw: bowler.idx, st: I.st });
    this.updateHints();
  }

  newBall({ local, relAt, runT, bw, st }) {
    const bat = this.battingTeam();
    const bowl = this.bowlingTeam();
    const bowler = bowl.players[bw];
    const striker = bat.players[st];
    const style = bowler.style || 'RM';
    const arm = bowlingArm(style);
    return {
      b: null, local, time: 0, relAt, runT, effort: null, spec: runSpec(style), bowler, striker,
      off: offSignFor(striker.hand), armSide: arm === 'R' ? 1 : -1, d: null, released: false, clock: 0,
      slot: null, shot: null, res: null, predicted: false, confirmed: false, contactDone: false, bounced: false,
      plan: null, rp: null, rpTime: 0, ci: 0, moved: new Set(), deadAt: null, visualDone: false, endAt: null,
      resultApplied: false, afterUntil: 0, frozen: null, frozenAt: 0, kept: false, broke: false, padded: false,
      stX: bodyXFor(offSignFor(striker.hand), 0) - offSignFor(striker.hand) * 0.07, swingInfo: null, doneSent: false,
      throwsSeen: 0, field: this.field, celebrated: false,
    };
  }

  sendBowl(b) {
    const types = typesFor(b.bowler.style || 'RM');
    this.send({ t: 'bowl', type: types[this.typeIdx % types.length].id, x: this.aim.x, z: this.aim.z, e: b.effort, rem: Math.max(0.55, b.relAt - b.time) });
  }

  onDelivery(m) {
    let b = this.ball;
    if (!(b && b.local && b.b == null)) {
      if (b) this.abortBall();
      if (!this.snap || this.snap.cur < 0) return;
      const bowler = this.bowlingTeam().players[m.bw];
      const spec = runSpec(bowler.style || 'RM');
      const full = spec.len / spec.speed + REL_IN_CLIP;
      const relAt = m.rem != null ? Math.max(0.65, m.rem) : full;
      this.hud.closeOverlay();
      b = this.newBall({ local: false, relAt, runT: full - REL_IN_CLIP, bw: m.bw, st: m.st });
      this.ball = b;
    }
    if (m.f !== this.preset) {
      this.preset = m.f;
      this.placeField(false);
    }
    Object.assign(b, { b: m.b, d: m.d, seed: m.seed, fh: m.fh, lv: m.lv, fl: m.fl, field: fieldPositions(m.f, b.off, m.d.spin, m.d.arm) });
    b.slot = this.slot(m.b);
    b.tFront = timeAtZ(m.d, FRONT_FOOT_Z);
    b.tStumps = timeAtZ(m.d, HALF);
    b.tBounce = m.d.segs[1] ? m.d.segs[1].t0 : 99;
    this.mode = 'ball';
    this.hud.status('');
    this.updateHints();
  }

  release(b) {
    b.released = true;
    b.clock = Math.max(0, (b.time - b.relAt) * DELIVERY_TIME_SCALE);
    this.hud.meter(false);
    this.S.aimMarker.visible = false;
    const mk = this.S.bounceMarker;
    mk.position.set(b.d.bounce[0], 0.016, b.d.bounce[1]);
    mk.visible = true;
    mk.mat.opacity = 0.95;
    this.hud.comm('');
    this.hud.speed(`${b.d.kmh.toFixed(1)} km/h · ${typeName(b.d)} · ${lengthName(b.d)}`);
    if (b.d.noBall) this.hud.flash('No ball', 'small', 1000);
    if (b.fh) this.hud.flash('Free hit', 'small', 900);
    const n = this.batters[1];
    if (n.rig.root.visible) n.goTo(n.x, -8.3, 1.6);
    this.updateHints();
  }

  // ---- batting -------------------------------------------------------------------------

  updateGuard(inp) {
    if (inp.hit('KeyZ')) this.guard = clamp(this.guard - 0.15, -GUARD_LIMIT, GUARD_LIMIT);
    if (inp.hit('KeyC')) this.guard = clamp(this.guard + 0.15, -GUARD_LIMIT, GUARD_LIMIT);
    const b = this.ball;
    const s = this.batters[0];
    if (!b || !b.shot) {
      const off = b ? b.off : offSignFor(this.battingTeam().players[this.I.st].hand);
      const x = bodyXFor(off, this.guard) - off * 0.07;
      s.x += (x - s.x) * 0.3;
      if (b) b.stX = x;
    }
  }

  playShot(b, kind) {
    const inp = this.input;
    let shot;
    if (kind === 'none') {
      shot = { kind: 'none', t: b.clock, guard: this.guard };
    } else {
      const d8 = inp.dir8();
      let dx = d8.x;
      let dz = -d8.y;
      if (!dx && !dz) dz = -1;
      const len = Math.hypot(dx, dz);
      let foot = kind;
      if (kind === 'defend') foot = ballAt(b.d, b.tFront)[1] > 0.92 ? 'back' : 'front';
      shot = { kind: kind === 'defend' ? 'defend' : 'stroke', foot, loft: kind !== 'defend' && inp.shift, dir: [dx / len, dz / len], t: b.clock, guard: this.guard };
    }
    shot = cleanShot(shot);
    b.shot = shot;
    b.res = evaluateShot({ d: b.d, shot, batter: b.striker, bowlerSkill: b.bowler.bowl / 100, level: b.lv, seed: b.seed });
    b.predicted = true;
    this.send({ t: 'shot', b: b.b, s: shot });
    this.updateHints();
  }

  swingFrom(b) {
    const s = b.shot;
    if (!s || s.kind === 'none') return null;
    const phi = Math.atan2(s.dir[0] * b.off, -s.dir[1]);
    return swingFor(phi, s.foot, s.loft, s.kind === 'defend');
  }

  onContact(b) {
    b.contactDone = true;
    const r = b.res;
    const mine = this.amBatting();
    if (r.kind === 'hit' || r.kind === 'edge') {
      const sp = Math.hypot(r.launch[0], r.launch[1], r.launch[2]);
      if (r.kind === 'edge') this.sound.edge();
      else this.sound.bat(clamp(sp / 30, 0.15, 1));
      this.startField(b);
      if (r.timing !== 'none') {
        const cls = r.kind === 'edge' ? 'bad' : r.timing === 'perfect' ? 'perfect' : r.timing === 'good' ? 'good' : 'bad';
        this.hud.timing(`<b>${r.kind === 'edge' ? 'Edged' : TIMING_LABEL[r.timing]}</b>${esc(r.name)}`, cls);
      }
    } else if (r.kind === 'miss') {
      this.hud.timing(`<b>${mine ? 'Beaten' : 'Beaten!'}</b>${esc(TIMING_LABEL[r.timing] || '')}`, 'bad');
    }
  }

  startField(b) {
    const r = b.res;
    b.plan = planFielding({ pos: r.pos, vel: r.launch, field: b.field, seed: (b.seed ^ 0x5bd1e995) >>> 0, level: b.fl, edge: r.kind === 'edge' });
    b.prot = !!(b.fh || b.d.noBall);
    b.rp = new RunPhase(b.plan, { noCatchOut: b.prot });
    b.rpTime = 0;
    b.ci = 0;
    b.moved = new Set();
    b.advice = this.amBatting() ? this.advisor.runs(b.plan, b.prot).length : 0;
    this.updateHints();
  }

  // ---- per-frame ball playback ---------------------------------------------------------

  updateBall(dt, gi) {
    const b = this.ball;
    const inp = this.input;
    const batting = this.amBatting();
    b.time += dt;

    if (!b.released) {
      if (b.local && b.effort == null) {
        const m = Math.min(1, b.time / b.runT);
        this.hud.meter(true, m, METER.sweetLo, METER.sweetHi);
        if ((gi && inp.hit('Space')) || m >= 1) {
          b.effort = m;
          this.sendBowl(b);
          this.hud.meter(true, m, METER.sweetLo, METER.sweetHi, m);
          this.sound.click();
        }
      }
      if (batting && gi) this.updateGuard(inp);
      if (b.time >= b.relAt) {
        if (!b.d) b.time = b.relAt;
        else this.release(b);
      }
    }

    if (b.released) {
      const sl = b.slot;
      if (!batting && !b.res && sl.res) {
        b.res = sl.res;
        b.shot = sl.shot;
      }
      if (batting && b.predicted && sl.res && !b.confirmed) {
        b.confirmed = true;
        const same = sl.res.kind === b.res.kind && JSON.stringify(sl.res.launch) === JSON.stringify(b.res.launch);
        if (!same) {
          b.res = sl.res;
          if (b.plan) {
            const cmds = b.rp.cmdLog;
            this.startField(b);
            b.rp = RunPhase.replay(b.plan, cmds, Math.floor(b.rpTime / TICK), { noCatchOut: b.prot });
          }
        }
      }
      let rate = 1;
      if (!batting && !b.res) rate = clamp((b.tFront - 0.03 - b.clock) / 0.12, 0.05, 1);
      b.clock += dt * DELIVERY_TIME_SCALE * rate;

      if (batting && !b.shot) {
        if (gi && b.clock > 0.04) {
          const press = inp.hit('KeyS') ? 'front' : inp.hit('KeyA') ? 'back' : inp.hit('KeyD') ? 'defend' : null;
          if (press) this.playShot(b, press);
        }
        if (!b.shot && b.clock > b.tStumps + 0.03) this.playShot(b, 'none');
      }
      if (b.shot && !b.swingInfo) b.swingInfo = this.swingFrom(b);
      if (!b.bounced && b.clock >= b.tBounce) {
        b.bounced = true;
        this.sound.bounce();
      }
      if (b.res && !b.contactDone && b.clock >= b.res.t) this.onContact(b);
      if (b.contactDone && !b.plan) this.trackPass(b);
      const mk = this.S.bounceMarker;
      if (mk.visible) {
        mk.mat.opacity = Math.max(0, mk.mat.opacity - dt * 0.6);
        if (mk.mat.opacity <= 0) mk.visible = false;
      }
    }

    if (b.rp) this.updateField(b, dt, gi);

    if (b.visualDone && b.slot && b.slot.result && !b.resultApplied) this.applyResult(b);
    if (b.resultApplied && b.time >= b.afterUntil) this.finishBall();
  }

  trackPass(b) {
    const r = b.res;
    const p = ballAt(b.d, b.clock);
    const padZ = b.shot && b.shot.kind !== 'none' && b.shot.foot === 'front' ? FRONT_FOOT_Z + 0.2 : 9.15;
    if (!b.endAt) {
      if (r.stumps && !r.pad && b.clock >= b.tStumps) {
        b.broke = true;
        b.frozen = ballAt(b.d, b.tStumps);
        b.frozenAt = b.time;
        this.S.stumpsS.break(1, 1, (p[0] - 0) * 3);
        this.sound.stumps();
        this.celebrate();
        b.endAt = b.time + 1.4;
      } else if (r.pad && p[2] >= padZ) {
        b.padded = true;
        b.frozen = p.slice();
        b.frozenAt = b.time;
        this.sound.glove();
        if (r.dismissal === 'lbw') this.celebrate();
        b.endAt = b.time + 1.5;
      } else if (!r.stumps && !r.pad && p[2] >= this.fielders[0].z - 0.4) {
        b.kept = true;
        this.sound.glove();
        b.endAt = b.time + 0.8;
      } else if (b.clock > b.tStumps + 2) {
        b.endAt = b.time;
      }
    } else if (b.time >= b.endAt) {
      b.visualDone = true;
    }
  }

  updateField(b, dt, gi) {
    const inp = this.input;
    b.rpTime += dt;
    const target = Math.floor(b.rpTime / TICK + 1e-6);
    const opts = { noCatchOut: b.prot };
    if (this.amBatting()) {
      if (gi && !b.rp.dead) {
        if (inp.hit('Space') && b.rp.command(true)) this.send({ t: 'run', b: b.b, k: 1, tick: b.rp.tick });
        if (inp.hit('KeyX') && b.rp.command(false)) this.send({ t: 'run', b: b.b, k: 0, tick: b.rp.tick });
      }
      while (!b.rp.dead && b.rp.tick < target) b.rp.step();
      if (b.rp.dead && !b.doneSent) {
        b.doneSent = true;
        this.send({ t: 'ballDone', b: b.b, cmds: b.rp.cmdLog });
      }
    } else {
      const sl = b.slot;
      if (sl.result && sl.result.cmds && !b.finalCmds) {
        b.finalCmds = true;
        sl.cmds = sl.result.cmds.slice();
        sl.dirty = true;
      }
      if (sl.dirty) {
        sl.dirty = false;
        b.rp = RunPhase.replay(b.plan, sl.cmds, target, opts);
        b.ci = b.rp.cmdLog.length;
      } else {
        while (!b.rp.dead && b.rp.tick < target) {
          while (b.ci < sl.cmds.length && sl.cmds[b.ci][1] <= b.rp.tick) {
            b.rp.apply(sl.cmds[b.ci][0] === 1);
            b.ci++;
          }
          b.rp.step();
        }
      }
    }
    if (b.rp.dead) {
      if (b.deadAt == null) {
        b.deadAt = b.time;
        const rp = b.rp;
        if (rp.runOut) {
          (rp.runOut.end === 'S' ? this.S.stumpsS : this.S.stumpsN).break(rp.runOut.end === 'S' ? 1 : -1, 0.8);
          this.sound.stumps();
          this.celebrate();
        } else if (rp.caught) {
          this.celebrate();
        }
      }
      const hold = b.rp.boundary ? 1.3 : b.rp.caught || b.rp.runOut ? 1.1 : 0.5;
      if (b.time >= b.deadAt + hold) b.visualDone = true;
    } else {
      b.deadAt = null;
      b.visualDone = false;
    }
    this.updateHints();
  }

  celebrate() {
    this.fielders.forEach((f, i) => {
      if (i === 1 || Math.random() < 0.5) {
        f.reach = null;
        f.set('celebrate');
      }
    });
  }

  applyResult(b) {
    const r = b.slot.result;
    const o = r.o;
    b.resultApplied = true;
    this.lastBall = b.b;
    const ump = this.umpires[0];
    let after = 1.5;
    if (o.wicket) {
      const how = { bowled: 'Bowled', caught: 'Caught', lbw: 'LBW', runout: 'Run out' }[o.wicket.kind] || 'Out';
      this.hud.flash(how, 'out', 2200);
      ump.set('out');
      this.sound.cheer(0.9, 2.6);
      after = 2.8;
    } else if (o.boundary === 6) {
      this.hud.flash('Six', 'six', 2000);
      ump.set('six');
      this.sound.cheer(1, 3);
      after = 2.8;
    } else if (o.boundary === 4) {
      this.hud.flash('Four', 'four', 2000);
      ump.set('four');
      this.sound.cheer(0.7, 2.4);
      after = 2.6;
    } else if (o.wide) {
      this.hud.flash('Wide', 'small', 1300);
      ump.set('wide');
      after = 1.7;
    } else if (o.noBall) {
      this.hud.flash(o.runs ? `No ball · ${o.runs}` : 'No ball', 'small', 1300);
      ump.set('noball');
      after = 1.8;
    } else if (o.runs) {
      this.hud.flash(`${o.runs} run${o.runs > 1 ? 's' : ''}`, 'small', 1100);
      if (o.runs >= 2) this.sound.cheer(0.25, 1.2);
    } else if (o.kind === 'miss') {
      this.sound.groan();
    }
    const bowlTeam = this.bowlingTeam();
    const fielderName = o.by >= 0 ? bowlTeam.players[o.by].short : 'the fielder';
    let line = describeBall({ o, d: b.d, striker: b.striker, bowler: b.bowler, fielderName });
    for (const e of r.ev) {
      if (e.e === 'over') {
        const I = r.snap.inn[r.snap.cur] || r.snap.inn[r.snap.inn.length - 1];
        line += ` End of over ${e.n}: ${I.ov[I.ov.length - 1] ?? 0} from it.`;
      }
      if (e.e === 'milestone') {
        const p = this.battingTeam().players[e.who];
        setTimeout(() => this.hud.flash(`${e.mark === 100 ? 'Hundred' : 'Fifty'} · ${p.short}`, 'mile', 2200), 1800);
        after = Math.max(after, 3.6);
      }
    }
    this.hud.comm(line);
    this.hud.score(r.snap);
    b.afterUntil = b.time + after;
  }

  finishBall() {
    const b = this.ball;
    this.ball = null;
    if (b.b != null) this.buffer.delete(b.b);
    for (const k of this.buffer.keys()) if (k < this.lastBall) this.buffer.delete(k);
    this.camCut = true;
    this.hud.speed('');
    this.mode = 'wait';
    if (this.latestSnap) this.applySnap(this.latestSnap);
    this.send({ t: 'ready', b: b.b });
    if (this.turn) this.takeTurn();
  }

  abortBall() {
    this.ball = null;
    this.mode = 'wait';
    this.hud.meter(false);
    this.hud.speed('');
    if (this.latestSnap) this.applySnap(this.latestSnap);
    if (this.turn) this.takeTurn();
  }

  // ---- visuals -------------------------------------------------------------------------

  poseBall(b, dt) {
    const bw = this.fielders[1];
    const clipDone = b.time > b.relAt + CLIP_T * (1 - BOWL_RELEASE);
    if (!(b.plan && clipDone)) {
      const tau = b.relAt - b.time;
      const st = bowlerAt(tau, b.spec);
      bw.moving = false;
      bw.reach = null;
      bw.heading = 0;
      bw.x = b.armSide * (st.anim === 'bowl' ? lerp(0.95, 0.7, st.s) : 0.95);
      bw.z = st.z;
      if (!b.celebratedBy) bw.set(st.anim, st.s);
      if (st.anim === 'stand' && tau < 0) bw.look = [this.ballPos ? this.ballPos[0] : 0, this.ballPos ? this.ballPos[2] : 0];
    }

    const s = this.batters[0];
    const n = this.batters[1];
    const tau = b.relAt - b.time;
    const running = b.rp && (b.rp.completed > 0 || b.rp.dir !== 0 || b.rp.wait > 0);
    if (!running) {
      if (!b.released || !b.shot) {
        s.set(tau < 0.42 && tau > -1 ? 'backlift' : 'stance');
        if (b.released && b.clock > b.tFront - 0.2) s.set('leave');
        s.x += (b.stX - s.x) * Math.min(1, dt * 10);
      } else if (b.swingInfo) {
        const sv = clamp((b.clock - b.shot.t) / (2 * SWING_TIME), 0, 1);
        s.swing = b.swingInfo;
        s.set('swing', sv);
        const stride = b.swingInfo.stride * smooth(Math.min(1, sv * 2.2));
        s.z = STANCE_Z + (stride > 0 ? -stride : -stride * 0.8);
        s.x = b.shot.guard != null ? bodyXFor(b.off, b.shot.guard) - b.off * 0.07 : s.x;
      } else {
        s.set(b.clock > b.tFront - 0.2 ? 'leave' : 'backlift');
      }
    }

    // Keeper and fielders before the hit.
    const k = this.fielders[0];
    if (!b.plan) {
      if (b.kept) {
        k.reach = null;
        k.set('catch');
      } else if (b.released && b.clock > b.tStumps - 0.15 && this.ballPos && Math.abs(this.ballPos[2] - k.z) < 3) {
        k.reachFor(this.ballPos);
      } else if (!b.broke && !b.padded) {
        k.reach = null;
        k.set('crouch');
      }
      for (let i = 2; i < 11; i++) {
        const f = this.fielders[i];
        if (f.anim === 'celebrate') continue;
        f.set(tau < 0.8 ? 'ready' : 'stand');
      }
    } else {
      this.poseField(b);
    }
    if (running) this.poseRunners(b);
    else if (b.rp && b.rp.dead && !b.rp.boundary) n.set('batIdle');
  }

  poseField(b) {
    const plan = b.plan;
    const rp = b.rp;
    const t = b.rpTime;
    for (let j = 0; j < plan.moves.length; j++) {
      const mv = plan.moves[j];
      if (b.moved.has(j) || t < mv.t0) continue;
      b.moved.add(j);
      const f = this.fielders[mv.i];
      const dist = Math.hypot(mv.x - f.x, mv.z - f.z);
      f.reach = null;
      f.goTo(mv.x, mv.z, clamp(dist / Math.max(0.15, mv.t1 - mv.t0), 2.5, 9));
    }
    const ball = this.ballPos;
    for (let i = 0; i < 11; i++) {
      const f = this.fielders[i];
      if (i === 1 && b.time < b.relAt + CLIP_T * (1 - BOWL_RELEASE)) continue;
      if (f.anim === 'celebrate') continue;
      if (i === plan.fielder) continue;
      if (!f.moving) {
        f.look = ball ? [ball[0], ball[2]] : null;
        if (f.anim !== 'throw' || f.t > 0.6) f.set('stand');
      }
    }
    // Receivers at the stumps reach for an incoming throw.
    if (rp.ball === 'thrown' && rp.throws.length) {
      const th = rp.throws[rp.throws.length - 1];
      const recv = this.fielders[plan.recv[th.end].i];
      if (ball && Math.hypot(ball[0] - recv.x, ball[2] - recv.z) < 6) recv.reachFor(ball);
    } else if (!rp.dead) {
      for (const end of ['S', 'N']) this.fielders[plan.recv[end].i].reach = null;
    }
    // The chaser.
    if (plan.fielder >= 0) {
      const c = this.fielders[plan.fielder];
      if (rp.throws.length > b.throwsSeen) {
        b.throwsSeen = rp.throws.length;
        const th = rp.throws[rp.throws.length - 1];
        c.reach = null;
        c.heading = Math.atan2(th.x1 - c.x, th.z1 - c.z);
        c.set('throw');
        this.sound.click();
      }
      if (c.anim !== 'throw' && c.anim !== 'celebrate') {
        const air = plan.air && t < plan.touch + 0.05;
        if (air && ball && Math.hypot(ball[0] - c.x, ball[2] - c.z) < 4 && t > plan.touch - 0.6) {
          c.reachFor(ball);
        } else if (t >= plan.touch - 0.15 && t < plan.t + 0.4 && !plan.air) {
          c.reach = null;
          c.set('pick');
        } else if (t >= plan.t) {
          c.reach = null;
          c.set('stand');
          if (ball) c.look = null;
        }
      }
      if (!b.caughtSound && t >= plan.touch && (plan.outcome === 'caught' || plan.dropped)) {
        b.caughtSound = true;
        this.sound.glove();
        if (plan.dropped) this.hud.flash('Dropped!', 'small', 1100);
      }
    }
  }

  poseRunners(b) {
    const rp = b.rp;
    const even = rp.completed % 2 === 0;
    const pos = (startS) => {
      const fromS = startS ? even : !even;
      const from = fromS ? RUN_Z : -RUN_Z;
      const to = -from;
      return { z: from + (to - from) * rp.p, to };
    };
    const lane = 1.05;
    const moving = rp.dir !== 0 || rp.wait > 0;
    [[this.batters[0], true, lane], [this.batters[1], false, -lane]].forEach(([a, startS, lx]) => {
      if (!a.rig.root.visible) return;
      const p = pos(startS);
      a.moving = false;
      a.reach = null;
      a.x += (lx - a.x) * 0.12;
      a.z = p.z;
      if (moving && rp.dir !== 0) {
        a.heading = rp.dir > 0 ? (p.to > p.z ? 0 : PI) : p.to > p.z ? PI : 0;
        a.set('jog');
      } else {
        a.heading = p.z > 0 ? PI : 0;
        a.set('batIdle');
      }
    });
  }

  handPos(actor, low) {
    const h = actor.heading;
    return [actor.x + Math.sin(h) * 0.32, low ? 0.45 : 1.05, actor.z + Math.cos(h) * 0.32];
  }

  ballVisual(b) {
    if (!b.d) return null;
    if (!b.released) {
      const hand = this.fielders[1].rig.armR.hand;
      hand.getWorldPosition(v3);
      return [v3.x, v3.y, v3.z];
    }
    if (b.plan) {
      const plan = b.plan;
      const rp = b.rp;
      const t = b.rpTime;
      if (plan.outcome === 'boundary') return ballOnPath(plan, t);
      const chaser = plan.fielder >= 0 ? this.fielders[plan.fielder] : null;
      if (rp.ball === 'path') {
        if (t >= plan.touch && !plan.dropped && chaser) return this.handPos(chaser, !plan.air);
        return ballOnPath(plan, t);
      }
      if (rp.ball === 'held' || rp.ball === 'idle') return chaser ? this.handPos(chaser, false) : ballOnPath(plan, t);
      if (rp.ball === 'thrown' || rp.throws.length) {
        const th = rp.throws[rp.throws.length - 1];
        const u = clamp((t - th.t0) / Math.max(0.05, th.t1 - th.t0), 0, 1);
        const dist = Math.hypot(th.x1 - th.x0, th.z1 - th.z0);
        return [lerp(th.x0, th.x1, u), lerp(1.7, 0.6, u) + Math.sin(PI * u) * Math.min(7, dist * 0.07), lerp(th.z0, th.z1, u)];
      }
      return ballOnPath(plan, t);
    }
    if (b.kept) return this.handPos(this.fielders[0], false);
    if (b.frozen) {
      const u = clamp((b.time - b.frozenAt) / 0.45, 0, 1);
      const f = b.frozen;
      return [f[0], lerp(f[1], BALL_R, u * u), f[2] + (b.broke ? 0.4 : -0.3) * u];
    }
    return ballAt(b.d, b.clock);
  }

  // ---- camera --------------------------------------------------------------------------

  updateCamera(dt) {
    const cam = this.S.camera;
    let pos;
    let look;
    const b = this.ball;
    if (!this.snap || this.snap.cur < 0 || this.mode === 'toss') {
      this.orbit += dt * 0.08;
      pos = [Math.sin(this.orbit) * 52, 20, Math.cos(this.orbit) * 52];
      look = [0, 1, 0];
    } else if (b && b.plan && b.rpTime > 0.1 && this.ballPos) {
      const p = this.ballPos;
      const dist = Math.hypot(p[0], p[2]);
      if (dist > 5) {
        const k = 1 - Math.exp(-dt * 2);
        this.followDir.x += (p[0] / dist - this.followDir.x) * k;
        this.followDir.y += (p[2] / dist - this.followDir.y) * k;
        this.followDir.normalize();
      }
      const back = 18 + dist * 0.32;
      pos = [-this.followDir.x * back + p[0] * 0.25, 9 + dist * 0.2, -this.followDir.y * back + p[2] * 0.25];
      look = [p[0] * 0.8, 0.5, p[2] * 0.8];
    } else {
      const set = this.amBatting() ? BAT_CAMS[this.camMode] : BOWL_CAMS[this.camMode];
      pos = set.pos;
      look = set.look;
      if (!b || !b.plan) this.followDir.set(0, this.amBatting() ? -1 : 1);
    }
    const k = this.camCut ? 1 : 1 - Math.exp(-dt * 3.2);
    this.camCut = false;
    this.camPos.x += (pos[0] - this.camPos.x) * k;
    this.camPos.y += (pos[1] - this.camPos.y) * k;
    this.camPos.z += (pos[2] - this.camPos.z) * k;
    this.camLook.x += (look[0] - this.camLook.x) * k;
    this.camLook.y += (look[1] - this.camLook.y) * k;
    this.camLook.z += (look[2] - this.camLook.z) * k;
    cam.position.copy(this.camPos);
    cam.lookAt(this.camLook);
  }

  // ---- hints ---------------------------------------------------------------------------

  updateHints() {
    const hud = this.hud;
    const inp = this.input;
    const b = this.ball;
    const batting = this.amBatting();
    let hints = [];
    let buttons = [];
    if (this.mode === 'aim' && !b) {
      const bowler = this.currentBowler();
      const types = typesFor(bowler.style || 'RM');
      hints = [['Arrows', 'Line & length'], ['Q / E', types[this.typeIdx % types.length].name], ['F', `Field: ${FIELD_NAMES[this.preset]}`], ['Space', 'Run in']];
      buttons = [{ label: 'Q', code: 'KeyQ' }, { label: 'E', code: 'KeyE' }, { label: 'Field', code: 'KeyF' }, { label: 'Bowl', code: 'Space', wide: true, accent: true }];
    } else if (b && b.local && !b.released && b.effort == null) {
      hints = [['Space', 'Release — stop in the gold band']];
      buttons = [{ label: 'Release', code: 'Space', wide: true, accent: true }];
    } else if (b && batting && b.rp && !b.rp.dead) {
      const adv = b.advice ? `Looks like ${b.advice}` : 'Risky';
      hints = [['Space', 'Run'], ['X', 'Send back'], ['', `${adv} · ${b.rp.completed} done`]];
      buttons = [{ label: 'Run', code: 'Space', wide: true, accent: true }, { label: 'Back', code: 'KeyX' }];
    } else if (batting && (!b || !b.shot)) {
      hints = [['Arrows', 'Direction'], ['S', 'Front foot'], ['A', 'Back foot'], ['D', 'Defend'], ['Shift', 'Loft'], ['Z / C', 'Shuffle']];
      buttons = [{ label: '◀', code: 'KeyZ' }, { label: '▶', code: 'KeyC' }, { label: 'Loft', hold: 'loft' }, { label: 'Defend', code: 'KeyD' }, { label: 'Back ft', code: 'KeyA' }, { label: 'Drive', code: 'KeyS', accent: true }];
    }
    const key = JSON.stringify(hints);
    if (key !== this.hintKey) {
      this.hintKey = key;
      hud.hint(hints);
    }
    inp.setButtons(buttons);
  }

  // ---- main update ---------------------------------------------------------------------

  update(rawDt) {
    const inp = this.input;
    const dt = Math.min(rawDt, 0.05);
    if (inp.hit('Escape')) this.togglePause();
    if (inp.hit('Tab') && this.snap && this.snap.cur >= 0) this.hud.showCard(!this.hud.cardOpen, this.snap);
    if (inp.hit('KeyV')) this.camMode = 1 - this.camMode;
    if (inp.hit('KeyM')) this.onMute();
    const overlay = this.hud.overlayOpen;
    if (overlay) this.hud.navigate(inp);
    const gi = !overlay;
    const step = this.paused ? 0 : dt;

    if (step > 0) {
      if (this.mode === 'aim' && gi && !this.ball) this.updateAim(step);
      if (this.ball) this.updateBall(step, gi);
      else if (gi && this.amBatting() && this.snap && this.snap.phase !== 'done') this.updateGuard(inp);
      if (this.ball) {
        this.poseBall(this.ball, step);
      }
      for (const a of this.actors) a.update(step);
      if (this.ball) {
        this.ballPos = this.ballVisual(this.ball);
        this.S.setBall(this.ballPos, !!this.ballPos);
      } else {
        this.ballPos = null;
      }
      this.S.update(step);
    }
    this.updateCamera(dt);
    if (this.snap && this.snap.cur >= 0) {
      const bat = this.battingTeam();
      const bowl = this.bowlingTeam();
      this.hud.radar(this.fielders, hex(bowl.shirt), this.batters.filter((a) => a.rig.root.visible), hex(bat.shirt), this.ballPos);
    }
  }
}

export function controlsHtml() {
  return `<div class="controls">
    <div><h3>Bowling</h3>
      <p><kbd>Arrows</kbd> move the pitch marker</p><p><kbd>Q</kbd> <kbd>E</kbd> change delivery</p><p><kbd>F</kbd> change field</p>
      <p><kbd>Space</kbd> run in, then <kbd>Space</kbd> again in the gold band to release. Past the top is a no-ball.</p></div>
    <div><h3>Batting</h3>
      <p>Hold <kbd>Arrows</kbd> for direction (up is straight back past the bowler)</p><p><kbd>S</kbd> front-foot shot · <kbd>A</kbd> back-foot shot</p>
      <p><kbd>D</kbd> defend · hold <kbd>Shift</kbd> to loft</p><p><kbd>Z</kbd> <kbd>C</kbd> shuffle across before the ball</p>
      <p>Press as the ball arrives. Full balls suit the front foot, short balls the back foot.</p></div>
    <div><h3>Running</h3><p><kbd>Space</kbd> run · <kbd>X</kbd> send your partner back</p><p>The hint shows how many look safe.</p></div>
    <div><h3>Anywhere</h3><p><kbd>Tab</kbd> scorecard · <kbd>V</kbd> camera · <kbd>M</kbd> sound · <kbd>Esc</kbd> menu</p></div>
  </div>`;
}
