// App shell: menus, room setup, transport choice and the render loop.

import * as THREE from 'three';
import { MARKUP } from './markup.js';
import { Scene3D } from './scene.js';
import { Hud, esc } from './hud.js';
import { Input } from './input.js';
import { Sound } from './audio.js';
import { GameClient, controlsHtml } from './game.js';
import { LocalLink, WsLink, PresenceHost, PresenceGuest } from './links.js';
import { Rig, Actor, makeKit, setKit } from './players.js';
import { RoomCore } from '../../shared/room.js';
import { TEAMS, TEAM_IDS, hex, getTeam } from '../../shared/teams.js';
import { OVERS_OPTIONS, HALF } from '../../shared/constants.js';

if (!document.getElementById('pc-root')) document.body.insertAdjacentHTML('afterbegin', MARKUP);

const $ = (id) => document.getElementById(id);

// ---- preferences (per viewer, optional) ----------------------------------------------------
const PREF_KEY = 'pavilion-cricket';
const prefs = { name: '', team: 'IND', opp: 'AUS', overs: 5, level: 'medium', muted: false, server: '' };
try {
  Object.assign(prefs, JSON.parse(localStorage.getItem(PREF_KEY) || '{}'));
} catch {
  /* storage unavailable */
}
const savePrefs = () => {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify(prefs));
  } catch {
    /* storage unavailable */
  }
};

// ---- core objects --------------------------------------------------------------------------
let scene;
try {
  scene = new Scene3D($('pc-canvas'));
} catch (e) {
  document.body.insertAdjacentHTML('beforeend', '<div class="nogl"><h2>3D graphics are unavailable</h2><p>Pavilion Cricket needs WebGL. Turn on hardware acceleration in your browser settings, or try another browser.</p></div>');
  throw e;
}
const hud = new Hud();
const input = new Input();
const sound = new Sound();
sound.setMuted(!!prefs.muted);
input.bindTouch($('touch'), $('stick'), $('knob'), $('tbtns'));
const unlock = () => sound.unlock();
window.addEventListener('pointerdown', unlock);
window.addEventListener('keydown', unlock);

const state = { game: null, link: null, closer: null, vsBot: false, lobby: null, screen: 'scr-title', online: 'checking', roomApi: null };

function show(id) {
  for (const s of document.querySelectorAll('.screen')) s.hidden = s.id !== id;
  state.screen = id;
  menuCast.visible = !!id;
}

function toggleMute() {
  sound.setMuted(!sound.muted);
  prefs.muted = sound.muted;
  savePrefs();
  hud.flash(sound.muted ? 'Sound off' : 'Sound on', 'small', 900);
}

// ---- decorative players on the menu screens ---------------------------------------------
const menuCast = new THREE.Group();
const castKitA = makeKit();
const castKitB = makeKit();
setKit(castKitA, getTeam(prefs.team));
setKit(castKitB, getTeam(prefs.opp));
const cast = [
  { a: new Actor(new Rig({ kit: castKitA, bat: true, pads: true, helmet: true, gloves: true, skin: 2 }), true), x: -0.35, z: 9.5, h: Math.PI / 2, anim: 'stance' },
  { a: new Actor(new Rig({ kit: castKitB, gloves: true, pads: true, skin: 4 })), x: 0.3, z: 19, h: Math.PI, anim: 'crouch' },
  { a: new Actor(new Rig({ kit: castKitB, skin: 1 })), x: 1.9, z: 19.9, h: Math.PI, anim: 'ready' },
  { a: new Actor(new Rig({ kit: castKitB, skin: 5 })), x: 0.9, z: -16, h: 0, anim: 'stand' },
  { a: new Actor(new Rig({ kit: castKitA, bat: true, pads: true, helmet: true, gloves: true, skin: 3 }), true), x: -1.3, z: -9.7, h: 0, anim: 'batIdle' },
  { a: new Actor(new Rig({ kit: castKitA, umpire: true, skin: 3 })), x: -1.05, z: -12.4, h: 0, anim: 'umpire' },
];
for (const c of cast) {
  c.a.place(c.x, c.z, c.h);
  c.a.set(c.anim);
  menuCast.add(c.a.rig.root);
}
scene.scene.add(menuCast);

// ---- title ---------------------------------------------------------------------------------
$('in-name').value = prefs.name;
$('in-name').addEventListener('input', (e) => {
  prefs.name = e.target.value.slice(0, 16);
  savePrefs();
});
const playerName = () => (prefs.name || '').trim() || 'Captain';
$('go-solo').addEventListener('click', () => {
  sound.click();
  renderSetup();
  show('scr-setup');
});
$('go-online').addEventListener('click', () => {
  sound.click();
  renderOnline();
  show('scr-online');
});
$('go-howto').addEventListener('click', () => {
  sound.click();
  $('howto-body').innerHTML = controlsHtml();
  show('scr-howto');
});
for (const b of document.querySelectorAll('[data-back]')) {
  b.addEventListener('click', () => {
    sound.click();
    show('scr-title');
  });
}

// ---- pickers -------------------------------------------------------------------------------
function teamGrid(el, selected, disabled, onPick) {
  el.innerHTML = TEAM_IDS.map((id) => {
    const t = TEAMS[id];
    return `<button type="button" class="team${id === selected ? ' on' : ''}" data-id="${id}"${id === disabled ? ' disabled' : ''}>
      <span class="sw" style="--a:${hex(t.shirt)};--b:${hex(t.trim)}"></span><b>${id}</b><small>${esc(t.name)}</small></button>`;
  }).join('');
  el.querySelectorAll('.team').forEach((b) => b.addEventListener('click', () => {
    sound.click();
    onPick(b.dataset.id);
  }));
}

function segment(el, options, selected, onPick, disabled = false) {
  el.innerHTML = options.map(([v, label]) => `<button type="button" class="${String(v) === String(selected) ? 'on' : ''}" data-v="${v}"${disabled ? ' disabled' : ''}>${esc(label)}</button>`).join('');
  el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    sound.click();
    onPick(b.dataset.v);
  }));
}

function renderSetup() {
  if (prefs.team === prefs.opp) prefs.opp = TEAM_IDS.find((t) => t !== prefs.team);
  teamGrid($('pick-team'), prefs.team, prefs.opp, (id) => {
    prefs.team = id;
    savePrefs();
    setKit(castKitA, getTeam(id));
    renderSetup();
  });
  teamGrid($('pick-opp'), prefs.opp, prefs.team, (id) => {
    prefs.opp = id;
    savePrefs();
    setKit(castKitB, getTeam(id));
    renderSetup();
  });
  segment($('pick-overs'), OVERS_OPTIONS.map((o) => [o, String(o)]), prefs.overs, (v) => {
    prefs.overs = +v;
    savePrefs();
    renderSetup();
  });
  segment($('pick-level'), [['easy', 'Easy'], ['medium', 'Medium'], ['hard', 'Hard']], prefs.level, (v) => {
    prefs.level = v;
    savePrefs();
    renderSetup();
  });
}

// ---- match against the computer ------------------------------------------------------------
$('solo-start').addEventListener('click', () => {
  sound.click();
  closeLink();
  const room = new RoomCore({ code: '', vsBot: true });
  const link = new LocalLink(room, 0, playerName());
  attach(link, true, () => {
    link.close();
    room.dispose();
  });
  link.send({ t: 'cfg', team: prefs.team, opp: prefs.opp, overs: prefs.overs, level: prefs.level });
  link.send({ t: 'start' });
});

// ---- online --------------------------------------------------------------------------------
const inArtifact = !!(window.claude && typeof window.claude.use === 'function');
const sameHostWs = !inArtifact && location.protocol.startsWith('http') ? `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws` : null;

if (inArtifact) {
  window.claude.use('room').then((api) => {
    state.roomApi = api;
    state.online = api ? 'room' : 'none';
    if (state.screen === 'scr-online') renderOnline();
  }).catch(() => {
    state.online = 'none';
  });
} else {
  state.online = sameHostWs ? 'server' : 'manual';
}

function renderOnline() {
  const mode = $('online-mode');
  const actions = $('online-actions');
  $('online-error').hidden = true;
  $('online-server').hidden = state.online !== 'manual';
  $('in-server').value = prefs.server || '';
  actions.hidden = state.online === 'checking' || state.online === 'none';
  if (state.online === 'checking') mode.textContent = 'Checking how this page can connect…';
  else if (state.online === 'room') mode.textContent = 'Your friend opens this same page, signed in, and joins with your code. Share the page with them first if they cannot open it.';
  else if (state.online === 'server') mode.textContent = 'Create a room and send the code to your friend. They open this same address and join.';
  else if (state.online === 'manual') mode.textContent = 'This copy of the game was opened from a file, so it needs the address of a Pavilion Cricket server to play online.';
  else mode.textContent = 'Online play is not available on this page. Run the game server (npm start) or open the published game, then try again.';
}

function onlineError(text) {
  const e = $('online-error');
  e.textContent = text;
  e.hidden = false;
  show('scr-online');
}

function serverUrl() {
  if (state.online === 'server') return sameHostWs;
  let v = $('in-server').value.trim();
  prefs.server = v;
  savePrefs();
  if (!v) return null;
  if (!/^wss?:\/\//.test(v)) v = (v.startsWith('localhost') || v.startsWith('127.') ? 'ws://' : 'wss://') + v.replace(/^https?:\/\//, '');
  if (!/\/ws$/.test(v)) v = v.replace(/\/$/, '') + '/ws';
  return v;
}

async function createRoom() {
  sound.click();
  closeLink();
  if (state.online === 'room') {
    const host = new PresenceHost(state.roomApi, playerName());
    attach(host.local, false, () => host.close());
    try {
      await host.open();
    } catch {
      closeLink();
      onlineError('Could not open a room on this page. Reload and try again.');
    }
    return;
  }
  const url = serverUrl();
  if (!url) return onlineError('Enter the address of a Pavilion Cricket server first.');
  const link = new WsLink(url);
  attach(link, false, () => link.close());
  link.open({ t: 'create', name: playerName() });
}

async function joinRoom() {
  sound.click();
  const code = $('in-code').value.trim().toUpperCase();
  if (!/^[A-Z]{5}$/.test(code)) return onlineError('Room codes are five letters, like KQWTR.');
  closeLink();
  if (state.online === 'room') {
    const guest = new PresenceGuest(state.roomApi, code, playerName());
    attach(guest, false, () => guest.close());
    $('online-mode').textContent = `Looking for room ${code}…`;
    await guest.open();
    return;
  }
  const url = serverUrl();
  if (!url) return onlineError('Enter the address of a Pavilion Cricket server first.');
  const link = new WsLink(url);
  attach(link, false, () => link.close());
  link.open({ t: 'join', code, name: playerName() });
}

$('on-create').addEventListener('click', createRoom);
$('on-join').addEventListener('click', joinRoom);
$('in-code').addEventListener('input', (e) => {
  e.target.value = e.target.value.toUpperCase().replace(/[^A-Z]/g, '');
});
$('in-code').addEventListener('keydown', (e) => e.key === 'Enter' && joinRoom());

// ---- lobby ---------------------------------------------------------------------------------
function renderLobby(m) {
  const me = m.you;
  const other = 1 - me;
  $('lobby-code').textContent = m.code || '-----';
  $('lobby-note').textContent = me === 0
    ? (m.players[1] ? 'Both players are here. Pick your teams and start when ready.' : 'Send this code to your friend. The match starts when you press Start.')
    : 'You are in. The host chooses the number of overs and starts the match.';
  $('lobby-seats').innerHTML = m.players.map((p, i) => {
    const team = getTeam(m.cfg.teams[i]);
    return `<div class="seat${p ? '' : ' empty'}"><span class="sw" style="--a:${hex(team.shirt)};--b:${hex(team.trim)}"></span>
      <div><b>${p ? esc(p.name) : 'Waiting for a friend…'}${i === me ? ' <small>(you)</small>' : ''}</b><small>${i === 0 ? 'Host' : 'Guest'} · ${esc(team.name)}${p && !p.on ? ' · reconnecting' : ''}</small></div></div>`;
  }).join('');
  teamGrid($('lobby-team'), m.cfg.teams[me], m.cfg.teams[other], (id) => state.link.send({ t: 'cfg', team: id }));
  segment($('lobby-overs'), OVERS_OPTIONS.map((o) => [o, String(o)]), m.cfg.overs, (v) => state.link.send({ t: 'cfg', overs: +v }), me !== 0);
  const ready = m.players[1] && m.players[1].on;
  $('lobby-start').hidden = me !== 0;
  $('lobby-start').disabled = !ready;
  $('lobby-start').textContent = ready ? 'Start match' : 'Waiting for your friend';
  $('lobby-wait').hidden = me === 0;
  show('scr-lobby');
}
$('lobby-start').addEventListener('click', () => {
  sound.click();
  if (state.link) state.link.send({ t: 'start' });
});
$('lobby-leave').addEventListener('click', () => {
  sound.click();
  closeLink();
  show('scr-online');
  renderOnline();
});
$('lobby-copy').addEventListener('click', () => {
  const code = $('lobby-code').textContent;
  const done = () => {
    $('lobby-copy').textContent = 'Copied';
    setTimeout(() => {
      $('lobby-copy').textContent = 'Copy code';
    }, 1500);
  };
  try {
    navigator.clipboard.writeText(code).then(done, () => selectCode());
  } catch {
    selectCode();
  }
});
function selectCode() {
  const r = document.createRange();
  r.selectNodeContents($('lobby-code'));
  const s = window.getSelection();
  s.removeAllRanges();
  s.addRange(r);
}

// ---- link plumbing -------------------------------------------------------------------------
function attach(link, vsBot, closer) {
  state.link = link;
  state.closer = closer;
  state.vsBot = vsBot;
  link.onMessage = (m) => onNet(link, m);
  link.onStatus = (s) => onStatus(link, s);
}

function closeLink() {
  if (state.game) {
    state.game.destroy();
    state.game = null;
  }
  const c = state.closer;
  state.link = null;
  state.closer = null;
  if (c) c();
}

function onNet(link, m) {
  if (link !== state.link) return;
  switch (m.t) {
    case 'lobby':
      state.lobby = m;
      if (m.state === 'lobby' && state.game) {
        // Rematch: back to the lobby (or straight into a new match against the computer).
        state.game.destroy();
        state.game = null;
        if (state.vsBot) {
          link.send({ t: 'start' });
          return;
        }
      }
      if (!state.vsBot && !state.game) renderLobby(m);
      return;
    case 'joined':
    case 'pong':
      return;
    case 'error':
      closeLink();
      onlineError(m.m);
      return;
    case 'begin':
      if (state.game) state.game.destroy();
      show(null);
      state.game = new GameClient({
        scene, hud, input, sound, link, vsBot: state.vsBot,
        onExit: () => {
          closeLink();
          show('scr-title');
        },
        onMute: toggleMute,
      });
      break;
    default:
      break;
  }
  if (state.game) state.game.onMessage(m);
}

function onStatus(link, s) {
  if (link !== state.link) return;
  const inGame = !!state.game;
  if (s === 'open') {
    if (inGame) hud.banner('');
    return;
  }
  if (s === 'reconnecting') {
    if (inGame) hud.banner('Connection dropped. Reconnecting…');
    return;
  }
  const msg = {
    failed: 'Could not reach the game server. Check the address and that the server is running.',
    lost: inGame ? 'The connection to the match was lost.' : 'The connection was lost.',
    notfound: 'No room with that code is open right now. Check the code with your friend.',
    full: 'That room already has two players.',
  }[s] || 'The connection was lost.';
  if (inGame && (s === 'lost')) {
    hud.banner(state.online === 'room' ? 'The host left the match.' : msg);
    return;
  }
  closeLink();
  onlineError(msg);
}

// ---- HUD buttons ---------------------------------------------------------------------------
$('btn-card').addEventListener('click', () => input.virtualPress('Tab'));
$('btn-cam').addEventListener('click', () => input.virtualPress('KeyV'));
$('btn-menu').addEventListener('click', () => input.virtualPress('Escape'));
$('ov').addEventListener('click', (e) => e.stopPropagation());

// ---- loop ----------------------------------------------------------------------------------
let last = performance.now();
let orbit = 0.6;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  scene.resize();
  if (state.game) {
    state.game.update(dt);
  } else {
    orbit += dt * 0.05;
    const r = 34;
    scene.camera.position.set(Math.sin(orbit) * r + 4, 9 + Math.sin(orbit * 0.7) * 2, Math.cos(orbit) * r);
    scene.camera.lookAt(0, 0.8, HALF * 0.25);
    for (const c of cast) c.a.update(dt);
    if (input.hit('Escape') && state.screen !== 'scr-title' && state.screen !== 'scr-lobby') show('scr-title');
  }
  scene.render();
  input.endFrame();
  requestAnimationFrame(frame);
}
show('scr-title');
requestAnimationFrame(frame);

// Debug hook for automated tests.
window.__pavilion = { state, scene, hud, input };
