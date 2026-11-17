// RoomCore: two seats, a lobby, and a MatchHost once the match starts. The same class runs
// on the game server, in the page for a match against the computer, and in the creator's
// page for a peer-to-peer match.

import { MatchHost } from './host.js';
import { TEAMS } from './teams.js';
import { OVERS_OPTIONS } from './constants.js';
import { LEVELS } from './batting.js';
import { randomSeed } from './rng.js';

export function cleanName(name, fallback = 'Player') {
  const s = String(name == null ? '' : name).replace(/[^\p{L}\p{N} _.'-]/gu, '').trim().slice(0, 16);
  return s || fallback;
}

export function makeRoomCode() {
  const letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
  let s = '';
  for (let i = 0; i < 5; i++) s += letters[Math.floor(Math.random() * letters.length)];
  return s;
}

export class RoomCore {
  constructor({ code = '', vsBot = false, cfg = {}, timers, seedFn = randomSeed } = {}) {
    this.code = code;
    this.vsBot = vsBot;
    this.cfg = { overs: 5, teams: ['IND', 'AUS'], level: 'medium', ...cfg };
    this.seats = [null, null];
    this.host = null;
    this.state = 'lobby';
    this.timers = timers;
    this.seedFn = seedFn;
    this.touched = Date.now();
  }

  freeSeat() {
    if (!this.seats[0]) return 0;
    if (!this.seats[1] && !this.vsBot) return 1;
    return -1;
  }

  connectedCount() {
    return this.seats.filter((s) => s && s.on).length;
  }

  attach(seat, name, send) {
    this.touched = Date.now();
    const s = this.seats[seat];
    if (s) {
      s.send = send;
      s.on = true;
      if (name) s.name = cleanName(name);
      this.pushLobby();
      if (this.host && this.state === 'playing') this.host.reconnect(seat);
      return;
    }
    this.seats[seat] = { kind: 'human', name: cleanName(name, 'Player ' + (seat + 1)), send, on: true };
    this.pushLobby();
  }

  detach(seat) {
    const s = this.seats[seat];
    if (!s) return;
    s.on = false;
    s.send = null;
    if (this.host && this.state === 'playing') this.host.disconnect(seat);
    else this.seats[seat] = null;
    this.pushLobby();
  }

  pushLobby() {
    const players = this.seats.map((s) => (s ? { name: s.name, on: s.on } : null));
    for (let i = 0; i < 2; i++) {
      const s = this.seats[i];
      if (s && s.on && s.send) s.send({ t: 'lobby', code: this.code, you: i, players, cfg: this.cfg, vsBot: this.vsBot, state: this.state });
    }
  }

  begin() {
    const seats = [0, 1].map((i) => this.seats[i] || { kind: 'bot', name: 'Computer', on: true });
    if (this.host) this.host.dispose();
    this.host = new MatchHost({ seats, cfg: { ...this.cfg, teams: [...this.cfg.teams] }, seed: this.seedFn(), timers: this.timers, timeouts: !this.vsBot });
    this.state = 'playing';
    this.host.start();
  }

  onMessage(seat, msg) {
    if (!msg || typeof msg.t !== 'string' || !this.seats[seat]) return;
    this.touched = Date.now();
    switch (msg.t) {
      case 'cfg': {
        if (this.state !== 'lobby') break;
        const c = this.cfg;
        if (seat === 0 && OVERS_OPTIONS.includes(msg.overs)) c.overs = msg.overs;
        if (seat === 0 && msg.level && LEVELS[msg.level]) c.level = msg.level;
        if (msg.team && TEAMS[msg.team] && c.teams[1 - seat] !== msg.team) c.teams[seat] = msg.team;
        if (this.vsBot && seat === 0 && msg.opp && TEAMS[msg.opp] && c.teams[0] !== msg.opp) c.teams[1] = msg.opp;
        this.pushLobby();
        break;
      }
      case 'start':
        if (this.state === 'lobby' && seat === 0 && (this.vsBot || (this.seats[1] && this.seats[1].on))) this.begin();
        break;
      case 'rematch':
        if (this.state === 'playing' && this.host && this.host.match.phase === 'done') {
          this.host.dispose();
          this.host = null;
          this.state = 'lobby';
          for (let i = 0; i < 2; i++) if (this.seats[i] && !this.seats[i].on) this.seats[i] = null;
          this.pushLobby();
        }
        break;
      case 'leave':
        if (this.host && this.state === 'playing') this.host.forfeit(seat);
        break;
      case 'ping': {
        const s = this.seats[seat];
        if (s.send) s.send({ t: 'pong', id: msg.id });
        break;
      }
      default:
        if (this.host && this.state === 'playing') this.host.onMessage(seat, msg);
    }
  }

  dispose() {
    if (this.host) this.host.dispose();
    this.host = null;
  }
}
