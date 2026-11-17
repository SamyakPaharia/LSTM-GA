// DOM overlay: scoreboard, over ticker, meter, radar, flashes, overlays and scorecard.

import { getTeam, hex } from '../../shared/teams.js';
import { HOW, oversText } from '../../shared/match.js';
import { styleName } from '../../shared/delivery.js';
import { BOUNDARY_RX, BOUNDARY_RZ, HALF } from '../../shared/constants.js';

export const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const $ = (id) => document.getElementById(id);

export function teamsOf(snap) {
  return [getTeam(snap.cfg.teams[0]), getTeam(snap.cfg.teams[1])];
}

function dismissal(row, bowlTeam) {
  const how = row[4];
  const b = row[5] >= 0 ? bowlTeam.players[row[5]].short : '';
  const f = row[6] >= 0 ? bowlTeam.players[row[6]].short : '';
  if (how === HOW.DNB) return '';
  if (how === HOW.IN) return 'not out';
  if (how === HOW.BOWLED) return `b ${b}`;
  if (how === HOW.LBW) return `lbw b ${b}`;
  if (how === HOW.CAUGHT) return row[6] === row[5] ? `c & b ${b}` : `c ${f} b ${b}`;
  if (how === HOW.RUNOUT) return f ? `run out (${f})` : 'run out';
  return '';
}

const sr = (r, b) => (b ? ((r / b) * 100).toFixed(0) : '–');
const econ = (r, balls) => (balls ? ((r / balls) * 6).toFixed(1) : '–');

export class Hud {
  constructor() {
    this.el = {
      hud: $('pc-hud'), sb: $('sb'), ticker: $('ticker'), speed: $('speed'), flash: $('flash'), timing: $('timing'),
      hint: $('hint'), meter: $('meter'), meterFill: $('meter-fill'), meterSweet: $('meter-sweet'), meterMark: $('meter-mark'),
      comm: $('comm'), banner: $('banner'), freeHit: $('freehit'), radar: $('radar'), ov: $('ov'), ovPanel: $('ov-panel'),
      card: $('card'), status: $('status'),
    };
    this.radarCtx = this.el.radar.getContext('2d');
    this.flashTimer = null;
    this.timingTimer = null;
    this.nav = null;
    this.cardOpen = false;
    this.cardInn = -1;
  }

  show(on) {
    this.el.hud.hidden = !on;
    if (!on) {
      this.closeOverlay();
      this.showCard(false);
    }
  }

  score(snap) {
    if (!snap || snap.cur < 0) {
      this.el.sb.innerHTML = '';
      return;
    }
    const teams = teamsOf(snap);
    const I = snap.inn[snap.cur];
    const bat = teams[I.bat];
    const bowl = teams[1 - I.bat];
    const row = (idx, on) => {
      if (idx < 0) return '';
      const r = I.bt[idx];
      return `<div class="sb-bat${on ? ' on' : ''}"><span class="nm">${on ? '<i class="strike"></i>' : '<i></i>'}${esc(bat.players[idx].short)}</span><span class="fig">${r[0]}<small>${r[1]}</small></span></div>`;
    };
    let bowler = '';
    if (I.bowler >= 0) {
      const b = I.bw[I.bowler];
      bowler = `<div class="sb-bowl"><span class="nm">${esc(bowl.players[I.bowler].short)}</span><span class="fig">${oversText(b[0])} · ${b[2]}-${b[1]}</span></div>`;
    }
    let need;
    if (I.target != null) {
      const runs = I.target - I.runs;
      const balls = snap.cfg.overs * 6 - I.balls;
      need = runs > 0 ? `Need <b>${runs}</b> from <b>${balls}</b> · RRR ${balls ? ((runs / balls) * 6).toFixed(2) : '–'}` : 'Target reached';
    } else {
      need = `CRR ${I.balls ? ((I.runs / I.balls) * 6).toFixed(2) : '0.00'} · ${snap.cfg.overs} overs`;
    }
    this.el.sb.innerHTML = `
      <div class="sb-top" style="--team:${hex(bat.shirt)};--trim:${hex(bat.trim)}">
        <span class="sb-team">${esc(bat.id)}</span>
        <span class="sb-score">${I.runs}<em>/</em>${I.wkts}</span>
        <span class="sb-overs">${oversText(I.balls)}</span>
        <span class="sb-v">v ${esc(bowl.id)}</span>
      </div>
      ${row(I.st, true)}${row(I.ns, false)}${bowler}
      <div class="sb-need">${need}</div>`;
    const over = I.thisOver.length ? I.thisOver : [];
    this.el.ticker.innerHTML = '<span class="lbl">This over</span>' + over.map((x) => {
      const cls = x.includes('W') ? 'w' : x === '4' || x === '6' ? 'b' : x.startsWith('Wd') || x.startsWith('Nb') ? 'x' : x === '0' ? 'z' : '';
      return `<span class="ball ${cls}">${esc(x === '0' ? '•' : x)}</span>`;
    }).join('');
    this.freeHit(I.freeHit && snap.phase !== 'done');
  }

  freeHit(on) {
    this.el.freeHit.hidden = !on;
  }

  speed(text) {
    this.el.speed.textContent = text || '';
    this.el.speed.hidden = !text;
  }

  flash(text, cls = '', ms = 1600) {
    const f = this.el.flash;
    f.textContent = text;
    f.className = 'flash show ' + cls;
    clearTimeout(this.flashTimer);
    this.flashTimer = setTimeout(() => {
      f.className = 'flash ' + cls;
    }, ms);
  }

  timing(text, cls = '') {
    const t = this.el.timing;
    t.innerHTML = text;
    t.className = 'timing show ' + cls;
    clearTimeout(this.timingTimer);
    this.timingTimer = setTimeout(() => {
      t.className = 'timing';
    }, 1500);
  }

  comm(text) {
    this.el.comm.textContent = text || '';
    this.el.comm.hidden = !text;
  }

  hint(items) {
    // items: [[keys, label], ...]
    this.el.hint.innerHTML = (items || []).map(([k, l]) => `<span class="h"><kbd>${esc(k)}</kbd>${esc(l)}</span>`).join('');
  }

  status(text) {
    this.el.status.textContent = text || '';
    this.el.status.hidden = !text;
  }

  banner(text) {
    this.el.banner.textContent = text || '';
    this.el.banner.hidden = !text;
  }

  meter(show, value = 0, lo = 0.8, hi = 0.95, locked = null) {
    const m = this.el.meter;
    m.hidden = !show;
    if (!show) return;
    this.el.meterFill.style.transform = `scaleY(${Math.min(1, value)})`;
    this.el.meterSweet.style.bottom = lo * 100 + '%';
    this.el.meterSweet.style.height = (hi - lo) * 100 + '%';
    m.classList.toggle('over', value > 0.97);
    this.el.meterMark.hidden = locked == null;
    if (locked != null) this.el.meterMark.style.bottom = locked * 100 + '%';
  }

  radar(field, fieldColor, batters, batColor, ball) {
    const g = this.radarCtx;
    const W = this.el.radar.width;
    const H = this.el.radar.height;
    const s = (W / 2 - 6) / BOUNDARY_RZ;
    const X = (x) => W / 2 + x * s;
    const Y = (z) => H / 2 + z * s;
    g.clearRect(0, 0, W, H);
    g.fillStyle = 'rgba(32, 92, 52, 0.85)';
    g.beginPath();
    g.ellipse(W / 2, H / 2, BOUNDARY_RX * s, BOUNDARY_RZ * s, 0, 0, Math.PI * 2);
    g.fill();
    g.strokeStyle = 'rgba(255,255,255,0.55)';
    g.lineWidth = 1;
    g.stroke();
    g.setLineDash([2, 3]);
    g.beginPath();
    g.ellipse(W / 2, H / 2, 27.4 * s, (27.4 + HALF) * s, 0, 0, Math.PI * 2);
    g.stroke();
    g.setLineDash([]);
    g.fillStyle = '#d8c68f';
    g.fillRect(X(-1.5), Y(-HALF - 1), 3 * s, (2 * HALF + 2) * s);
    const dot = (x, z, color, r) => {
      g.fillStyle = color;
      g.beginPath();
      g.arc(X(x), Y(z), r, 0, Math.PI * 2);
      g.fill();
    };
    for (const f of field) dot(f.x, f.z, fieldColor, 3.2);
    for (const b of batters) dot(b.x, b.z, batColor, 3.2);
    if (ball) {
      g.strokeStyle = '#000';
      dot(ball[0], ball[2], '#ffffff', 2.4);
    }
  }

  // Generic modal: {title, sub, body (HTML string), buttons: [{label, primary, onClick}]}
  overlay({ title = '', sub = '', body = '', buttons = [], cls = '' }) {
    const p = this.el.ovPanel;
    p.className = 'panel ' + cls;
    p.innerHTML = `${title ? `<h2>${title}</h2>` : ''}${sub ? `<p class="sub">${sub}</p>` : ''}${body}<div class="btns"></div>`;
    const wrap = p.querySelector('.btns');
    const btns = buttons.map((b, i) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'btn' + (b.primary ? ' primary' : '');
      el.innerHTML = b.label;
      el.addEventListener('click', () => b.onClick());
      el.addEventListener('mouseenter', () => this.select(i));
      wrap.appendChild(el);
      return el;
    });
    this.el.ov.hidden = false;
    this.nav = { btns, sel: 0, list: null };
    if (btns.length) this.select(Math.max(0, buttons.findIndex((b) => b.primary)));
  }

  // A list overlay where arrows move the selection: {title, sub, items:[{html, disabled}], onPick(i), buttons}
  listOverlay({ title, sub, items, onPick, buttons = [] }) {
    this.overlay({ title, sub, body: '<div class="pick-list"></div>', buttons });
    const list = this.el.ovPanel.querySelector('.pick-list');
    const rows = items.map((it, i) => {
      const r = document.createElement('button');
      r.type = 'button';
      r.className = 'pick' + (it.disabled ? ' off' : '');
      r.disabled = !!it.disabled;
      r.innerHTML = it.html;
      r.addEventListener('click', () => !it.disabled && onPick(i));
      r.addEventListener('mouseenter', () => !it.disabled && this.selectRow(i));
      list.appendChild(r);
      return r;
    });
    this.nav.list = { rows, sel: items.findIndex((x) => !x.disabled), onPick, items };
    this.selectRow(this.nav.list.sel);
  }

  select(i) {
    if (!this.nav) return;
    this.nav.btns.forEach((b, j) => b.classList.toggle('sel', j === i));
    this.nav.sel = i;
  }

  selectRow(i) {
    const l = this.nav && this.nav.list;
    if (!l || i < 0) return;
    l.rows.forEach((r, j) => r.classList.toggle('sel', j === i));
    l.sel = i;
    l.rows[i].scrollIntoView({ block: 'nearest' });
  }

  get overlayOpen() {
    return !this.el.ov.hidden;
  }

  closeOverlay() {
    this.el.ov.hidden = true;
    this.el.ovPanel.innerHTML = '';
    this.nav = null;
  }

  // Keyboard navigation for the open overlay.
  navigate(input) {
    const n = this.nav;
    if (!n) return;
    const l = n.list;
    if (l) {
      const step = input.hit('ArrowDown') ? 1 : input.hit('ArrowUp') ? -1 : 0;
      if (step) {
        let i = l.sel;
        for (let k = 0; k < l.items.length; k++) {
          i = (i + step + l.items.length) % l.items.length;
          if (!l.items[i].disabled) break;
        }
        this.selectRow(i);
      }
      if ((input.hit('Space') || input.hit('Enter') || input.hit('NumpadEnter')) && l.sel >= 0) {
        l.onPick(l.sel);
        return;
      }
    }
    if (n.btns.length) {
      const step = input.hit('ArrowRight') || (!l && input.hit('ArrowDown')) ? 1 : input.hit('ArrowLeft') || (!l && input.hit('ArrowUp')) ? -1 : 0;
      if (step) this.select((n.sel + step + n.btns.length) % n.btns.length);
      if (!l && (input.hit('Space') || input.hit('Enter') || input.hit('NumpadEnter'))) n.btns[n.sel].click();
    }
  }

  showCard(on, snap) {
    this.cardOpen = on;
    this.el.card.hidden = !on;
    if (on && snap) this.renderCard(snap);
  }

  renderCard(snap, which) {
    if (!snap || snap.cur < 0) return;
    const teams = teamsOf(snap);
    const idx = which == null ? (this.cardInn >= 0 && this.cardInn < snap.inn.length ? this.cardInn : snap.cur) : which;
    this.cardInn = idx;
    const I = snap.inn[idx];
    const bat = teams[I.bat];
    const bowl = teams[1 - I.bat];
    const tabs = snap.inn.map((inn, i) => `<button type="button" class="tab${i === idx ? ' on' : ''}" data-i="${i}">${esc(teams[inn.bat].name)} ${inn.runs}/${inn.wkts}</button>`).join('');
    const batRows = I.bt.map((r, i) => {
      if (r[4] === HOW.DNB) return '';
      return `<tr${r[4] === HOW.IN ? ' class="in"' : ''}><td>${esc(bat.players[i].name)}</td><td class="how">${esc(dismissal(r, bowl))}</td><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${sr(r[0], r[1])}</td></tr>`;
    }).join('');
    const dnb = I.bt.map((r, i) => (r[4] === HOW.DNB ? bat.players[i].short : null)).filter(Boolean);
    const bowlRows = I.bw.map((r, i) => {
      if (!r[0] && !r[3] && !r[4]) return '';
      return `<tr><td>${esc(bowl.players[i].name)}</td><td class="how">${esc(styleName(bowl.players[i].style))}</td><td>${oversText(r[0])}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3] + r[4]}</td><td>${econ(r[1], r[0])}</td></tr>`;
    }).join('');
    const fow = I.fow.map((f) => `${f[0]}-${f[1]} (${esc(bat.players[f[2]].short)}, ${oversText(f[3])})`).join(' · ');
    this.el.card.innerHTML = `
      <div class="card-in">
        <div class="card-head"><h2>Scorecard</h2><div class="tabs">${tabs}</div><button type="button" class="x" aria-label="Close scorecard">×</button></div>
        ${snap.result ? `<p class="card-result">${esc(snap.result.text)}</p>` : ''}
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>${esc(bat.name)}</th><th></th><th>R</th><th>B</th><th>4s</th><th>6s</th><th>SR</th></tr></thead><tbody>${batRows}</tbody></table></div>
        <p class="card-line">Extras ${I.ex.wd + I.ex.nb} (wd ${I.ex.wd}, nb ${I.ex.nb}) · <b>Total ${I.runs}/${I.wkts}</b> (${oversText(I.balls)} ov)${dnb.length ? ` · Did not bat: ${esc(dnb.join(', '))}` : ''}</p>
        ${fow ? `<p class="card-line">Fall of wickets: ${fow}</p>` : ''}
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>${esc(bowl.name)}</th><th></th><th>O</th><th>R</th><th>W</th><th>Ex</th><th>Econ</th></tr></thead><tbody>${bowlRows}</tbody></table></div>
      </div>`;
    this.el.card.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => this.renderCard(snap, +b.dataset.i)));
    this.el.card.querySelector('.x').addEventListener('click', () => this.showCard(false));
  }
}
