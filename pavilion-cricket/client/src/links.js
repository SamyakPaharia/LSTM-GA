// Transports. Each link has send(msg), close(), and the callbacks onMessage and onStatus.

import { RoomCore, makeRoomCode, cleanName } from '../../shared/room.js';

const clone = (m) => JSON.parse(JSON.stringify(m));

// The host runs in this page; messages cross by microtask.
export class LocalLink {
  constructor(room, seat, name) {
    this.room = room;
    this.seat = seat;
    this.onMessage = null;
    this.onStatus = null;
    this.closed = false;
    room.attach(seat, name, (msg) => {
      const copy = clone(msg);
      queueMicrotask(() => {
        if (!this.closed && this.onMessage) this.onMessage(copy);
      });
    });
  }

  send(msg) {
    const copy = clone(msg);
    queueMicrotask(() => {
      if (!this.closed) this.room.onMessage(this.seat, copy);
    });
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.room.detach(this.seat);
  }
}

// JSON over WebSocket to the game server, with a seat token for reconnection.
export class WsLink {
  constructor(url) {
    this.url = url;
    this.onMessage = null;
    this.onStatus = null;
    this.code = null;
    this.token = null;
    this.seat = -1;
    this.closed = false;
    this.retry = 0;
    this.ws = null;
    this.first = null;
  }

  open(first) {
    this.first = first;
    this.connect();
  }

  status(s) {
    if (this.onStatus) this.onStatus(s);
  }

  connect() {
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch {
      this.status('failed');
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      ws.send(JSON.stringify(this.token ? { t: 'rejoin', code: this.code, token: this.token } : this.first));
    };
    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.t === 'joined') {
        this.code = msg.code;
        this.token = msg.token;
        this.seat = msg.seat;
        this.status('open');
      }
      if (this.onMessage) this.onMessage(msg);
    };
    ws.onclose = () => {
      if (this.closed || this.ws !== ws) return;
      if (!this.token) {
        this.status('failed');
        return;
      }
      if (this.retry++ < 8) {
        this.status('reconnecting');
        setTimeout(() => !this.closed && this.connect(), 800 * Math.min(5, this.retry));
      } else {
        this.status('lost');
      }
    };
    ws.onerror = () => {};
  }

  send(msg) {
    if (this.ws && this.ws.readyState === 1 && this.seat >= 0) this.ws.send(JSON.stringify(msg));
  }

  close() {
    this.closed = true;
    if (this.ws) this.ws.close();
  }
}

// Ordered, reliable messages on top of the artifact room's presence state. Each side
// publishes its unacknowledged chunks and the highest sequence number it has received.
const CHUNK = 700;
const BUDGET = 2900;

class PresenceStream {
  constructor(named, role, name) {
    this.named = named;
    this.role = role;
    this.name = name;
    this.to = null;
    this.outSeq = 0;
    this.inSeq = 0;
    this.outbox = [];
    this.partial = '';
    this.onMessage = null;
  }

  reset(to) {
    this.to = to;
    this.outSeq = 0;
    this.inSeq = 0;
    this.outbox = [];
    this.partial = '';
    this.publish();
  }

  send(msg) {
    const s = JSON.stringify(msg);
    for (let i = 0; i < s.length; i += CHUNK) this.outbox.push([++this.outSeq, s.slice(i, i + CHUNK), i + CHUNK >= s.length ? 1 : 0]);
    this.publish();
  }

  publish() {
    const q = [];
    let size = 0;
    for (const e of this.outbox) {
      const n = JSON.stringify(e).length + 1;
      if (q.length && size + n > BUDGET) break;
      q.push(e);
      size += n;
    }
    this.named.presence({ r: this.role, n: this.name, to: this.to, ack: this.inSeq, q }).catch(() => {});
  }

  // Read the counterpart's presence: take new chunks in order, drop what it has acknowledged.
  absorb(pr) {
    let changed = false;
    const ack = typeof pr.ack === 'number' ? pr.ack : 0;
    if (this.outbox.length && this.outbox[0][0] <= ack) {
      this.outbox = this.outbox.filter((e) => e[0] > ack);
      changed = true;
    }
    if (Array.isArray(pr.q)) {
      for (const e of pr.q) {
        if (!Array.isArray(e) || e[0] !== this.inSeq + 1 || typeof e[1] !== 'string') continue;
        this.inSeq++;
        this.partial += e[1];
        changed = true;
        if (e[2]) {
          const text = this.partial;
          this.partial = '';
          let msg = null;
          try {
            msg = JSON.parse(text);
          } catch {
            msg = null;
          }
          if (msg && this.onMessage) this.onMessage(msg);
        }
      }
    }
    if (changed) this.publish();
  }
}

const myPeer = (peers) => {
  const me = peers.find((p) => p.isMe && p.sameTab);
  return me ? me.peer : null;
};

// Creator's side of a peer-to-peer room: runs the RoomCore here and serves one guest.
export class PresenceHost {
  constructor(room, name) {
    this.api = room;
    this.code = makeRoomCode();
    this.core = new RoomCore({ code: this.code });
    this.local = new LocalLink(this.core, 0, name);
    this.named = null;
    this.stream = null;
    this.guest = null;
    this.dropTimer = null;
    this.closed = false;
    this.off = [];
  }

  async open() {
    this.named = await this.api.join('pv-' + this.code.toLowerCase());
    if (this.closed) {
      this.named.leave().catch(() => {});
      return;
    }
    this.stream = new PresenceStream(this.named, 'h', '');
    this.stream.onMessage = (msg) => {
      if (this.guest) this.core.onMessage(1, msg);
    };
    this.stream.publish();
    this.off.push(this.named.onPeers((change) => this.onPeers(change.peers)));
  }

  onPeers(peers) {
    if (this.closed) return;
    const guests = peers.filter((p) => !(p.isMe && p.sameTab) && p.presence && p.presence.r === 'g');
    const current = this.guest ? guests.find((p) => p.peer === this.guest) : null;
    if (this.guest && !current) {
      if (!this.dropTimer) this.dropTimer = setTimeout(() => this.dropGuest(), 5000);
    } else if (current && this.dropTimer) {
      clearTimeout(this.dropTimer);
      this.dropTimer = null;
    }
    if (!this.guest && guests.length) {
      const g = guests[0];
      this.guest = g.peer;
      this.stream.reset(g.peer);
      this.core.attach(1, cleanName(g.presence.n, 'Guest'), (msg) => this.stream.send(msg));
    }
    const live = this.guest ? guests.find((p) => p.peer === this.guest) : null;
    if (live && live.presence.to === myPeer(peers)) this.stream.absorb(live.presence);
  }

  dropGuest() {
    this.dropTimer = null;
    if (!this.guest) return;
    this.guest = null;
    this.stream.reset(null);
    this.core.detach(1);
    if (this.named) this.onPeers(this.named.peers());
  }

  close() {
    this.closed = true;
    if (this.dropTimer) clearTimeout(this.dropTimer);
    for (const f of this.off) f();
    this.local.close();
    this.core.dispose();
    if (this.named) this.named.leave().catch(() => {});
  }
}

// Joiner's side of a peer-to-peer room.
export class PresenceGuest {
  constructor(room, code, name) {
    this.api = room;
    this.code = String(code).toUpperCase().trim();
    this.name = cleanName(name, 'Guest');
    this.onMessage = null;
    this.onStatus = null;
    this.named = null;
    this.stream = null;
    this.host = null;
    this.closed = false;
    this.found = false;
    this.off = [];
    this.timer = null;
    this.lostTimer = null;
  }

  status(s) {
    if (this.onStatus) this.onStatus(s);
  }

  async open() {
    try {
      this.named = await this.api.join('pv-' + this.code.toLowerCase());
    } catch {
      this.status('failed');
      return;
    }
    if (this.closed) {
      this.named.leave().catch(() => {});
      return;
    }
    this.stream = new PresenceStream(this.named, 'g', this.name);
    this.stream.onMessage = (msg) => {
      if (this.onMessage) this.onMessage(msg);
    };
    this.stream.publish();
    this.off.push(this.named.onPeers((change) => this.onPeers(change.peers)));
    this.timer = setTimeout(() => {
      if (!this.found) this.status('notfound');
    }, 7000);
  }

  onPeers(peers) {
    if (this.closed) return;
    const host = peers.find((p) => !(p.isMe && p.sameTab) && p.presence && p.presence.r === 'h');
    if (!host) {
      if (this.found && !this.lostTimer) this.lostTimer = setTimeout(() => this.status('lost'), 6000);
      return;
    }
    if (this.lostTimer) {
      clearTimeout(this.lostTimer);
      this.lostTimer = null;
    }
    const me = myPeer(peers);
    if (this.host !== host.peer) {
      this.host = host.peer;
      this.stream.reset(host.peer);
    }
    if (host.presence.to === me) {
      if (!this.found) {
        this.found = true;
        this.status('open');
      }
      this.stream.absorb(host.presence);
    } else if (host.presence.to && !this.found) {
      this.status('full');
    }
  }

  send(msg) {
    if (this.stream && this.found) this.stream.send(msg);
  }

  close() {
    this.closed = true;
    clearTimeout(this.timer);
    clearTimeout(this.lostTimer);
    for (const f of this.off) f();
    if (this.named) this.named.leave().catch(() => {});
  }
}
