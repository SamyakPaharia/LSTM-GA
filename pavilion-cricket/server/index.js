// Game server: static files and WebSocket rooms on one port.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';
import { RoomCore, makeRoomCode } from '../shared/room.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PORT = +(process.env.PORT || 3000);
const MAX_ROOMS = +(process.env.MAX_ROOMS || 500);
const EMPTY_ROOM_TTL = 2 * 60 * 1000;
const IDLE_ROOM_TTL = 60 * 60 * 1000;

const MOUNTS = [
  ['/shared/', path.join(ROOT, 'shared')],
  ['/vendor/three/', path.join(ROOT, 'node_modules', 'three', 'build')],
  ['/', path.join(ROOT, 'client')],
];
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon',
};

const rooms = new Map();

function serveStatic(req, res) {
  let urlPath;
  try {
    urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  } catch {
    res.writeHead(400).end();
    return;
  }
  if (urlPath === '/health') {
    res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: true, rooms: rooms.size }));
    return;
  }
  for (const [prefix, dir] of MOUNTS) {
    if (!urlPath.startsWith(prefix)) continue;
    let rel = urlPath.slice(prefix.length);
    if (rel === '' || rel.endsWith('/')) rel += 'index.html';
    const file = path.resolve(dir, rel);
    if (file !== dir && !file.startsWith(dir + path.sep)) break;
    fs.stat(file, (err, st) => {
      if (err || !st.isFile()) {
        res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
        return;
      }
      res.writeHead(200, {
        'content-type': MIME[path.extname(file)] || 'application/octet-stream',
        'cache-control': prefix === '/vendor/three/' ? 'public, max-age=86400' : 'no-cache',
      });
      fs.createReadStream(file).pipe(res);
    });
    return;
  }
  res.writeHead(404, { 'content-type': 'text/plain' }).end('Not found');
}

const server = http.createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405).end();
    return;
  }
  serveStatic(req, res);
});

const wss = new WebSocketServer({ server, path: '/ws', maxPayload: 16 * 1024 });

function newCode() {
  for (let i = 0; i < 50; i++) {
    const code = makeRoomCode();
    if (!rooms.has(code)) return code;
  }
  return null;
}

wss.on('connection', (ws) => {
  const conn = { room: null, seat: -1, count: 0, stamp: Date.now(), alive: true };
  const send = (msg) => {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg));
  };
  const fail = (m) => send({ t: 'error', m });

  const bind = (entry, seat, name) => {
    conn.room = entry;
    conn.seat = seat;
    entry.sockets[seat] = ws;
    send({ t: 'joined', code: entry.core.code, seat, token: entry.tokens[seat] });
    entry.core.attach(seat, name, send);
  };

  ws.on('pong', () => {
    conn.alive = true;
  });

  ws.on('message', (data) => {
    const now = Date.now();
    if (now - conn.stamp > 1000) {
      conn.stamp = now;
      conn.count = 0;
    }
    if (++conn.count > 80) {
      ws.close(1008, 'rate limit');
      return;
    }
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (!msg || typeof msg !== 'object' || typeof msg.t !== 'string') return;

    if (!conn.room) {
      if (msg.t === 'create') {
        if (rooms.size >= MAX_ROOMS) return fail('The server is full. Try again in a few minutes.');
        const code = newCode();
        if (!code) return fail('Could not create a room. Try again.');
        const entry = { core: new RoomCore({ code }), tokens: [crypto.randomBytes(12).toString('hex'), crypto.randomBytes(12).toString('hex')], sockets: [null, null], emptySince: 0 };
        rooms.set(code, entry);
        bind(entry, 0, msg.name);
      } else if (msg.t === 'join') {
        const entry = rooms.get(String(msg.code || '').toUpperCase().trim());
        if (!entry) return fail('No room with that code. Check the code and try again.');
        const seat = entry.core.freeSeat();
        if (seat < 0) return fail('That room already has two players.');
        bind(entry, seat, msg.name);
      } else if (msg.t === 'rejoin') {
        const entry = rooms.get(String(msg.code || '').toUpperCase().trim());
        const seat = entry ? entry.tokens.indexOf(String(msg.token)) : -1;
        if (seat < 0 || !entry.core.seats[seat]) return fail('That match is no longer available.');
        const old = entry.sockets[seat];
        if (old && old !== ws) old.close(4000, 'replaced');
        bind(entry, seat, null);
      }
      return;
    }
    conn.room.core.onMessage(conn.seat, msg);
  });

  ws.on('close', () => {
    const entry = conn.room;
    if (!entry || entry.sockets[conn.seat] !== ws) return;
    entry.sockets[conn.seat] = null;
    entry.core.detach(conn.seat);
    if (entry.core.connectedCount() === 0) entry.emptySince = Date.now();
  });

  ws.on('error', () => {});
  ws.conn = conn;
});

setInterval(() => {
  const now = Date.now();
  for (const [code, entry] of rooms) {
    const empty = entry.core.connectedCount() === 0;
    if ((empty && now - (entry.emptySince || now) > EMPTY_ROOM_TTL) || now - entry.core.touched > IDLE_ROOM_TTL) {
      entry.core.dispose();
      for (const s of entry.sockets) if (s) s.close(4001, 'room closed');
      rooms.delete(code);
    }
  }
  for (const ws of wss.clients) {
    if (ws.conn && !ws.conn.alive) {
      ws.terminate();
      continue;
    }
    if (ws.conn) ws.conn.alive = false;
    ws.ping();
  }
}, 30000).unref();

server.listen(PORT, () => {
  console.log(`Pavilion Cricket running at http://localhost:${PORT}`);
});
