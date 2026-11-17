// Builds two single-file versions of the game:
//   dist/pavilion-cricket.html  opens straight from disk (play the computer; online needs a server address)
//   dist/artifact.html          the same game packaged as a Claude artifact page

import { build } from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MARKUP } from '../client/src/markup.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(ROOT, 'dist');
fs.mkdirSync(out, { recursive: true });

// The artifact page loads three.js from a CDN its sandbox allows and reads the global.
const THREE_CDN = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js';
const threeGlobal = {
  name: 'three-global',
  setup(b) {
    b.onResolve({ filter: /^three$/ }, () => ({ path: 'three', namespace: 'three-global' }));
    b.onLoad({ filter: /.*/, namespace: 'three-global' }, () => ({ contents: 'module.exports = window.THREE;', loader: 'js' }));
  },
};

async function bundle(plugins = []) {
  const result = await build({
    entryPoints: [path.join(ROOT, 'client/src/main.js')],
    bundle: true,
    format: 'iife',
    minify: true,
    write: false,
    target: 'es2020',
    legalComments: 'none',
    plugins,
  });
  return result.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
}
const js = await bundle();
const jsLite = await bundle([threeGlobal]);
const css = fs.readFileSync(path.join(ROOT, 'client/styles.css'), 'utf8');
const fonts = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Saira+Condensed:ital,wght@0,500;0,600;0,700;0,800;1,800&display=swap">';

const standalone = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Pavilion Cricket</title>${fonts}<style>${css}</style></head><body>${MARKUP}<script>${js}</script></body></html>`;
const artifact = `<title>Pavilion Cricket</title>\n${fonts}\n<style>${css}</style>\n${MARKUP}\n<script src="${THREE_CDN}"></script>\n<script>${jsLite}</script>\n`;

fs.writeFileSync(path.join(out, 'pavilion-cricket.html'), standalone);
fs.writeFileSync(path.join(out, 'artifact.html'), artifact);
const kb = (s) => (Buffer.byteLength(s) / 1024).toFixed(0) + ' KB';
console.log(`dist/pavilion-cricket.html ${kb(standalone)}`);
console.log(`dist/artifact.html         ${kb(artifact)}`);
