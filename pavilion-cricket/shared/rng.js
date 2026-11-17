// Seeded PRNG built on integer arithmetic so it gives the same stream in every JS engine.

export function makeRng(seed) {
  let a = (seed >>> 0) || 0x9e3779b9;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const rng = () => next();
  rng.range = (lo, hi) => lo + (hi - lo) * next();
  rng.int = (lo, hi) => lo + Math.floor(next() * (hi - lo + 1));
  rng.chance = (p) => next() < p;
  // Approximately normal(0, 1) from four uniforms; no transcendental functions.
  rng.gauss = () => (next() + next() + next() + next() - 2) * 1.7320508;
  rng.pick = (arr) => arr[Math.floor(next() * arr.length)];
  rng.weighted = (items, weightOf) => {
    let total = 0;
    for (const it of items) total += Math.max(0, weightOf(it));
    let r = next() * total;
    for (const it of items) {
      r -= Math.max(0, weightOf(it));
      if (r <= 0) return it;
    }
    return items[items.length - 1];
  };
  rng.seed = () => Math.floor(next() * 4294967296) >>> 0;
  return rng;
}

export function randomSeed() {
  return (Math.floor(Math.random() * 4294967296) ^ (Date.now() & 0xffffffff)) >>> 0;
}
