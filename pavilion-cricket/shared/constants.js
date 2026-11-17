// World dimensions and physics constants. Metres, seconds, Y up, pitch along Z.

export const HALF = 10.06;                  // stumps at z = ±HALF
export const CREASE = 1.22;                 // popping crease distance from stumps
export const STRIKER_STUMPS_Z = HALF;
export const BOWLER_STUMPS_Z = -HALF;
export const STRIKER_CREASE_Z = HALF - CREASE;
export const BOWLER_CREASE_Z = -(HALF - CREASE);
export const RUN_LENGTH = 2 * (HALF - CREASE);

export const STUMP_H = 0.71;
export const STUMP_HALF_W = 0.1143;
export const BALL_R = 0.036;

export const BOUNDARY_RX = 62;
export const BOUNDARY_RZ = 68;

export const G = 9.81;

// The delivery is played back slower than real time so it can be timed by hand.
export const DELIVERY_TIME_SCALE = 0.7;

// Contact planes for the two kinds of footwork.
export const FRONT_FOOT_Z = 8.25;
export const BACK_FOOT_Z = 9.25;
export const SWING_TIME = 0.14;             // press → bat arrives (ball-clock seconds)

export const TICK = 1 / 60;                 // RunPhase fixed step
export const RUN_TIME = 2.5;                // seconds for one run
export const TURN_TIME = 0.22;

export const OVERS_OPTIONS = [2, 5, 10, 20];

export function insideBoundary(x, z) {
  const a = x / BOUNDARY_RX;
  const b = z / BOUNDARY_RZ;
  return a * a + b * b < 1;
}

export function offSignFor(hand) {
  return hand === 'L' ? -1 : 1;
}

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const lerp = (a, b, t) => a + (b - a) * t;
