// Thumb-to-finger-segment gesture engine.
//
// Input: 21 hand landmarks in the MediaPipe Hands order, as 3D points in any
// consistent unit (MediaPipe world landmarks in metres, WebXR joint poses in
// metres). Everything is normalised by hand size, so the unit does not matter.
//
// Output: events {type:'press'|'longpress'|'release', seg} where seg is 0..11,
// matching the patent drawing (Gambar 5):
//   P1  P2  P3   index   tip / middle / base
//   P4  P5  P6   middle
//   P7  P8  P9   ring
//   P10 P11 P12  pinky
//
// Neutral state is the open palm. A key fires when the thumb tip touches a
// segment, and the engine re-arms only after the hand returns to open palm.

export const LM = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
};

// [MCP, PIP, DIP, TIP] landmark indices for index, middle, ring, pinky
export const FINGERS = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];

export const FINGER_NAMES = ['index', 'middle', 'ring', 'pinky'];
export const SEG_NAMES = ['tip', 'middle', 'base'];
export const NUM_SEGS = 12;
export const OPEN = 12; // class id for "open palm" in the trained classifier

// Segment s (0..11): finger = floor(s/3), part = s%3 (0 tip, 1 middle, 2 base)
export function segmentEndpoints(lm, s) {
  const [mcp, pip, dip, tip] = FINGERS[Math.floor(s / 3)];
  switch (s % 3) {
    case 0: return [lm[dip], lm[tip]];
    case 1: return [lm[pip], lm[dip]];
    default: return [lm[mcp], lm[pip]];
  }
}

export function segmentCenter(lm, s) {
  const [a, b] = segmentEndpoints(lm, s);
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z || 0) + (b.z || 0)) / 2 };
}

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) });
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
const len = (a) => Math.sqrt(dot(a, a));
export const dist = (a, b) => len(sub(a, b));
const norm = (a) => { const l = len(a) || 1; return { x: a.x / l, y: a.y / l, z: a.z / l }; };

export function pointSegmentDistance(p, a, b) {
  const ab = sub(b, a);
  const ap = sub(p, a);
  const d = dot(ab, ab);
  const t = d > 0 ? Math.max(0, Math.min(1, dot(ap, ab) / d)) : 0;
  return dist(p, { x: a.x + ab.x * t, y: a.y + ab.y * t, z: (a.z || 0) + ab.z * t });
}

export function handScale(lm) {
  // wrist -> middle MCP, plus index MCP -> pinky MCP (palm width) for robustness
  return (dist(lm[0], lm[9]) + dist(lm[5], lm[17])) / 2 || 1;
}

// A finger counts as extended when its tip is clearly farther from the wrist
// than its PIP joint (works for any hand orientation).
export function fingerExtended(lm, f) {
  const [mcp, pip, , tip] = FINGERS[f];
  const w = lm[0];
  return dist(lm[tip], w) > dist(lm[pip], w) * 1.12 && dist(lm[tip], w) > dist(lm[mcp], w) * 1.3;
}

// Per-frame geometric features.
export function computeFeatures(lm) {
  const scale = handScale(lm);
  const thumb = lm[LM.THUMB_TIP];
  const d = new Array(NUM_SEGS);
  for (let s = 0; s < NUM_SEGS; s++) {
    const [a, b] = segmentEndpoints(lm, s);
    d[s] = pointSegmentDistance(thumb, a, b) / scale;
  }
  let best = 0;
  for (let s = 1; s < NUM_SEGS; s++) if (d[s] < d[best]) best = s;

  // Thumb tip in a hand-local frame (origin wrist, x across palm, y up the
  // middle finger, z palm normal), normalised by hand size.
  const o = lm[0];
  const yAxis = norm(sub(lm[9], o));
  const across = sub(lm[17], lm[5]);
  const zAxis = norm(cross(across, yAxis));
  const xAxis = cross(yAxis, zAxis);
  const t = sub(thumb, o);
  const local = [dot(t, xAxis) / scale, dot(t, yAxis) / scale, dot(t, zAxis) / scale];

  const extended = [0, 1, 2, 3].map((f) => fingerExtended(lm, f));
  return { scale, d, best, bestDist: d[best], local, extended };
}

// Feature vector for the trainable classifier.
export function featureVector(f) {
  return [...f.d, ...f.local.map((v) => v * 0.5)];
}

// ---------------------------------------------------------------------------
// k-nearest-neighbour classifier: the image-based stand-in for the patent's
// "machine learning classification" stage (Gambar 4). Trained per user from a
// short calibration session; classes 0..11 = segments, 12 = open palm.
export class KNN {
  constructor(k = 7) { this.k = k; this.X = []; this.y = []; }
  get size() { return this.X.length; }
  clear() { this.X = []; this.y = []; }
  add(x, label) { this.X.push(x); this.y.push(label); }
  classes() { return [...new Set(this.y)].sort((a, b) => a - b); }
  predict(x) {
    if (!this.X.length) return { label: -1, confidence: 0 };
    const k = Math.min(this.k, this.X.length);
    const best = []; // [dist, label] sorted ascending, length <= k
    for (let i = 0; i < this.X.length; i++) {
      const xi = this.X[i];
      let s = 0;
      for (let j = 0; j < xi.length; j++) { const v = xi[j] - x[j]; s += v * v; }
      if (best.length < k || s < best[best.length - 1][0]) {
        best.push([s, this.y[i]]);
        best.sort((a, b) => a[0] - b[0]);
        if (best.length > k) best.pop();
      }
    }
    const votes = new Map();
    for (const [s, l] of best) votes.set(l, (votes.get(l) || 0) + 1 / (Math.sqrt(s) + 1e-3));
    let label = -1, top = -1, total = 0;
    for (const [l, v] of votes) { total += v; if (v > top) { top = v; label = l; } }
    return { label, confidence: top / total };
  }
  toJSON() { return { k: this.k, X: this.X, y: this.y }; }
  static fromJSON(o) { const m = new KNN(o.k); m.X = o.X || []; m.y = o.y || []; return m; }
}

// ---------------------------------------------------------------------------
export const DEFAULTS = {
  touchThreshold: 0.20,   // thumb-to-segment distance / hand size to count as touch
  releaseThreshold: 0.34, // must move this far away to release (hysteresis)
  stableFrames: 3,        // consecutive frames on the same segment before firing
  openFrames: 2,          // consecutive open-palm frames needed to re-arm
  requireOpenPalm: true,  // re-arm only after seeing an open palm
  longPressMs: 650,
  useClassifier: false,   // use trained KNN instead of pure geometry
  minConfidence: 0.6,
};

export class ThumbTypeEngine {
  constructor(opts = {}) {
    this.opts = { ...DEFAULTS, ...opts };
    this.knn = new KNN(7);
    this.reset();
  }

  reset() {
    this.armed = true;
    this.pressed = -1;      // segment currently held
    this.pressStart = 0;
    this.longFired = false;
    this.candidate = -1;
    this.candidateFrames = 0;
    this.openCount = 0;
    this.last = null;       // last frame info for rendering
  }

  // Raw per-frame classification: segment 0..11, OPEN (12), or -1 (between
  // touch and release thresholds / not open: "hover").
  classify(f) {
    const o = this.opts;
    const fingersOpen = f.extended.filter(Boolean).length;
    if (o.useClassifier && this.knn.size > 0) {
      const { label, confidence } = this.knn.predict(featureVector(f));
      if (label === OPEN) return { raw: OPEN, hover: f.best, confidence };
      if (label >= 0 && confidence >= o.minConfidence) return { raw: label, hover: label, confidence };
      return { raw: -1, hover: f.best, confidence };
    }
    if (f.bestDist <= o.touchThreshold) return { raw: f.best, hover: f.best, confidence: 1 };
    if (f.bestDist >= o.releaseThreshold && fingersOpen >= 3) return { raw: OPEN, hover: f.best, confidence: 1 };
    return { raw: -1, hover: f.best, confidence: 0 };
  }

  // landmarks: array of 21 points, or null when no hand is visible.
  update(landmarks, now = performance.now()) {
    const events = [];
    if (!landmarks) {
      if (this.pressed >= 0) events.push({ type: 'release', seg: this.pressed, long: this.longFired });
      const armed = !this.opts.requireOpenPalm; // lost hand: need open palm again
      this.reset();
      this.armed = armed;
      this.last = { hand: false, armed: this.armed };
      return events;
    }

    const f = computeFeatures(landmarks);
    const c = this.classify(f);
    const o = this.opts;

    // Held key: check for release / long press.
    if (this.pressed >= 0) {
      const stillTouching = o.useClassifier
        ? c.raw === this.pressed || (c.raw === -1 && f.d[this.pressed] < o.releaseThreshold)
        : f.d[this.pressed] < o.releaseThreshold;
      if (!stillTouching) {
        events.push({ type: 'release', seg: this.pressed, long: this.longFired });
        this.pressed = -1;
        this.longFired = false;
      } else if (!this.longFired && now - this.pressStart >= o.longPressMs) {
        this.longFired = true;
        events.push({ type: 'longpress', seg: this.pressed });
      }
    }

    // Arming: open palm (or, if not required, simply thumb away from fingers).
    if (c.raw === OPEN || (!o.requireOpenPalm && f.bestDist >= o.releaseThreshold)) {
      this.openCount++;
      if (this.openCount >= o.openFrames && this.pressed < 0) this.armed = true;
    } else {
      this.openCount = 0;
    }

    // Press detection with temporal stability.
    if (c.raw >= 0 && c.raw < NUM_SEGS) {
      if (c.raw === this.candidate) this.candidateFrames++;
      else { this.candidate = c.raw; this.candidateFrames = 1; }
      if (this.armed && this.pressed < 0 && this.candidateFrames >= o.stableFrames) {
        this.pressed = c.raw;
        this.pressStart = now;
        this.longFired = false;
        this.armed = false;
        events.push({ type: 'press', seg: c.raw });
      }
    } else {
      this.candidate = -1;
      this.candidateFrames = 0;
    }

    this.last = {
      hand: true, features: f, raw: c.raw, hover: c.hover, confidence: c.confidence,
      armed: this.armed, pressed: this.pressed, open: c.raw === OPEN,
    };
    return events;
  }
}
