// Run: node --test test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { ThumbTypeEngine, segmentCenter, computeFeatures, featureVector, KNN, OPEN, NUM_SEGS } from '../js/gesture.js';
import { ThumbKeyboard } from '../js/keyboard.js';

// Synthetic right hand, palm facing -z, metres. Fingers point +y.
function openHand() {
  const lm = [];
  lm[0] = { x: 0, y: 0, z: 0 };
  // thumb out to the side
  lm[1] = { x: 0.025, y: 0.02, z: 0 }; lm[2] = { x: 0.05, y: 0.04, z: 0 };
  lm[3] = { x: 0.07, y: 0.055, z: 0 }; lm[4] = { x: 0.09, y: 0.07, z: 0 };
  const xs = [0.03, 0.01, -0.01, -0.03];
  const lens = [[0.04, 0.025, 0.02], [0.045, 0.028, 0.022], [0.042, 0.026, 0.02], [0.033, 0.02, 0.018]];
  const base = [0.09, 0.095, 0.09, 0.08];
  for (let f = 0; f < 4; f++) {
    let y = base[f];
    const i = 5 + f * 4;
    lm[i] = { x: xs[f], y, z: 0 };
    for (let k = 0; k < 3; k++) { y += lens[f][k]; lm[i + k + 1] = { x: xs[f], y, z: 0 }; }
  }
  return lm;
}
function touching(s) {
  const lm = openHand();
  const c = segmentCenter(lm, s);
  lm[4] = { x: c.x, y: c.y, z: c.z - 0.008 }; // thumb pad just in front of the segment
  return lm;
}

function run(engine, frames, t0 = 0) {
  const ev = [];
  frames.forEach((lm, i) => ev.push(...engine.update(lm, t0 + i * 33)));
  return ev;
}
const rep = (n, f) => Array.from({ length: n }, f);

test('geometry picks the touched segment for all 12 segments', () => {
  for (let s = 0; s < NUM_SEGS; s++) {
    const f = computeFeatures(touching(s));
    assert.equal(f.best, s, `segment P${s + 1}`);
    assert.ok(f.bestDist < 0.2, `P${s + 1} distance ${f.bestDist}`);
  }
  const open = computeFeatures(openHand());
  assert.ok(open.bestDist > 0.34, `open palm distance ${open.bestDist}`);
  assert.equal(open.extended.filter(Boolean).length, 4);
});

test('open palm -> touch fires one press, then release; re-arms after open palm', () => {
  const e = new ThumbTypeEngine();
  const ev = run(e, [...rep(3, openHand), ...rep(6, () => touching(4)), ...rep(3, openHand), ...rep(4, () => touching(0))]);
  assert.deepEqual(ev.map((x) => `${x.type}:${x.seg}`), ['press:4', 'release:4', 'press:0']);
});

test('no second press without returning to open palm', () => {
  const e = new ThumbTypeEngine();
  const ev = run(e, [...rep(3, openHand), ...rep(5, () => touching(1)), ...rep(5, () => touching(7))]);
  assert.deepEqual(ev.filter((x) => x.type === 'press').map((x) => x.seg), [1]);
});

test('long press fires after holding', () => {
  const e = new ThumbTypeEngine({ longPressMs: 300 });
  const ev = run(e, [...rep(3, openHand), ...rep(20, () => touching(11)), ...rep(3, openHand)]);
  assert.deepEqual(ev.map((x) => x.type), ['press', 'longpress', 'release']);
  assert.equal(ev[2].long, true);
});

test('kNN classifier learns segments from calibration samples', () => {
  const knn = new KNN(5);
  const jitter = (lm) => lm.map((p) => ({ x: p.x + (Math.random() - 0.5) * 0.003, y: p.y + (Math.random() - 0.5) * 0.003, z: p.z }));
  for (let r = 0; r < 20; r++) {
    knn.add(featureVector(computeFeatures(jitter(openHand()))), OPEN);
    for (let s = 0; s < NUM_SEGS; s++) knn.add(featureVector(computeFeatures(jitter(touching(s)))), s);
  }
  for (let s = 0; s < NUM_SEGS; s++) assert.equal(knn.predict(featureVector(computeFeatures(touching(s)))).label, s);
  assert.equal(knn.predict(featureVector(computeFeatures(openHand()))).label, OPEN);

  const e = new ThumbTypeEngine({ useClassifier: true });
  e.knn = knn;
  const ev = run(e, [...rep(3, openHand), ...rep(5, () => touching(6)), ...rep(3, openHand)]);
  assert.deepEqual(ev.map((x) => `${x.type}:${x.seg}`), ['press:6', 'release:6']);
});

test('multi-tap keyboard types "Hi there"', () => {
  const kb = new ThumbKeyboard({ multitapMs: 10_000 });
  const tap = (s) => { kb.handle({ type: 'press', seg: s }); kb.handle({ type: 'release', seg: s, long: false }); };
  // H = P4 x2, i = P4 x3 (needs commit between: tap a different key or wait)
  tap(3); tap(3); kb.commit();
  tap(3); tap(3); tap(3); tap(10);
  tap(7); tap(3); tap(3); tap(2); tap(2); tap(6); tap(6); tap(6); tap(2); tap(2); kb.commit();
  assert.equal(kb.text, 'Hi there');
  tap(9); // backspace (fires on release)
  assert.equal(kb.text, 'Hi ther');
  kb.handle({ type: 'press', seg: 9 }); kb.handle({ type: 'longpress', seg: 9 }); kb.handle({ type: 'release', seg: 9, long: true });
  assert.equal(kb.mode, 'menu');
  assert.equal(kb.text, 'Hi ther');
  kb.clearTimer();
});
