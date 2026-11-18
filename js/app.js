// Camera version: MediaPipe hand landmarks -> ThumbTypeEngine -> ThumbKeyboard.
// Two views: flat screen (desktop/phone) and Google Cardboard side-by-side.

import { ThumbTypeEngine, KNN, NUM_SEGS, OPEN, segmentCenter, computeFeatures, featureVector, FINGERS } from './gesture.js';
import { ThumbKeyboard, MENU_ITEMS } from './keyboard.js';

const MP_VERSION = '0.10.14';
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');
const video = $('video');

// ---------------------------------------------------------------- settings
const SETTINGS_KEY = 'thumbtype.settings.v1';
const TRAIN_KEY = 'thumbtype.knn.v1';
const settings = {
  touch: 0.20, release: 0.34, stable: 3, openPalm: true, useKnn: false,
  space: 'world', mirror: true, sound: true, zoom: 1.0, shift: 0.0, facing: 'user',
};
function loadJSON(key) { try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch { return null; } }
function saveJSON(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* storage unavailable */ } }
Object.assign(settings, loadJSON(SETTINGS_KEY) || {});

const engine = new ThumbTypeEngine();
const saved = loadJSON(TRAIN_KEY);
if (saved) engine.knn = KNN.fromJSON(saved);

function applySettings() {
  Object.assign(engine.opts, {
    touchThreshold: settings.touch,
    releaseThreshold: Math.max(settings.release, settings.touch + 0.02),
    stableFrames: settings.stable,
    requireOpenPalm: settings.openPalm,
    useClassifier: settings.useKnn && engine.knn.size > 0,
  });
  saveJSON(SETTINGS_KEY, settings);
}

// ---------------------------------------------------------------- keyboard
let toastTimer = null;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 900);
}
const kb = new ThumbKeyboard({
  onChange: updateHud,
  onAction: (item) => toast(`${item.icon} ${item.label}`),
});

function updateHud() {
  $('text').textContent = kb.text;
  $('pending').textContent = kb.pendingChar();
  $('modeTag').textContent = kb.mode === 'menu' ? 'MENU' : 'TYPE';
  $('caseTag').textContent = kb.mode === 'menu' ? (kb.lastAction || '—') : kb.caseName;
  const box = $('textBox'); box.scrollTop = box.scrollHeight;
}

// ---------------------------------------------------------------- feedback
let audio = null;
function click(freq = 880) {
  if (!settings.sound) return;
  try {
    audio ||= new AudioContext();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = freq; g.gain.value = 0.08;
    g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.06);
    o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.07);
  } catch { /* no audio */ }
  navigator.vibrate?.(12);
}

let flash = { seg: -1, until: 0 };
function handleEvents(events) {
  for (const ev of events) {
    if (ev.type === 'press') { click(); flash = { seg: ev.seg, until: performance.now() + 250 }; }
    if (ev.type === 'longpress') click(1320);
    kb.handle(ev);
  }
}

// ---------------------------------------------------------------- debug keys
const DEBUG_KEYS = { '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '*': 9, q: 9, '0': 10, '#': 11, w: 11 };
const heldKeys = new Map();
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.repeat) return;
  const seg = DEBUG_KEYS[e.key.toLowerCase()];
  if (seg === undefined) return;
  const timer = setTimeout(() => { heldKeys.get(e.key).long = true; handleEvents([{ type: 'longpress', seg }]); }, engine.opts.longPressMs);
  heldKeys.set(e.key, { seg, timer, long: false });
  handleEvents([{ type: 'press', seg }]);
});
addEventListener('keyup', (e) => {
  const h = heldKeys.get(e.key);
  if (!h) return;
  clearTimeout(h.timer); heldKeys.delete(e.key);
  handleEvents([{ type: 'release', seg: h.seg, long: h.long }]);
});

// ---------------------------------------------------------------- camera + model
let landmarker = null;
async function initModel() {
  const { FilesetResolver, HandLandmarker } = await import(`${MP_URL}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
  const make = (delegate) => HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO', numHands: 1,
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { landmarker = await make('GPU'); } catch { landmarker = await make('CPU'); }
}

async function startCamera(facing) {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
  video.srcObject = stream;
  await video.play();
}

// ---------------------------------------------------------------- frame loop
let mode = 'off'; // 'screen' | 'cardboard'
let imageLm = null;
let lastVideoTime = -1;
let fpsCount = 0, fpsT = performance.now();
let calibration = null;
let injectedUntil = 0; // camera detection paused while test frames are injected

function toEnginePoints(result) {
  if (!result?.landmarks?.length) return null;
  if (settings.space === 'world' && result.worldLandmarks?.length) return result.worldLandmarks[0];
  const w = video.videoWidth, h = video.videoHeight;
  return result.landmarks[0].map((p) => ({ x: p.x * w, y: p.y * h, z: p.z * w }));
}

function frame() {
  if (mode === 'off') return;
  requestAnimationFrame(frame);
  if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime && performance.now() > injectedUntil) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    imageLm = result.landmarks?.[0] || null;
    const pts = toEnginePoints(result);
    if (calibration) calibrationStep(pts);
    else handleEvents(engine.update(pts));
    fpsCount++;
  }
  const now = performance.now();
  if (now - fpsT > 1000) { $('fps').textContent = `${fpsCount} fps`; fpsCount = 0; fpsT = now; }
  updateStateTag();
  render();
}

function updateStateTag() {
  const tag = $('stateTag'), L = engine.last;
  let txt = 'no hand', cls = 'tag';
  if (calibration) txt = 'calibrating';
  else if (L?.hand) {
    if (L.pressed >= 0) { txt = `P${L.pressed + 1} held`; cls += ' held'; }
    else if (L.armed) { txt = 'ready'; cls += ' armed'; }
    else txt = 'open palm to re-arm';
  }
  tag.textContent = txt; tag.className = cls;
}

// ---------------------------------------------------------------- rendering
function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
}
addEventListener('resize', resize);

function render() {
  const W = canvas.width, H = canvas.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  if (mode === 'cardboard') {
    const ew = W / 2;
    const shift = settings.shift * ew;
    drawEye(0, 0, ew, H, +shift);
    drawEye(ew, 0, ew, H, -shift);
    ctx.fillStyle = '#333'; ctx.fillRect(ew - 1, 0, 2, H);
  } else {
    drawEye(0, 0, W, H, 0);
  }
}

function drawEye(x, y, w, h, shift) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  const cardboard = mode === 'cardboard';
  const mirror = !cardboard && settings.mirror;
  const vw = video.videoWidth || 16, vh = video.videoHeight || 9;
  const scale = Math.max(w / vw, h / vh) * (cardboard ? settings.zoom : 1);
  const dw = vw * scale, dh = vh * scale;
  const ox = x + (w - dw) / 2 + shift, oy = y + (h - dh) / 2;
  const map = (p) => {
    let px = ox + p.x * dw;
    if (mirror) px = 2 * (x + w / 2) - px;
    return [px, oy + p.y * dh];
  };

  if (video.readyState >= 2) {
    if (mirror) {
      ctx.save(); ctx.translate(2 * (x + w / 2), 0); ctx.scale(-1, 1);
      ctx.drawImage(video, ox, oy, dw, dh); ctx.restore();
    } else ctx.drawImage(video, ox, oy, dw, dh);
  }
  ctx.fillStyle = 'rgba(0,0,0,0.25)'; ctx.fillRect(x, y, w, h);

  const unit = Math.min(w, h) / 100;
  if (imageLm) drawHand(map, unit);
  else drawLegend(x, y, w, h, unit);
  if (cardboard) drawCardboardHud(x, y, w, h, unit);
  ctx.restore();
}

function drawHand(map, unit) {
  const L = engine.last || {};
  // skeleton
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = Math.max(1, unit * 0.4);
  const chains = [[0, 1, 2, 3, 4], ...FINGERS.map((f) => [0, ...f])];
  for (const c of chains) {
    ctx.beginPath();
    c.forEach((i, k) => { const [px, py] = map(imageLm[i]); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke();
  }
  // thumb tip
  const [tx, ty] = map(imageLm[4]);
  ctx.fillStyle = L.pressed >= 0 ? '#ffb547' : (L.armed ? '#4ade80' : '#8a99aa');
  ctx.beginPath(); ctx.arc(tx, ty, unit * 1.4, 0, Math.PI * 2); ctx.fill();

  // segment labels
  const now = performance.now();
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let s = 0; s < NUM_SEGS; s++) {
    const [px, py] = map(segmentCenter(imageLm, s));
    const held = L.pressed === s || (flash.seg === s && now < flash.until);
    const hover = !held && L.hover === s && L.features && L.features.d[s] < engine.opts.releaseThreshold;
    const label = kb.label(s);
    const fs = unit * (label.length > 3 ? 2.6 : 3.4);
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const tw = Math.max(ctx.measureText(label).width + unit * 2, unit * 5);
    const th = fs * 1.5;
    ctx.fillStyle = held ? 'rgba(255,181,71,0.95)' : hover ? 'rgba(63,184,255,0.85)' : 'rgba(10,15,22,0.62)';
    roundRect(px - tw / 2, py - th / 2, tw, th, th / 2); ctx.fill();
    ctx.fillStyle = held || hover ? '#031019' : '#fff';
    ctx.fillText(label, px, py + fs * 0.05);
  }
}

function drawLegend(x, y, w, h, unit) {
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.font = `${unit * 4}px system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(landmarker ? 'Show your open hand, palm to camera' : 'Loading hand model…', x + w / 2, y + h * 0.55);
}

function drawCardboardHud(x, y, w, h, unit) {
  // text panel across the top of each eye
  const pw = w * 0.78, ph = h * 0.2, px = x + (w - pw) / 2, py = y + h * 0.08;
  ctx.fillStyle = 'rgba(16,22,30,0.88)'; roundRect(px, py, pw, ph, unit * 2); ctx.fill();
  const fs = unit * 4.2;
  ctx.font = `${fs}px system-ui, sans-serif`; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  const lines = wrap((kb.text + kb.pendingChar()).split('\n').pop() + '▏', pw - unit * 4);
  lines.slice(-2).forEach((ln, i) => { ctx.fillStyle = '#e8eef5'; ctx.fillText(ln, px + unit * 2, py + unit * 2 + i * fs * 1.25); });
  ctx.font = `${unit * 2.8}px system-ui, sans-serif`; ctx.fillStyle = '#8a99aa';
  const L = engine.last;
  const st = !L?.hand ? 'no hand' : L.pressed >= 0 ? `P${L.pressed + 1}` : L.armed ? 'ready' : 'open palm';
  ctx.fillText(`${kb.mode === 'menu' ? 'MENU ' + (kb.lastAction || '') : kb.caseName}  ·  ${st}`, px + unit * 2, py + ph - unit * 4.2);
}

function wrap(text, maxW) {
  const out = []; let line = '';
  for (const ch of text) {
    if (ctx.measureText(line + ch).width > maxW) { out.push(line); line = ch; } else line += ch;
  }
  out.push(line); return out;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
}

// ---------------------------------------------------------------- calibration (kNN training)
const CAL_ORDER = [OPEN, ...Array.from({ length: NUM_SEGS }, (_, i) => i)];
const CAL_SAMPLES = 40, CAL_PAUSE_MS = 1500;

function startCalibration() {
  calibration = { step: 0, count: 0, waitUntil: performance.now() + CAL_PAUSE_MS, knn: new KNN(7) };
  $('calib').hidden = false; $('settings').hidden = true;
  showCalibPrompt();
}
function showCalibPrompt() {
  const c = calibration, cls = CAL_ORDER[c.step];
  $('calibPrompt').textContent = cls === OPEN ? 'Open palm' : `Touch P${cls + 1}`;
  $('calibInfo').textContent = `Step ${c.step + 1} of ${CAL_ORDER.length}. ` +
    (cls === OPEN ? 'Hold your hand open, fingers spread, thumb away.' : `Thumb on the ${['tip', 'middle', 'base'][cls % 3]} of your ${['index', 'middle', 'ring', 'little'][Math.floor(cls / 3)]} finger. Move it a little while holding.`);
  $('calibBar').style.width = '0%';
}
function calibrationStep(pts) {
  const c = calibration;
  if (!pts || performance.now() < c.waitUntil) return;
  c.knn.add(featureVector(computeFeatures(pts)), CAL_ORDER[c.step]);
  c.count++;
  $('calibBar').style.width = `${(100 * c.count) / CAL_SAMPLES}%`;
  if (c.count >= CAL_SAMPLES) {
    click(); c.step++; c.count = 0; c.waitUntil = performance.now() + CAL_PAUSE_MS;
    if (c.step >= CAL_ORDER.length) return finishCalibration(true);
    showCalibPrompt();
  }
}
function finishCalibration(ok) {
  if (ok) {
    engine.knn = calibration.knn;
    saveJSON(TRAIN_KEY, engine.knn.toJSON());
    settings.useKnn = true; $('useKnn').checked = true;
    applySettings();
    toast('Calibrated');
  }
  calibration = null; engine.reset();
  $('calib').hidden = true;
}

// ---------------------------------------------------------------- UI wiring
function bindRange(id, key, fmt = (v) => v) {
  const el = $(id), out = $(id + 'Out');
  el.value = settings[key]; out.textContent = fmt(+el.value);
  el.addEventListener('input', () => { settings[key] = +el.value; out.textContent = fmt(+el.value); applySettings(); });
}
function bindCheck(id, key) {
  const el = $(id); el.checked = settings[key];
  el.addEventListener('change', () => { settings[key] = el.checked; applySettings(); });
}
bindRange('touch', 'touch', (v) => v.toFixed(2));
bindRange('release', 'release', (v) => v.toFixed(2));
bindRange('stable', 'stable');
bindRange('zoom', 'zoom', (v) => v.toFixed(2));
bindRange('shift', 'shift', (v) => v.toFixed(3));
bindCheck('openPalm', 'openPalm');
bindCheck('useKnn', 'useKnn');
bindCheck('mirror', 'mirror');
bindCheck('sound', 'sound');
$('space').value = settings.space;
$('space').addEventListener('change', (e) => { settings.space = e.target.value; engine.reset(); applySettings(); });
$('facing').value = settings.facing;

$('btnSettings').onclick = () => { $('settings').hidden = !$('settings').hidden; };
$('btnCloseSettings').onclick = () => { $('settings').hidden = true; };
$('btnClear').onclick = () => kb.clearText();
$('btnCopy').onclick = async () => { try { await navigator.clipboard.writeText(kb.text); toast('Copied'); } catch { toast('Copy failed'); } };
$('btnCalib').onclick = startCalibration;
$('btnCalibCancel').onclick = () => finishCalibration(false);
$('btnResetCal').onclick = () => { engine.knn = new KNN(7); saveJSON(TRAIN_KEY, null); settings.useKnn = false; $('useKnn').checked = false; applySettings(); toast('Training cleared'); };
$('btnExport').onclick = () => {
  const blob = new Blob([JSON.stringify({ format: 'thumbtype-knn-v1', features: '12 thumb-segment distances + thumb local xyz', classes: 'P1..P12 = 0..11, open = 12', ...engine.knn.toJSON() })], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'thumbtype-training.json'; a.click();
};
$('importFile').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { engine.knn = KNN.fromJSON(JSON.parse(await f.text())); saveJSON(TRAIN_KEY, engine.knn.toJSON()); applySettings(); toast(`Loaded ${engine.knn.size} samples`); }
  catch { toast('Invalid file'); }
};
$('btnExit').onclick = () => location.reload();

// In Cardboard, a screen tap (Cardboard button) toggles the settings drawer.
canvas.addEventListener('click', () => { if (mode === 'cardboard') $('settings').hidden = !$('settings').hidden; });

async function start(which) {
  const status = $('startStatus');
  status.textContent = 'Starting camera…';
  if (!window.isSecureContext) { status.textContent = 'Camera needs HTTPS (or localhost). See README.'; return; }
  try {
    settings.facing = which === 'cardboard' ? 'environment' : $('facing').value;
    if (which === 'cardboard') settings.mirror = false;
    saveJSON(SETTINGS_KEY, settings);
    if (which === 'cardboard') {
      await document.documentElement.requestFullscreen?.().catch(() => {});
      await screen.orientation?.lock?.('landscape').catch(() => {});
    }
    await startCamera(settings.facing);
  } catch (err) {
    status.textContent = `Camera error: ${err.message}`; return;
  }
  mode = which;
  $('start').hidden = true;
  $('hud').hidden = which === 'cardboard';
  $('mirror').checked = settings.mirror;
  resize(); applySettings(); updateHud();
  navigator.wakeLock?.request('screen').catch(() => {});
  requestAnimationFrame(frame);
  if (!landmarker) {
    try { await initModel(); }
    catch (err) { status.textContent = ''; toast('Model failed to load'); console.error(err); }
  }
}
$('btnScreen').onclick = () => start('screen');
$('btnCardboard').onclick = () => start('cardboard');

applySettings();
updateHud();
// Expose for debugging / automated tests.
window.thumbtype = {
  engine, kb, settings, MENU_ITEMS,
  // Feed one frame of landmarks without a camera: image = 21 normalised {x,y,z}, world = 21 points.
  inject(image, world) { injectedUntil = performance.now() + 3000; imageLm = image; handleEvents(engine.update(world || image)); },
};
