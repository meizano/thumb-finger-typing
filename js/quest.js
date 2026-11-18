// Meta Quest 3 version: WebXR hand tracking joints -> ThumbTypeEngine.
// Renders a floating text panel and labels on each finger segment, over passthrough.

import * as THREE from 'three';
import { ThumbTypeEngine, NUM_SEGS, segmentCenter } from './gesture.js';
import { ThumbKeyboard } from './keyboard.js';

const $ = (id) => document.getElementById(id);

// WebXR joint names in MediaPipe's 21-landmark order.
const XR_JOINTS = [
  'wrist',
  'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
  'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip',
  'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip',
  'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip',
  'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip',
];

// Joint positions are tracked accurately in 3D, so thresholds can be a bit
// tighter than for the camera version.
const engine = new ThumbTypeEngine({ touchThreshold: 0.17, releaseThreshold: 0.30, stableFrames: 2 });
let dirty = true;
const kb = new ThumbKeyboard({ onChange: () => { dirty = true; } });

// ---------------------------------------------------------------- three.js scene
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType('local');
document.body.appendChild(renderer.domElement);
renderer.domElement.style.display = 'none';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 20);
scene.add(new THREE.AmbientLight(0xffffff, 1));

function canvasPlane(wPx, hPx, wM, hM) {
  const c = document.createElement('canvas'); c.width = wPx; c.height = hPx;
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthTest: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(wM, hM), mat);
  mesh.renderOrder = 10;
  return { c, g: c.getContext('2d'), tex, mesh };
}

// Text panel
const panel = canvasPlane(1024, 360, 0.6, 0.21);
scene.add(panel.mesh);
function drawPanel() {
  const { g, c } = panel;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = 'rgba(16,22,30,0.9)'; g.beginPath(); g.roundRect(0, 0, c.width, c.height, 36); g.fill();
  g.font = '56px system-ui, sans-serif'; g.textBaseline = 'top';
  const all = kb.text.split('\n');
  const lines = all.slice(-3);
  lines.forEach((ln, i) => {
    g.fillStyle = '#e8eef5';
    const isLast = i === lines.length - 1;
    const t = isLast ? ln : ln;
    g.fillText(t.slice(-30), 36, 36 + i * 70);
    if (isLast) {
      const w = g.measureText(t.slice(-30)).width;
      g.fillStyle = '#ffb547'; g.fillText(kb.pendingChar() + '▏', 36 + w, 36 + i * 70);
    }
  });
  g.font = '38px system-ui, sans-serif'; g.fillStyle = '#8a99aa';
  const L = engine.last;
  const st = !L?.hand ? 'no hand' : L.pressed >= 0 ? `P${L.pressed + 1} held` : L.armed ? 'ready' : 'open palm to re-arm';
  g.fillText(`${kb.mode === 'menu' ? 'MENU · ' + (kb.lastAction || '') : kb.caseName}   ·   ${st}`, 36, c.height - 64);
  panel.tex.needsUpdate = true;
}

// Segment labels
const labels = Array.from({ length: NUM_SEGS }, () => {
  const p = canvasPlane(256, 128, 0.036, 0.018);
  p.key = ''; scene.add(p.mesh); p.mesh.visible = false;
  return p;
});
function drawLabel(p, text, state) {
  const key = text + state;
  if (p.key === key) return;
  p.key = key;
  const { g, c } = p;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = state === 'held' ? '#ffb547' : state === 'hover' ? '#3fb8ff' : 'rgba(10,15,22,0.75)';
  g.beginPath(); g.roundRect(4, 4, c.width - 8, c.height - 8, 56); g.fill();
  g.fillStyle = state === 'idle' ? '#fff' : '#031019';
  g.font = `600 ${text.length > 3 ? 54 : 72}px system-ui, sans-serif`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, c.width / 2, c.height / 2 + 4);
  p.tex.needsUpdate = true;
}

const thumbDot = new THREE.Mesh(new THREE.SphereGeometry(0.006), new THREE.MeshBasicMaterial({ color: 0x4ade80, depthTest: false }));
thumbDot.renderOrder = 11; scene.add(thumbDot);

// ---------------------------------------------------------------- per-frame
const tmp = new THREE.Vector3(), camPos = new THREE.Vector3(), camDir = new THREE.Vector3();
let panelPlaced = false;

function readHand(frame, ref, handedness) {
  for (const src of frame.session.inputSources) {
    if (!src.hand || src.handedness !== handedness) continue;
    const pts = [];
    for (const name of XR_JOINTS) {
      const space = src.hand.get(name);
      const pose = space && frame.getJointPose(space, ref);
      if (!pose) return null;
      const p = pose.transform.position;
      pts.push({ x: p.x, y: p.y, z: p.z });
    }
    return pts;
  }
  return null;
}

function onXRFrame(time, frame) {
  if (!frame) return;
  const ref = renderer.xr.getReferenceSpace();
  const xrCam = renderer.xr.getCamera();
  xrCam.getWorldPosition(camPos); xrCam.getWorldDirection(camDir);

  // Panel follows the head loosely: 0.7 m ahead, slightly above eye line.
  tmp.copy(camDir).setY(0).normalize().multiplyScalar(0.7).add(camPos); tmp.y = camPos.y + 0.12;
  if (!panelPlaced) { panel.mesh.position.copy(tmp); panelPlaced = true; } else panel.mesh.position.lerp(tmp, 0.02);
  panel.mesh.lookAt(camPos);

  const pts = readHand(frame, ref, $('handSel').value);
  const events = engine.update(pts, time);
  for (const ev of events) kb.handle(ev);
  if (events.length) dirty = true;

  const L = engine.last;
  const stateKey = `${L?.hand}|${L?.pressed}|${L?.armed}`;
  if (stateKey !== onXRFrame.lastKey) { onXRFrame.lastKey = stateKey; dirty = true; }
  if (dirty) { drawPanel(); dirty = false; }

  for (let s = 0; s < NUM_SEGS; s++) {
    const p = labels[s];
    p.mesh.visible = !!pts;
    if (!pts) continue;
    const c = segmentCenter(pts, s);
    p.mesh.position.set(c.x, c.y, c.z);
    tmp.copy(camPos).sub(p.mesh.position).normalize().multiplyScalar(0.012);
    p.mesh.position.add(tmp);            // lift toward the viewer so it is not inside the finger
    p.mesh.quaternion.copy(xrCam.quaternion);
    const held = L.pressed === s;
    const hover = !held && L.hover === s && L.features.d[s] < engine.opts.releaseThreshold;
    drawLabel(p, kb.label(s), held ? 'held' : hover ? 'hover' : 'idle');
  }
  thumbDot.visible = !!pts;
  if (pts) {
    thumbDot.position.set(pts[4].x, pts[4].y, pts[4].z);
    thumbDot.material.color.set(L.pressed >= 0 ? 0xffb547 : L.armed ? 0x4ade80 : 0x8a99aa);
  }
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- session
async function enter(mode) {
  try {
    const session = await navigator.xr.requestSession(mode, { optionalFeatures: ['hand-tracking', 'local-floor'] });
    renderer.domElement.style.display = 'block';
    await renderer.xr.setSession(session);
    panelPlaced = false; engine.reset(); dirty = true;
    renderer.setAnimationLoop(onXRFrame);
    session.addEventListener('end', () => { renderer.setAnimationLoop(null); renderer.domElement.style.display = 'none'; });
  } catch (err) {
    $('xrStatus').textContent = `Could not start ${mode}: ${err.message}`;
  }
}

(async () => {
  const ar = $('btnAR'), vr = $('btnVR');
  if (!navigator.xr) { ar.textContent = 'WebXR not available'; $('xrStatus').textContent = 'Open this page in the Meta Quest Browser (HTTPS).'; return; }
  const [arOk, vrOk] = await Promise.all([
    navigator.xr.isSessionSupported('immersive-ar').catch(() => false),
    navigator.xr.isSessionSupported('immersive-vr').catch(() => false),
  ]);
  ar.textContent = arOk ? 'Enter AR (passthrough)' : 'AR not supported';
  ar.disabled = !arOk; vr.disabled = !vrOk;
  ar.onclick = () => enter('immersive-ar');
  vr.onclick = () => enter('immersive-vr');
  if (!arOk && !vrOk) $('xrStatus').textContent = 'This browser has no immersive WebXR. Use a Quest 3 headset.';
})();

window.thumbtype = { engine, kb };
