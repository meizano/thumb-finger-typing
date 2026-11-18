# Thumb Typing: a camera prototype of the sEMG wristband keyboard

This is a vision-based stand-in for the patent *Gelang Electromyography untuk Interaksi Ketik Teks pada Mixed Reality*. It uses camera hand tracking instead of sEMG signals. It recognises the same gestures: an **open palm** as the neutral state, and the **thumb touching one of 12 finger segments** (P1–P12, Gambar 5) as a key press.

| Segment | Finger | Key | Tap | Hold |
|---|---|---|---|---|
| P1 / P2 / P3 | index tip / middle / base | 1 / 2 / 3 | `.,?!'-` / `abc` / `def` | |
| P4 / P5 / P6 | middle finger | 4 / 5 / 6 | `ghi` / `jkl` / `mno` | |
| P7 / P8 / P9 | ring finger | 7 / 8 / 9 | `pqrs` / `tuv` / `wxyz` | |
| P10 | little finger tip | * | backspace | switch to **Menu UI** |
| P11 | little finger middle | 0 | space | |
| P12 | little finger base | # | case: `abc → Abc → ABC → 123` | Enter |

Typing works like multi-tap on a phone keypad: tap P2 twice quickly to get `b`. A letter is committed after 0.9 s, or when you press a different key. In **Menu UI** mode the 12 segments act as 12 app buttons (Home, Chat, Play/Pause, Vol ±, …). The ⌨ button (P12) returns to typing.

The gesture flow is: open palm (armed) → thumb touches a segment (press fires once it is stable for 3 frames) → open palm again (re-armed). You must return to an open palm before the next key, so a sliding thumb does not type extra letters.

## Versions

| File | Platform | Tracking |
|---|---|---|
| `index.html`, **Start on screen** | Desktop or phone browser | MediaPipe Hand Landmarker on the webcam |
| `index.html`, **Start Cardboard VR** | Android phone in a Google Cardboard viewer | Rear camera as stereo passthrough, with labels drawn on your fingers |
| `quest.html` | Meta Quest 3 browser | WebXR hand-tracking joints, over passthrough AR |

All three use the same engine (`js/gesture.js`) and keyboard (`js/keyboard.js`).

## Run

```bash
npm start
```

Then open http://localhost:8080 on the PC.

**Phone / Cardboard / Quest:** browsers only allow the camera and WebXR on **HTTPS** (localhost is the only exception). Pick one option:

1. **Host it as a static site** (easiest). Push the folder to GitHub and enable GitHub Pages, or drop it on Netlify. No build step is needed.
2. **Use a local HTTPS certificate.** Put `cert/key.pem` and `cert/cert.pem` in the folder (for example from `mkcert <your-PC-IP>`). `serve.mjs` then serves HTTPS and prints the LAN URL.
3. **Use a tunnel**, such as `cloudflared tunnel --url http://localhost:8080`.

On Android, *Add to Home screen* installs it as a full-screen app (PWA).

**Cardboard:** press *Start Cardboard VR*, rotate the phone to landscape, and put it in the viewer. Hold your hand about 30–40 cm in front with the palm facing you. Tap the screen (or the Cardboard button) to open settings, where *Zoom* and *Eye offset* align the image with the lenses.

## Calibration (the "machine learning" stage)

The default detector is geometric. It measures the 3D distance from the thumb tip to each finger segment, divided by hand size. **Calibrate** records 40 frames each of *open palm* and P1…P12. It then trains a per-user k-nearest-neighbour classifier on 15 features: the 12 distances plus the thumb position in a hand-local frame. This mirrors Fig. 4 of the patent (features → ML classifier → key command). *Export training data* saves the dataset as JSON for analysis.

In Settings you can also tune the touch and release distance, the number of stable frames, whether an open palm is required between keys, and 3D vs 2D landmark distance.

## Testing without a camera

- `npm test` runs the engine and keyboard tests on synthetic hands.
- In the page, keys `1`–`9`, `*`/`Q`, `0`, `#`/`W` act as P1–P12 (hold for long press).
- `thumbtype.inject(imageLandmarks, worldLandmarks)` in the console feeds one frame of landmarks.

## Moving to the real sEMG band

`ThumbTypeEngine` turns per-frame landmarks into `press` / `longpress` / `release` events for segments 0–11. An sEMG classifier that outputs the same 13 classes (P1–P12 + open) can feed `ThumbKeyboard.handle()` directly. The text-entry and menu layers stay the same, so this prototype can collect usability data such as WPM and error rate before the hardware exists.
