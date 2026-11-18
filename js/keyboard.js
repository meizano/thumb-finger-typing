// Text entry + menu UI driven by the 12 finger-segment buttons (P1..P12).
// Layout follows a 12-key phone keypad so it is learnable: P1..P9 are the
// digit keys 1..9, P10 = *, P11 = 0, P12 = #.

export const MULTITAP = [
  ['.', ',', '?', '!', "'", '-', '1'], // P1
  ['a', 'b', 'c', '2'],                 // P2
  ['d', 'e', 'f', '3'],                 // P3
  ['g', 'h', 'i', '4'],                 // P4
  ['j', 'k', 'l', '5'],                 // P5
  ['m', 'n', 'o', '6'],                 // P6
  ['p', 'q', 'r', 's', '7'],            // P7
  ['t', 'u', 'v', '8'],                 // P8
  ['w', 'x', 'y', 'z', '9'],            // P9
];
export const KEY_BACK = 9, KEY_SPACE = 10, KEY_SHIFT = 11;
export const CASES = ['abc', 'Abc', 'ABC', '123'];

export const MENU_ITEMS = [
  { icon: '🏠', label: 'Home' },     { icon: '💬', label: 'Chat' },     { icon: '🌐', label: 'Browser' },
  { icon: '▶', label: 'Play/Pause' }, { icon: '🔊', label: 'Vol +' },   { icon: '🔉', label: 'Vol −' },
  { icon: '📷', label: 'Camera' },    { icon: '📝', label: 'Notes' },   { icon: '⚙', label: 'Settings' },
  { icon: '←', label: 'Back' },       { icon: '✔', label: 'OK' },       { icon: '⌨', label: 'Keyboard' },
];

export class ThumbKeyboard {
  constructor({ multitapMs = 900, onChange = () => {}, onAction = () => {} } = {}) {
    this.multitapMs = multitapMs;
    this.onChange = onChange;
    this.onAction = onAction;
    this.mode = 'type';  // 'type' | 'menu'
    this.text = '';
    this.caseIdx = 1;    // start with "Abc" (capitalise sentence start)
    this.pending = null; // { seg, idx, time } multi-tap letter being chosen
    this.timer = null;
    this.lastAction = '';
  }

  get caseName() { return CASES[this.caseIdx]; }

  // Label shown on segment s in the current mode.
  label(s) {
    if (this.mode === 'menu') return MENU_ITEMS[s].icon;
    if (s === KEY_BACK) return '⌫';
    if (s === KEY_SPACE) return this.caseName === '123' ? '0' : '␣';
    if (s === KEY_SHIFT) return this.caseName;
    if (this.caseName === '123') return String(s + 1);
    const letters = MULTITAP[s].filter((c) => !/\d/.test(c)).join('');
    return this.caseName === 'abc' ? letters : letters.toUpperCase();
  }

  sublabel(s) {
    if (this.mode === 'menu') return MENU_ITEMS[s].label;
    if (s === KEY_BACK) return 'hold: menu';
    if (s === KEY_SHIFT) return 'hold: ⏎';
    if (s === KEY_SPACE) return 'space';
    return '';
  }

  // Keys with a long-press action fire their tap action on release instead of
  // on press, so a long press does not also trigger the tap.
  hasLongAction(s) { return this.mode === 'type' && (s === KEY_BACK || s === KEY_SHIFT); }

  handle(ev) {
    if (ev.type === 'press' && !this.hasLongAction(ev.seg)) this.tap(ev.seg);
    else if (ev.type === 'release' && this.hasLongAction(ev.seg) && !ev.long) this.tap(ev.seg);
    else if (ev.type === 'longpress' && this.hasLongAction(ev.seg)) this.longPress(ev.seg);
  }

  tap(s) {
    if (this.mode === 'menu') return this.menuTap(s);
    if (s === KEY_BACK) {
      if (this.pending) this.pending = null;
      else this.text = this.text.slice(0, -1);
      this.clearTimer();
    } else if (s === KEY_SPACE) {
      this.commit();
      this.text += this.caseName === '123' ? '0' : ' ';
    } else if (s === KEY_SHIFT) {
      this.commit();
      this.caseIdx = (this.caseIdx + 1) % CASES.length;
    } else if (this.caseName === '123') {
      this.commit();
      this.text += String(s + 1);
    } else {
      const now = Date.now();
      if (this.pending && this.pending.seg === s && now - this.pending.time < this.multitapMs) {
        this.pending.idx = (this.pending.idx + 1) % MULTITAP[s].length;
        this.pending.time = now;
      } else {
        this.commit();
        this.pending = { seg: s, idx: 0, time: now };
      }
      this.clearTimer();
      this.timer = setTimeout(() => { this.commit(); this.emit(); }, this.multitapMs);
    }
    this.emit();
  }

  longPress(s) {
    if (s === KEY_SHIFT) { this.commit(); this.text += '\n'; this.lastAction = 'Enter'; }
    if (s === KEY_BACK) { this.commit(); this.mode = 'menu'; this.lastAction = 'Menu'; }
    this.emit();
  }

  menuTap(s) {
    const item = MENU_ITEMS[s];
    if (item.label === 'Keyboard') { this.mode = 'type'; this.lastAction = 'Keyboard'; }
    else { this.lastAction = item.label; this.onAction(item, s); }
    this.emit();
  }

  pendingChar() {
    if (!this.pending) return '';
    let c = MULTITAP[this.pending.seg][this.pending.idx];
    if (this.caseName === 'ABC' || this.caseName === 'Abc') c = c.toUpperCase();
    return c;
  }

  commit() {
    if (!this.pending) return;
    const c = this.pendingChar();
    this.text += c;
    this.pending = null;
    this.clearTimer();
    if (this.caseName === 'Abc' && /[a-z]/i.test(c)) this.caseIdx = 0; // one-shot capital
    if (this.caseName === 'abc' && /[.!?]/.test(c)) this.caseIdx = 1;
  }

  clearTimer() { if (this.timer) { clearTimeout(this.timer); this.timer = null; } }
  clearText() { this.text = ''; this.pending = null; this.clearTimer(); this.emit(); }
  emit() { this.onChange(this); }
}
