"""Çil Yapı v2 identity: emerald + cream blocks, white caption pills, emerald panels (no gold 3D / yellow captions)."""
import numpy as np, cv2, sys
sys.path.insert(0, '/tmp/edit/pipeline')
from ugfx import text_mask, solid, over, shadowed, rrect_mask, padspr
from gfx import composite, place, ease_out, ease_in, ease_io, ease_out_back
EM = np.array([0.03, 0.54, 0.37], np.float32)
EM_D = np.array([0.02, 0.22, 0.16], np.float32)
MINT = np.array([0.60, 0.97, 0.78], np.float32)
CREAM = np.array([0.985, 0.975, 0.935], np.float32)
GOLDL = np.array([0.90, 0.74, 0.42], np.float32)
WH = np.array([1, 1, 1], np.float32)

def block(text, font, size, tracking, fg, bg, padx=34, pady=18, radius=10):
    m = text_mask(text, font, size, tracking)
    w, h = m.shape[1] + 2 * padx, m.shape[0] + 2 * pady
    box = solid(rrect_mask(w, h, radius), bg, 1.0)
    txt = np.zeros((h, w, 4), np.float32); over(txt, solid(m, fg), padx, pady)
    return box, txt

def block_title(l1, l2, s1, s2, tr1=6, tr2=20):
    b1, t1 = block(l1, 'Montserrat-Black.ttf', s1, tr1, WH, EM, padx=36, pady=16)
    b2, t2 = block(l2, 'Montserrat-ExtraBold.ttf', s2, tr2, EM_D, CREAM, padx=30, pady=12)
    return dict(b1=b1, t1=t1, b2=b2, t2=t2)

def _wipe(spr, p, from_left=True):
    """Reveal fraction p of a sprite horizontally with a soft edge."""
    h, w = spr.shape[:2]
    if p >= 1: return spr
    if p <= 0: return None
    xs = np.arange(w, dtype=np.float32)
    edge = p * (w + 24) - 12
    m = np.clip((edge - xs) / 12.0, 0, 1) if from_left else np.clip((xs - (w - edge)) / 12.0, 0, 1)
    return spr * m[None, :, None]

def draw_block_title(frame, T, spec, t_in, t_out, cy, slam=False):
    if not (t_in <= T < t_out): return
    W = frame.shape[1]
    u = T - t_in; q = np.clip((t_out - T) / 0.16, 0, 1)
    b1, t1, b2, t2 = spec['b1'], spec['t1'], spec['b2'], spec['t2']
    grow = 1.0 + 0.025 * np.clip(u / max(0.1, t_out - t_in), 0, 1)
    sc = grow * ((1.22 - 0.22 * ease_out(np.clip(u / 0.16, 0, 1), 3)) if slam else 1.0)
    h1, w1 = b1.shape[:2]; h2, w2 = b2.shape[:2]
    tot_h = h1 + h2 - 6
    y1 = cy - tot_h / 2 * sc; y2 = y1 + (h1 - 6) * sc
    def put(spr, cx, top, s):
        if spr is None: return
        h, w = spr.shape[:2]
        place(frame, spr, cx, top + h * s / 2, scale=s)
    # block 1: wipe from left, text slides in
    p1 = ease_out(np.clip(u / 0.22, 0, 1), 3); pt1 = ease_out(np.clip((u - 0.04) / 0.26, 0, 1), 3)
    o1 = _wipe(b1, p1, True)
    if q < 1: o1 = _wipe(b1, q, False) if o1 is not None else None
    put(o1, W / 2, y1, sc)
    if pt1 > 0:
        tt = t1 * pt1 * q
        if pt1 < 0.9: tt = cv2.blur(tt, (int(2 + 30 * (1 - pt1)), 1))
        place(frame, tt, W / 2 - 70 * (1 - pt1), y1 + h1 * sc / 2, scale=sc)
    # block 2: wipe from right
    p2 = ease_out(np.clip((u - 0.08) / 0.22, 0, 1), 3); pt2 = ease_out(np.clip((u - 0.12) / 0.26, 0, 1), 3)
    o2 = _wipe(b2, p2, False)
    if q < 1 and o2 is not None: o2 = _wipe(b2, q, True)
    put(o2, W / 2, y2, sc)
    if pt2 > 0:
        tt = t2 * pt2 * q
        if pt2 < 0.9: tt = cv2.blur(tt, (int(2 + 30 * (1 - pt2)), 1))
        place(frame, tt, W / 2 + 70 * (1 - pt2), y2 + h2 * sc / 2, scale=sc)
    # gold hairline under block 2
    pl = ease_io(np.clip((u - 0.20) / 0.28, 0, 1)) * q
    if pl > 0:
        half = w2 * 0.32 * pl * sc; y = int(y2 + (h2 + 16) * sc)
        cv2.line(frame, (int(W / 2 - half), y), (int(W / 2 + half), y), tuple(float(c) for c in GOLDL), 3, cv2.LINE_AA)

# ---------------------------------------------------------------- captions: cream pill, deep-green words, emerald word boxes
CAPF, CAPS = 'Montserrat-ExtraBold.ttf', 60
_cw = {}
def cap_mask(w):
    if w not in _cw: _cw[w] = text_mask(w, CAPF, CAPS, 0.3)
    return _cw[w]

def draw_caption_pill(frame, T, items, start, end, cy):
    W = frame.shape[1]
    ms = [cap_mask(wd) for wd, st, _, _ in items]
    gap = 16; pad_x, pad_y = 30, 18
    widths = [m.shape[1] + (20 if st == 'g' else 0) for m, (wd, st, _, _) in zip(ms, items)]
    total = sum(widths) + gap * (len(ms) - 1)
    hh = max(m.shape[0] for m in ms)
    sc = min(1.0, 900 / (total + 2 * pad_x))
    x0 = W / 2 - total * sc / 2
    xs = []; x = x0
    for w in widths: xs.append(x); x += (w + gap) * sc
    fo = np.clip((end - T) / 0.10, 0, 1)
    times = [t0 for _, _, t0, _ in items]
    shown = [T >= t - 0.04 for t in times]
    if not any(shown): return
    last = max(i for i, s in enumerate(shown) if s)
    # pill grows to cover spoken words
    tgt = xs[last] + widths[last] * sc
    kgrow = ease_out(np.clip((T - (times[last] - 0.04)) / 0.12, 0, 1), 3)
    prev_r = xs[last - 1] + widths[last - 1] * sc if last > 0 else xs[0] + 10
    right = prev_r + (tgt - prev_r) * kgrow
    pw = int(right - xs[0] + 2 * pad_x * sc); ph = int((hh + 2 * pad_y) * sc)
    kin = ease_out(np.clip((T - (times[0] - 0.04)) / 0.14, 0, 1), 3)
    if pw > 10:
        pill = solid(rrect_mask(pw, ph, int(ph * 0.42)), CREAM, 0.95 * fo)
        pill = shadowed(pill, blur=14, off=(0, 8), strength=0.35 * fo, pad=24)
        place(frame, pill, xs[0] - pad_x * sc + pw / 2, cy + (1 - kin) * 14, scale=0.94 + 0.06 * kin, opacity=float(kin))
    for i, (m, (wd, st, t0, _)) in enumerate(zip(ms, items)):
        if not shown[i]: continue
        p = np.clip((T - t0 + 0.04) / 0.14, 0, 1)
        s = sc * (0.78 + 0.22 * ease_out_back(p, 2.0))
        cx = xs[i] + widths[i] * sc / 2
        if st == 'g':
            bw, bh = m.shape[1] + 20, m.shape[0] + 14
            bx = solid(rrect_mask(bw, bh, 10), EM, 1.0); over(bx, solid(m, WH), 10, 7)
            place(frame, bx, cx, cy, scale=s, opacity=float(min(1, p * 2) * fo))
        else:
            place(frame, solid(m, EM_D), cx, cy, scale=s, opacity=float(min(1, p * 2) * fo))

# ---------------------------------------------------------------- PIP card with emerald frame, flip-in
def em_card(img_rgb, width, border=9, radius=24):
    h0, w0 = img_rgb.shape[:2]
    iw = width - 2 * border; ih = int(h0 * iw / w0)
    im = cv2.resize(img_rgb, (iw, ih), interpolation=cv2.INTER_AREA)
    Wc, Hc = width, ih + 2 * border
    frame = solid(rrect_mask(Wc, Hc, radius), EM, 1.0)
    inner = rrect_mask(iw, ih, max(4, radius - border))
    pic = np.concatenate([im * inner[..., None], inner[..., None]], -1).astype(np.float32)
    over(frame, pic, border, border)
    hl = np.zeros((Hc, Wc), np.float32)
    cv2.rectangle(hl, (border - 2, border - 2), (border + iw + 1, border + ih + 1), 1.0, 2, cv2.LINE_AA)
    over(frame, solid(hl, CREAM, 0.8), 0, 0)
    return shadowed(frame, blur=24, off=(0, 16), strength=0.55, pad=50)

def draw_em_card(frame, spr, T, t_in, t_out, cy):
    if not (t_in <= T < t_out): return
    W = frame.shape[1]
    u = T - t_in; p = np.clip(u / 0.34, 0, 1); q = np.clip((t_out - T) / 0.18, 0, 1)
    sx = 0.04 + 0.96 * ease_out_back(p, 1.8)            # horizontal flip-in
    h, w = spr.shape[:2]
    s2 = cv2.resize(spr, (max(2, int(w * sx)), h), interpolation=cv2.INTER_AREA)
    sc = (1.0 + 0.03 * np.clip(u / 1.6, 0, 1)) * (0.9 + 0.1 * q)
    place(frame, s2 * float(q), W / 2 + (1 - q) * 120, cy - 10 * u, scale=sc, angle=-4 * (1 - ease_out(p)), blur=3 * (1 - p))

# ---------------------------------------------------------------- checklist: emerald panel
def check_badge(sz, prog):
    S = sz * 4
    c = np.zeros((S, S), np.float32); cv2.circle(c, (S // 2, S // 2), int(S * 0.46), 1.0, -1, cv2.LINE_AA)
    ck = np.zeros((S, S), np.float32)
    P = [np.array(p) * S for p in ((0.28, 0.52), (0.44, 0.68), (0.74, 0.34))]
    l1 = np.linalg.norm(P[1] - P[0]); l2 = np.linalg.norm(P[2] - P[1]); L = (l1 + l2) * np.clip(prog, 0, 1)
    th = int(S * 0.10)
    if L > 0:
        e = P[0] + (P[1] - P[0]) * min(1, L / l1); cv2.line(ck, tuple(P[0].astype(int)), tuple(e.astype(int)), 1.0, th, cv2.LINE_AA)
        if L > l1:
            e2 = P[1] + (P[2] - P[1]) * min(1, (L - l1) / l2); cv2.line(ck, tuple(P[1].astype(int)), tuple(e2.astype(int)), 1.0, th, cv2.LINE_AA)
    c = cv2.resize(c, (sz, sz), interpolation=cv2.INTER_AREA); ck = cv2.resize(ck, (sz, sz), interpolation=cv2.INTER_AREA)
    out = solid(c, WH, 1.0); out[..., :3] = out[..., :3] * (1 - ck[..., None]) + EM * ck[..., None]
    return out

def panel_sprite(w, h):
    p = solid(rrect_mask(w, h, 30), EM_D, 0.90)
    top = np.zeros((h, w), np.float32); cv2.line(top, (40, 3), (w - 40, 3), 1.0, 3, cv2.LINE_AA)
    over(p, solid(top, GOLDL, 0.95), 0, 0)
    return shadowed(p, blur=22, off=(0, 12), strength=0.5, pad=40)
