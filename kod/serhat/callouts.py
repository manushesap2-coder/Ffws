"""Tracked map-style callouts for the v3 city shots (customer markings)."""
import numpy as np, cv2, sys, json
sys.path.insert(0, '/tmp/jobs/serhat/lib')
from gfx import composite, place, ease_out, ease_in, ease_io, ease_out_back
from sgfx import text_mask, solid, over, shadowed, rrect_mask, icon_mask, OR, OR_L, CH, WH, FD
from PIL import Image, ImageDraw

LINE_COL = np.array([1.0, 0.66, 0.20], np.float32)
HI_COL = np.array([1.0, 0.93, 0.80], np.float32)

def chaikin(P, closed, it=2):
    P = np.asarray(P, np.float64)
    for _ in range(it):
        Q = []
        n = len(P); rng = range(n) if closed else range(n - 1)
        if not closed: Q.append(P[0])
        for i in rng:
            a, b = P[i], P[(i + 1) % n]
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        if not closed: Q.append(P[-1])
        P = np.array(Q)
    return P

def partial(P, closed, p):
    pts = np.vstack([P, P[:1]]) if closed else np.asarray(P)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = seg.sum(); tgt = p * L
    out = [pts[0]]; acc = 0.0
    for j, sl in enumerate(seg):
        if acc + sl <= tgt: out.append(pts[j + 1]); acc += sl
        else:
            out.append(pts[j] + (pts[j + 1] - pts[j]) * (tgt - acc) / max(sl, 1e-6)); break
    return np.array(out)

def _roi(frame, pts, pad):
    H, W = frame.shape[:2]
    x0, y0 = np.floor(pts.min(0) - pad).astype(int); x1, y1 = np.ceil(pts.max(0) + pad).astype(int)
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

def glow_path(frame, path, alpha=1.0, width=5, head=False, arrows=()):
    """Glowing orange stroke (shadow + glow + core + highlight) along an open polyline in output px."""
    if alpha <= 0 or len(path) < 2: return
    x0, y0, x1, y1 = _roi(frame, path, 60)
    if x1 <= x0 or y1 <= y0: return
    core = np.zeros((y1 - y0, x1 - x0), np.float32)
    S = 4; q = np.round((path - [x0, y0]) * S).astype(np.int32)
    cv2.polylines(core, [q], False, 1.0, width, cv2.LINE_AA, shift=2)
    for tip, d in arrows:                        # (tip point, unit direction)
        n = np.array([-d[1], d[0]]); b = tip - d * 26
        tri = np.round((np.array([tip + d * 2, b + n * 12, b - n * 12]) - [x0, y0]) * S).astype(np.int32)
        cv2.fillConvexPoly(core, tri, 1.0, cv2.LINE_AA, shift=2)
    if head:
        cv2.circle(core, tuple(q[-1]), int(S * 7), 1.0, -1, cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(core, (0, 0), 7) * 1.1 + cv2.GaussianBlur(core, (0, 0), 18) * 0.7
    sh = cv2.GaussianBlur(np.roll(core, 3, 0), (0, 0), 3.5)
    reg = frame[y0:y1, x0:x1]
    reg *= (1 - np.clip(sh * 0.6 * alpha, 0, 1))[..., None]
    reg += (glow * alpha * (0.85 if head else 0.55))[..., None] * OR
    c = np.clip(core * alpha, 0, 1)[..., None]
    reg[:] = np.clip(reg * (1 - c) + LINE_COL * c, 0, 1)
    hi = np.clip(cv2.erode(core, np.ones((3, 3), np.uint8)) * alpha * 0.55, 0, 1)[..., None]
    reg[:] = reg * (1 - hi) + HI_COL * hi

def draw_outline(frame, poly, prog, fill=0.0, alpha=1.0):
    """Self-drawing closed outline; fill (0..1) adds a soft orange wash inside."""
    if prog <= 0 or alpha <= 0: return
    if fill > 0:
        x0, y0, x1, y1 = _roi(frame, poly, 4)
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillPoly(m, [np.round((poly - [x0, y0]) * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
        a = (m * 0.20 * fill * alpha)[..., None]
        reg = frame[y0:y1, x0:x1]; reg[:] = reg * (1 - a) + OR * a
    glow_path(frame, partial(poly, True, prog), alpha, head=prog < 1)

def ring(frame, P, T, t0, alpha, period=1.0, r0=14, r1=64, squash=0.55):
    """Expanding ground rings (perspective-squashed ellipses) + solid centre dot."""
    if T < t0 or alpha <= 0: return
    x0, y0, x1, y1 = _roi(frame, np.array([P]), r1 + 20)
    lay = np.zeros((y1 - y0, x1 - x0), np.float32)
    c = (int(round((P[0] - x0) * 4)), int(round((P[1] - y0) * 4)))
    for k in range(2):
        ph = ((T - t0) / period + 0.5 * k) % 1.0
        r = r0 + (r1 - r0) * ease_out(ph, 2); a = (1 - ph) ** 1.5 * 0.95
        cv2.ellipse(lay, c, (int(r * 4), int(r * squash * 4)), 0, 0, 360, a, 12, cv2.LINE_AA, shift=2)
    cv2.ellipse(lay, c, (40, int(40 * squash)), 0, 0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.0) * alpha
    reg = frame[y0:y1, x0:x1]; reg[:] = reg * (1 - lay[..., None]) + OR * lay[..., None]

# ---------------------------------------------------------------- tags
def school_mask(sz):
    ss = 4; S = sz * ss; im = Image.new('L', (S, S), 0); d = ImageDraw.Draw(im); lw = int(S * 0.075)
    d.polygon([(S * .50, S * .16), (S * .12, S * .40), (S * .88, S * .40)], fill=255)
    d.rectangle([S * .18, S * .44, S * .82, S * .86], fill=255)
    d.rectangle([S * .42, S * .60, S * .58, S * .86], fill=0)
    d.rectangle([S * .24, S * .52, S * .34, S * .62], fill=0); d.rectangle([S * .66, S * .52, S * .76, S * .62], fill=0)
    d.line([(S * .50, S * .16), (S * .50, S * .02)], fill=255, width=lw)
    d.polygon([(S * .50, S * .02), (S * .70, S * .06), (S * .50, S * .11)], fill=255)
    return cv2.resize(np.asarray(im, np.float32) / 255, (sz, sz), interpolation=cv2.INTER_AREA)

def tag(lines, icon, size=38, pointer=None):
    """Charcoal label with an orange icon badge; pointer 'down'/'up' adds a tail. Returns (sprite, tip_xy in sprite px)."""
    ms = [text_mask(t, 'Montserrat-ExtraBold.ttf', size if i == 0 else int(size * 0.84), tracking=1.4) for i, t in enumerate(lines)]
    gap = 6; th = sum(m.shape[0] for m in ms) + gap * (len(ms) - 1)
    bsz = int(max(size * 1.9, th + 16)); pad = 12
    h = max(bsz, th) + 2 * pad; w = bsz + max(m.shape[1] for m in ms) + 3 * pad + 26
    tail = 18 if pointer else 0
    spr = np.zeros((h + tail, w, 4), np.float32)
    y_box = tail if pointer == 'up' else 0
    over(spr, solid(rrect_mask(w, h, min(h // 2, 34)), CH, 0.88), 0, y_box)
    circ = np.zeros((bsz, bsz), np.float32); cv2.circle(circ, (bsz * 2, bsz * 2), bsz * 2 - 4, 1.0, -1, cv2.LINE_AA, shift=2)
    bd = solid(circ, OR, 1.0)
    im = school_mask(int(bsz * 0.62)) if icon == 'school' else icon_mask(icon, int(bsz * 0.62))
    o = (bsz - im.shape[0]) // 2; ic = np.zeros((bsz, bsz), np.float32); ic[o:o + im.shape[0], o:o + im.shape[1]] = im
    over(bd, solid(ic, WH), 0, 0); over(spr, bd, pad, y_box + (h - bsz) // 2)
    y = y_box + (h - th) // 2
    for m in ms:
        over(spr, solid(m, WH), bsz + 2 * pad + 8, y); y += m.shape[0] + gap
    tip = None
    if pointer:
        tri = np.zeros(spr.shape[:2], np.float32); cx = w // 2
        pts = [(cx - 16, h), (cx + 16, h), (cx, h + tail)] if pointer == 'down' else [(cx - 16, tail), (cx + 16, tail), (cx, 0)]
        cv2.fillConvexPoly(tri, np.array(pts, np.int32) * 4, 1.0, cv2.LINE_AA, shift=2)
        over(spr, solid(tri, CH, 0.88), 0, 0)
        tip = (cx, h + tail) if pointer == 'down' else (cx, 0)
    P = 30
    return shadowed(spr, blur=14, off=(0, 8), strength=0.5, pad=P), ((tip[0] + P, tip[1] + P) if tip else None)

def show_tag(frame, spr_tip, at, T, t_in, t_out, center=False, clamp=True):
    """Pop a tag so its tip lands on `at` (or centre it on `at`), scaling from the tip."""
    if T < t_in or T > t_out: return
    spr, tip = spr_tip; h, w = spr.shape[:2]
    p = np.clip((T - t_in) / 0.30, 0, 1); q = np.clip((t_out - T) / 0.18, 0, 1)
    sc = (0.55 + 0.45 * ease_out_back(p, 2.0)) * (0.92 + 0.08 * q)
    if center or tip is None: px, py = w / 2, h / 2
    else: px, py = tip
    cx = at[0] + (w / 2 - px) * sc; cy = at[1] + (h / 2 - py) * sc
    if clamp:
        W, H = frame.shape[1], frame.shape[0]
        cx = float(np.clip(cx, w * sc / 2 - 24, W - w * sc / 2 + 24))
    place(frame, spr, cx, cy, scale=float(sc), opacity=float(min(1, p * 2.2) * q), blur=2.5 * (1 - p))

def street_text(text, size=36):
    m = text_mask(text, 'Montserrat-ExtraBold.ttf', size, tracking=3.0)
    return shadowed(solid(m, WH), blur=5, off=(0, 3), strength=0.85, pad=22)

def show_rotated(frame, spr, center, angle_deg, T, t_in, t_out):
    if T < t_in or T > t_out: return
    p = ease_out(np.clip((T - t_in) / 0.35, 0, 1), 3); q = np.clip((t_out - T) / 0.18, 0, 1)
    h, w = spr.shape[:2]
    # reveal left->right along the text
    xs = np.arange(w, dtype=np.float32)[None, :]
    m = np.clip((p * (w + 40) - 20 - xs) / 20.0, 0, 1)
    s2 = spr * m[..., None]
    place(frame, s2, center[0], center[1], angle=-angle_deg, opacity=float(q))

PIN = None
def pin(frame, P, T, t_in, t_out, size=74):
    global PIN
    if PIN is None:
        PIN = shadowed(solid(icon_mask('pin', size), OR), blur=6, off=(0, 5), strength=0.6, pad=18)
    if T < t_in or T > t_out: return
    p = np.clip((T - t_in) / 0.42, 0, 1); q = np.clip((t_out - T) / 0.18, 0, 1)
    drop = (1 - ease_out_back(p, 2.2)) * 180
    h, w = PIN.shape[:2]
    composite(frame, PIN * float(min(1, p * 2.5) * q), P[0] - w / 2, P[1] - h + 22 - drop)
