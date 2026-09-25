"""Serhat Insaat motion graphics: premultiplied RGBA float32 sprites, drawn at 2x with PIL/cv2."""
import numpy as np, cv2, sys
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
sys.path.insert(0, '/tmp/edit/pipeline')
from gfx import composite, transform, place, ease_out, ease_in, ease_io, ease_out_back

FD = '/tmp/jobs/fonts/'
OR = np.array([0.98, 0.58, 0.12], np.float32)      # logo orange
OR_L = np.array([1.00, 0.76, 0.36], np.float32)    # light orange (gradient top)
CH = np.array([0.075, 0.078, 0.086], np.float32)   # charcoal
WH = np.array([1.0, 1.0, 1.0], np.float32)
SIL = np.array([0.84, 0.85, 0.87], np.float32)

@lru_cache(None)
def fnt(name, size, wght=None):
    f = ImageFont.truetype(FD + name, size)
    if wght is not None:
        try: f.set_variation_by_axes([wght])
        except Exception: pass
    return f

def text_mask(text, name, size, tracking=0.0, wght=None, ss=2):
    """Anti-aliased text coverage mask at 1x (drawn at ss x). tracking in px at 1x."""
    f = fnt(name, size * ss, wght)
    asc, desc = f.getmetrics()
    widths = [f.getlength(c) for c in text]
    Wd = int(sum(widths) + tracking * ss * max(0, len(text) - 1) + 10 * ss)
    Hd = asc + desc + 10 * ss
    im = Image.new('L', (Wd, Hd), 0); d = ImageDraw.Draw(im)
    x = 5 * ss
    for c, w in zip(text, widths):
        d.text((x, 5 * ss), c, font=f, fill=255); x += w + tracking * ss
    m = np.asarray(im, np.float32) / 255.0
    m = cv2.resize(m, (Wd // ss, Hd // ss), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(m > 0.02)
    return m[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]

def solid(mask, color, alpha=1.0):
    a = np.clip(mask, 0, 1)[..., None] * alpha
    return np.concatenate([np.asarray(color, np.float32) * a, a], -1).astype(np.float32)

def padspr(s, p):
    return cv2.copyMakeBorder(s, p, p, p, p, cv2.BORDER_CONSTANT, value=0)

def over(dst, src, x, y):
    """Composite premultiplied sprite src onto premultiplied sprite dst at (x, y)."""
    h, w = src.shape[:2]
    d = dst[y:y + h, x:x + w]
    d[:] = src + d * (1 - src[..., 3:4])

def shadowed(spr, blur=10, off=(0, 6), strength=0.55, pad=26):
    s = padspr(spr, pad)
    a = s[..., 3]
    sh = cv2.GaussianBlur(np.roll(np.roll(a, off[1], 0), off[0], 1), (0, 0), blur) * strength
    base = np.zeros_like(s); base[..., 3] = sh
    return s + base * (1 - s[..., 3:4])

def rrect_mask(w, h, r, ss=3):
    im = Image.new('L', (w * ss, h * ss), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, fill=255)
    return cv2.resize(np.asarray(im, np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)

def text_spr(text, name, size, color=WH, tracking=0, wght=None, shadow=True):
    m = text_mask(text, name, size, tracking, wght)
    s = solid(m, color)
    return shadowed(s, blur=8, off=(0, 4), strength=0.45, pad=20) if shadow else s

def grad_text_spr(text, name, size, top=OR_L, bot=OR, tracking=0, wght=None, shadow=True):
    m = text_mask(text, name, size, tracking, wght)
    h = m.shape[0]
    g = (np.linspace(0, 1, h, dtype=np.float32)[:, None, None])
    col = top * (1 - g) + bot * g
    s = np.concatenate([col * m[..., None], m[..., None]], -1).astype(np.float32)
    return shadowed(s, blur=9, off=(0, 5), strength=0.5, pad=20) if shadow else s

# ---------------------------------------------------------------- measurement tag
def tag_spr(caption, value):
    """Two-row label: small charcoal strip with white spaced caption, big orange block with white value."""
    cm = text_mask(caption, 'Montserrat-Bold.ttf', 25, tracking=3.2)
    vm = text_mask(value, 'Montserrat-ExtraBold.ttf', 70, tracking=0.5)
    w = int(max(cm.shape[1] + 36, vm.shape[1] + 52))
    h1, h2 = cm.shape[0] + 20, vm.shape[0] + 30
    spr = np.zeros((h1 + h2, w, 4), np.float32)
    top = solid(rrect_mask(w, h1 + 12, 10), CH, 0.92); over(spr, top[:h1], 0, 0)
    bot = solid(rrect_mask(w, h2, 12), OR, 1.0)
    # subtle vertical gradient on the orange block
    g = np.linspace(1.06, 0.92, h2, dtype=np.float32)[:, None, None]
    bot[..., :3] = np.clip(bot[..., :3] * g, 0, 1) * (bot[..., 3:4] > 0)
    over(spr, bot, 0, h1)
    over(spr, solid(cm, WH), (w - cm.shape[1]) // 2, (h1 - cm.shape[0]) // 2 + 1)
    over(spr, solid(vm, WH), (w - vm.shape[1]) // 2, h1 + (h2 - vm.shape[0]) // 2)
    return shadowed(spr, blur=14, off=(0, 8), strength=0.5, pad=30)

# ---------------------------------------------------------------- dimension line
def draw_dim(frame, P0, P1, prog, alpha=1.0, mode='center', tick=26, head=30, width=6.0):
    """Architectural dimension line P0->P1 in output px. prog 0..1 grows the line
    ('center': outward from middle, 'fwd': from P0 to P1). White core, orange glow, arrow heads, end ticks."""
    if prog <= 0 or alpha <= 0: return
    P0, P1 = np.asarray(P0, np.float64), np.asarray(P1, np.float64)
    d = P1 - P0; L = np.linalg.norm(d)
    if L < 4: return
    u = d / L; n = np.array([-u[1], u[0]])
    if mode == 'center':
        m = (P0 + P1) / 2; A = m - (m - P0) * prog; B = m + (P1 - m) * prog
        heads = [(A, -u), (B, u)]
    else:
        A = P0; B = P0 + d * prog
        heads = [(A, -u), (B, u)]
    x0, y0 = np.floor(np.minimum(np.minimum(A, B), np.minimum(P0, P1)) - 80).astype(int)
    x1, y1 = np.ceil(np.maximum(np.maximum(A, B), np.maximum(P0, P1)) + 80).astype(int)
    Hh, Ww = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(Ww, x1), min(Hh, y1)
    if x1 <= x0 or y1 <= y0: return
    S = 4
    core = np.zeros(((y1 - y0), (x1 - x0)), np.float32)
    def pt(P): return (int(round((P[0] - x0) * S)), int(round((P[1] - y0) * S)))
    th = max(1, int(round(width)))
    cv2.line(core, pt(A), pt(B), 1.0, th, cv2.LINE_AA, shift=2)
    for P, dirv in heads:
        tip = P + dirv * 2.0; base = P - dirv * head; wv = n * head * 0.42
        tri = np.array([pt(tip), pt(base + wv), pt(base - wv)], np.int32)
        cv2.fillConvexPoly(core, tri, 1.0, cv2.LINE_AA, shift=2)
    for P in (P0, P1):   # end ticks appear once the line reaches them
        reach = (np.linalg.norm(A - P) < 3) or (np.linalg.norm(B - P) < 3)
        if reach:
            cv2.line(core, pt(P - n * tick), pt(P + n * tick), 1.0, max(1, th - 1), cv2.LINE_AA, shift=2)
    glow = cv2.GaussianBlur(core, (0, 0), 7) * 1.1 + cv2.GaussianBlur(core, (0, 0), 18) * 0.7
    sh = cv2.GaussianBlur(np.roll(core, 3, 0), (0, 0), 3.5)
    reg = frame[y0:y1, x0:x1]
    reg *= (1 - np.clip(sh * 0.6 * alpha, 0, 1))[..., None]
    reg += (glow * alpha * 0.55)[..., None] * OR
    c = np.clip(core * alpha, 0, 1)[..., None]
    col = np.array([1.0, 0.66, 0.20], np.float32)
    reg[:] = np.clip(reg * (1 - c) + col * c, 0, 1)
    hi = np.clip(cv2.erode(core, np.ones((3, 3), np.uint8)) * alpha * 0.55, 0, 1)[..., None]
    reg[:] = reg * (1 - hi) + np.array([1.0, 0.93, 0.80], np.float32) * hi

# ---------------------------------------------------------------- captions (speaker)
CAP_FONT = 'Montserrat-ExtraBold.ttf'; CAP_SIZE = 62
@lru_cache(None)
def cap_word_mask(w):
    return text_mask(w, CAP_FONT, CAP_SIZE, tracking=0.5)

def layout_line(words):
    ms = [cap_word_mask(w) for w in words]
    gap = 18
    tw = sum(m.shape[1] for m in ms) + gap * (len(ms) - 1)
    hh = max(m.shape[0] for m in ms)
    xs = []; x = -tw / 2
    for m in ms:
        xs.append(x); x += m.shape[1] + gap
    return ms, xs, tw, hh

def draw_caption(frame, T, chunk, cy=1585):
    """chunk: list of dicts w, T, Te. Words pop in when spoken; active word sits on an orange box that slides."""
    words = [c['w'] for c in chunk]
    ms, xs, tw, hh = layout_line(words)
    t_first, t_last = chunk[0]['T'], chunk[-1]['Te']
    cx = frame.shape[1] / 2
    # active index + box interpolation
    times = [c['T'] for c in chunk]
    act = max([i for i, t in enumerate(times) if T >= t - 0.02] or [0])
    def box(i):
        m = ms[i]; return cx + xs[i] - 14, cy - hh / 2 - 8, m.shape[1] + 28, hh + 16
    bx, by, bw, bh = box(act)
    if act > 0:
        k = np.clip((T - times[act] + 0.02) / 0.12, 0, 1); k = ease_out(k, 3)
        px, py, pw, ph = box(act - 1)
        bx, bw = px + (bx - px) * k, pw + (bw - pw) * k
    fade = np.clip((t_last + 0.35 - T) / 0.18, 0, 1) if T > t_last else 1.0
    if T >= times[0] - 0.02:
        kin = ease_out(np.clip((T - times[0] + 0.02) / 0.10, 0, 1), 2)
        bm = rrect_mask(int(bw), int(bh), 14)
        spr = solid(bm, OR, 0.96 * kin * fade)
        spr = shadowed(spr, blur=10, off=(0, 6), strength=0.35, pad=20)
        composite(frame, spr, bx - 20, by - 20)
    for i, (m, x) in enumerate(zip(ms, xs)):
        t = chunk[i]['T']
        if T < t - 0.04: continue
        p = np.clip((T - t + 0.04) / 0.14, 0, 1)
        sc = 0.72 + 0.28 * ease_out_back(p, 2.2)
        spr = shadowed(solid(m, WH, 1.0), blur=7, off=(0, 4), strength=0.6, pad=18)
        place(frame, spr, cx + x + m.shape[1] / 2, cy + 4 * (1 - p), scale=sc, opacity=min(1.0, p * 1.6) * fade)

# ---------------------------------------------------------------- icons (drawn at 4x then reduced)
def _canvas(sz, ss=4):
    im = Image.new('L', (sz * ss, sz * ss), 0); return im, ImageDraw.Draw(im), ss

def _fin(im, sz):
    return cv2.resize(np.asarray(im, np.float32) / 255, (sz, sz), interpolation=cv2.INTER_AREA)

def icon_mask(kind, sz=64):
    im, d, s = _canvas(sz); S = sz * s; lw = int(S * 0.075)
    if kind == 'check':
        d.line([(S*0.24, S*0.52), (S*0.43, S*0.70), (S*0.77, S*0.33)], fill=255, width=int(S*0.11), joint='curve')
    elif kind == 'area':
        m = S * 0.16
        d.rectangle([m, m, S - m, S - m], outline=255, width=lw)
        for (ax, ay, bx, by) in [(0.30, 0.30, 0.44, 0.44), (0.70, 0.30, 0.56, 0.44), (0.30, 0.70, 0.44, 0.56), (0.70, 0.70, 0.56, 0.56)]:
            d.line([(S*ax, S*ay), (S*bx, S*by)], fill=255, width=lw)
    elif kind == 'height':
        d.line([(S*0.18, S*0.14), (S*0.82, S*0.14)], fill=255, width=lw)
        d.line([(S*0.18, S*0.86), (S*0.82, S*0.86)], fill=255, width=lw)
        d.line([(S*0.5, S*0.24), (S*0.5, S*0.76)], fill=255, width=lw)
        d.polygon([(S*0.5, S*0.20), (S*0.38, S*0.36), (S*0.62, S*0.36)], fill=255)
        d.polygon([(S*0.5, S*0.80), (S*0.38, S*0.64), (S*0.62, S*0.64)], fill=255)
    elif kind == 'stairs':
        pts = [(0.14, 0.86), (0.14, 0.70), (0.34, 0.70), (0.34, 0.52), (0.54, 0.52), (0.54, 0.34), (0.74, 0.34), (0.74, 0.16), (0.88, 0.16)]
        d.line([(S*x, S*y) for x, y in pts], fill=255, width=lw, joint='curve')
        d.line([(S*0.14, S*0.86), (S*0.88, S*0.86)], fill=255, width=lw)
    elif kind == 'walk':
        d.ellipse([S*0.44, S*0.08, S*0.60, S*0.24], fill=255)
        d.line([(S*0.50, S*0.30), (S*0.44, S*0.56)], fill=255, width=int(S*0.11))
        d.line([(S*0.44, S*0.56), (S*0.30, S*0.88)], fill=255, width=lw)
        d.line([(S*0.44, S*0.56), (S*0.60, S*0.72), (S*0.62, S*0.90)], fill=255, width=lw, joint='curve')
        d.line([(S*0.49, S*0.34), (S*0.32, S*0.50)], fill=255, width=lw)
        d.line([(S*0.49, S*0.34), (S*0.64, S*0.48)], fill=255, width=lw)
    elif kind == 'car':
        d.rounded_rectangle([S*0.10, S*0.44, S*0.90, S*0.72], radius=S*0.08, fill=255)
        d.polygon([(S*0.24, S*0.46), (S*0.33, S*0.26), (S*0.67, S*0.26), (S*0.78, S*0.46)], fill=255)
        d.polygon([(S*0.32, S*0.44), (S*0.38, S*0.31), (S*0.49, S*0.31), (S*0.49, S*0.44)], fill=0)
        d.polygon([(S*0.53, S*0.44), (S*0.53, S*0.31), (S*0.63, S*0.31), (S*0.70, S*0.44)], fill=0)
        for cx in (0.28, 0.72):
            d.ellipse([S*(cx-0.10), S*0.62, S*(cx+0.10), S*0.82], fill=255)
            d.ellipse([S*(cx-0.045), S*0.675, S*(cx+0.045), S*0.765], fill=0)
    elif kind == 'parking':
        d.rounded_rectangle([S*0.12, S*0.12, S*0.88, S*0.88], radius=S*0.16, outline=255, width=lw)
        f = ImageFont.truetype(FD + 'Montserrat-ExtraBold.ttf', int(S * 0.56))
        bb = d.textbbox((0, 0), 'P', font=f); tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((S - tw) / 2 - bb[0], (S - th) / 2 - bb[1]), 'P', font=f, fill=255)
    elif kind == 'tree':
        d.ellipse([S*0.22, S*0.10, S*0.78, S*0.62], fill=255)
        d.rectangle([S*0.45, S*0.55, S*0.55, S*0.88], fill=255)
        d.line([(S*0.18, S*0.88), (S*0.82, S*0.88)], fill=255, width=lw)
    elif kind == 'gov':
        d.polygon([(S*0.50, S*0.10), (S*0.12, S*0.32), (S*0.88, S*0.32)], fill=255)
        for x in (0.22, 0.38, 0.54, 0.70):
            d.rectangle([S*x, S*0.40, S*(x+0.08), S*0.76], fill=255)
        d.rectangle([S*0.12, S*0.80, S*0.88, S*0.88], fill=255)
    elif kind == 'pin':
        d.ellipse([S*0.18, S*0.06, S*0.82, S*0.70], fill=255)
        d.polygon([(S*0.24, S*0.50), (S*0.76, S*0.50), (S*0.50, S*0.96)], fill=255)
        d.ellipse([S*0.37, S*0.25, S*0.63, S*0.51], fill=0)
    elif kind == 'star':
        import math
        pts = []
        for i in range(10):
            r = 0.42 if i % 2 == 0 else 0.18; a = -math.pi / 2 + i * math.pi / 5
            pts.append((S*(0.5 + r*math.cos(a)), S*(0.52 + r*math.sin(a))))
        d.polygon(pts, fill=255)
    elif kind == 'trend':
        d.line([(S*0.12, S*0.78), (S*0.38, S*0.52), (S*0.56, S*0.64), (S*0.86, S*0.30)], fill=255, width=lw, joint='curve')
        d.polygon([(S*0.90, S*0.22), (S*0.70, S*0.28), (S*0.86, S*0.44)], fill=255)
    return _fin(im, sz)

def badge_spr(kind, sz=86, ring=OR, fill=OR, icon_col=WH):
    """Orange circular badge with white icon."""
    circ = np.zeros((sz, sz), np.float32)
    cv2.circle(circ, (sz * 2, sz * 2), sz * 2 - 4, 1.0, -1, cv2.LINE_AA, shift=2)
    spr = solid(circ, fill, 1.0)
    im = icon_mask(kind, int(sz * 0.6))
    ic = np.zeros((sz, sz), np.float32); o = (sz - im.shape[0]) // 2
    ic[o:o + im.shape[0], o:o + im.shape[1]] = im
    over(spr, solid(ic, icon_col), 0, 0)
    return spr

def chip_spr(kind, text, size=44, dark=0.80):
    """Icon badge + uppercase label on a dark rounded bar."""
    tm = text_mask(text, 'Montserrat-ExtraBold.ttf', size, tracking=1.6)
    bsz = int(tm.shape[0] * 1.9)
    b = badge_spr(kind, bsz)
    h = bsz + 20; w = bsz + tm.shape[1] + 64
    spr = np.zeros((h, w, 4), np.float32)
    over(spr, solid(rrect_mask(w, h, h // 2), CH, dark), 0, 0)
    over(spr, b, 10, 10)
    over(spr, solid(tm, WH), bsz + 34, (h - tm.shape[0]) // 2)
    return shadowed(spr, blur=16, off=(0, 8), strength=0.45, pad=30)
