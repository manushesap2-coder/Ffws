"""Motion-graphics sprites (premultiplied RGBA float32) rendered with PIL at 2x and composited with cv2."""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

F = '/tmp/jobs/fonts/'
GOLD_HI = np.array([1.00, 0.93, 0.68], np.float32)
GOLD_MID = np.array([0.96, 0.76, 0.33], np.float32)
GOLD_LO = np.array([0.66, 0.45, 0.14], np.float32)
GOLD_TXT = (255, 206, 84)      # caption highlight gold
GREEN = (61, 220, 132)

@lru_cache(None)
def font(name, size):
    return ImageFont.truetype(F + name, size)

def to_premul(rgba_u8):
    a = rgba_u8[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba_u8[..., :3].astype(np.float32) / 255.0
    return np.concatenate([rgb * a, a], -1)

def text_mask(text, fnt, tracking=0, scale=2):
    """Return (mask float32 HxW at 1x, advance width) for text drawn with optional letter spacing."""
    f2 = font(fnt.path.split('/')[-1], fnt.size * scale)
    asc, desc = f2.getmetrics()
    widths = [f2.getlength(ch) for ch in text]
    W = int(sum(widths) + tracking * scale * max(0, len(text) - 1) + 8 * scale)
    Hh = asc + desc + 8 * scale
    im = Image.new('L', (W, Hh), 0)
    d = ImageDraw.Draw(im)
    x = 4 * scale
    for ch, w in zip(text, widths):
        d.text((x, 4 * scale), ch, font=f2, fill=255)
        x += w + tracking * scale
    m = np.asarray(im, np.float32) / 255.0
    m = cv2.resize(m, (W // scale, Hh // scale), interpolation=cv2.INTER_AREA)
    return m

def pad(m, p):
    return cv2.copyMakeBorder(m, p, p, p, p, cv2.BORDER_CONSTANT, value=0)

def composite(dst, spr, x, y):
    """Alpha-over premultiplied sprite onto dst (float RGB) with top-left at (x, y); clipped."""
    H, W = dst.shape[:2]
    h, w = spr.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    ax0, ay0 = max(0, x0), max(0, y0)
    ax1, ay1 = min(W, x0 + w), min(H, y0 + h)
    if ax1 <= ax0 or ay1 <= ay0: return
    s = spr[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
    d = dst[ay0:ay1, ax0:ax1]
    d *= (1 - s[..., 3:4])
    d += s[..., :3]

def transform(spr, scale=1.0, angle=0.0, blur=0.0, opacity=1.0, extra=0):
    """Scale/rotate a premultiplied sprite about its center; returns (sprite, dx, dy) offset of new top-left."""
    h, w = spr.shape[:2]
    if blur > 0.3:
        k = blur
        spr = cv2.GaussianBlur(spr, (0, 0), k)
    if abs(scale - 1) < 1e-3 and abs(angle) < 1e-3:
        out = spr
        dx = dy = 0
    else:
        nw, nh = int(w * scale * 1.0 + 2 * extra + abs(np.sin(np.radians(angle))) * h * scale) + 2, int(h * scale + 2 * extra + abs(np.sin(np.radians(angle))) * w * scale) + 2
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        out = cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dx, dy = (w - nw) / 2, (h - nh) / 2
    if opacity < 1: out = out * opacity
    return out, dx, dy

def place(dst, spr, cx, cy, **kw):
    s, dx, dy = transform(spr, **kw)
    h, w = spr.shape[:2]
    composite(dst, s, cx - w / 2 + dx, cy - h / 2 + dy)

# ---------------------------------------------------------------- captions
def caption_word(text, color=(255, 255, 255), size=66, fnt='Poppins-Bold.ttf', glow=None):
    """Caption word sprite: fill + soft dark shadow + thin dark outline; optional colored glow."""
    m = pad(text_mask(text, font(fnt, size)), 30)
    fill = np.zeros(m.shape + (4,), np.float32)
    col = np.array(color, np.float32) / 255.0
    # outline (dilate) and drop shadow
    ol = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    sh = cv2.GaussianBlur(np.roll(np.roll(ol, 4, 0), 2, 1), (0, 0), 7) * 0.75
    base = np.zeros(m.shape + (4,), np.float32)
    a_sh = np.clip(sh, 0, 1)
    base[..., 3] = a_sh
    if glow is not None:
        g = cv2.GaussianBlur(m, (0, 0), 10) * 0.9
        gc = np.array(glow, np.float32) / 255.0
        a = g
        base[..., :3] = base[..., :3] * (1 - a[..., None]) + gc * a[..., None]
        base[..., 3] = base[..., 3] * (1 - a) + a
    a_ol = np.clip(ol * 0.55, 0, 1)
    base[..., :3] *= (1 - a_ol[..., None]); base[..., 3] = base[..., 3] * (1 - a_ol) + a_ol
    base[..., :3] = base[..., :3] * (1 - m[..., None]) + col * m[..., None]
    base[..., 3] = base[..., 3] * (1 - m) + m
    return base

# ---------------------------------------------------------------- 3D gold title
def gold_face(m, hi=GOLD_HI, mid=GOLD_MID, lo=GOLD_LO):
    """Metallic gradient fill with a bevel highlight derived from the distance field."""
    h, w = m.shape
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    # two-band metallic ramp: bright top, darker band, bright lower rim
    t = np.clip(yy, 0, 1)[..., None]                     # (h,1,1)
    ramp = np.where(t < 0.5, hi + (mid - hi) * (t / 0.5), mid + (lo - mid) * ((t - 0.5) / 0.5))
    ramp = np.broadcast_to(ramp, (h, w, 3)).copy()
    dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    bevel = np.clip(dist / 4.0, 0, 1)
    gx = cv2.Sobel(cv2.GaussianBlur(bevel, (0, 0), 1.5), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cv2.GaussianBlur(bevel, (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=3)
    light = np.clip(-(gx * -0.5 + gy * -0.8), -1, 1)   # light from top-left
    col = ramp * (1 + 0.35 * light[..., None])
    col = col + 0.25 * np.clip(light, 0, 1)[..., None]
    return np.clip(col, 0, 1)

def title_3d(lines, sizes, fnts, tracking, depth=14, line_gap=0.16, face='gold', glow=(255, 170, 40)):
    """Stacked multi-line 3D title sprite (premultiplied). face: 'gold' or 'white'."""
    masks = []
    for t, s, f, tr in zip(lines, sizes, fnts, tracking):
        mm = text_mask(t, font(f, s), tr)
        ys = np.nonzero(mm.max(1) > 0.02)[0]
        masks.append(mm[ys[0]:ys[-1] + 1])            # trim to ink rows
    W = max(m.shape[1] for m in masks)
    rows = []
    for m in masks:
        p = (W - m.shape[1]) // 2
        rows.append(cv2.copyMakeBorder(m, 0, 0, p, W - m.shape[1] - p, cv2.BORDER_CONSTANT, value=0))
    gap = max(2, int(line_gap * rows[0].shape[0]))
    m = rows[0]
    for r in rows[1:]:
        m = np.vstack([m, np.zeros((gap, W), np.float32), r])
    P = 60 + depth
    m = pad(m, P)
    h, w = m.shape
    out = np.zeros((h, w, 4), np.float32)
    # glow + shadow
    g = cv2.GaussianBlur(m, (0, 0), 22)
    gc = np.array(glow, np.float32) / 255.0
    out[..., :3] = gc * g[..., None] * 0.55; out[..., 3] = g * 0.55
    sh = cv2.GaussianBlur(np.roll(np.roll(m, depth + 10, 0), depth // 2 + 6, 1), (0, 0), 12) * 0.7
    a = sh; out[..., :3] *= (1 - a[..., None]); out[..., 3] = out[..., 3] * (1 - a) + a
    # extrusion: darker layers offset down-right
    for i in range(depth, 0, -1):
        layer = np.roll(np.roll(m, i, 0), i // 2, 1)
        k = i / depth
        c = (np.array([0.36, 0.22, 0.05]) * (1 - 0.35 * k)) if face == 'gold' else (np.array([0.55, 0.57, 0.62]) * (1 - 0.4 * k))
        a = layer
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + c * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a
    # thin dark rim then face
    rim = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    a = rim; c = np.array([0.18, 0.10, 0.02]) if face == 'gold' else np.array([0.15, 0.16, 0.2])
    out[..., :3] = out[..., :3] * (1 - a[..., None]) + c * a[..., None]; out[..., 3] = out[..., 3] * (1 - a) + a
    if face == 'gold':
        col = gold_face(m)
    else:
        col = gold_face(m, hi=np.array([1, 1, 1], np.float32), mid=np.array([0.93, 0.94, 0.97], np.float32), lo=np.array([0.72, 0.75, 0.82], np.float32))
    out[..., :3] = out[..., :3] * (1 - m[..., None]) + col * m[..., None]
    out[..., 3] = out[..., 3] * (1 - m) + m
    return out, m

def shine(spr, face_mask, pos, width=0.12, strength=0.8, angle=20):
    """Diagonal specular sweep over the title face; pos in [-0.3, 1.3]."""
    h, w = face_mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + yy * np.tan(np.radians(angle))) / (w + h * np.tan(np.radians(angle)))
    band = np.exp(-((u - pos) / width) ** 2) * face_mask * strength
    out = spr.copy()
    out[..., :3] = np.clip(out[..., :3] + band[..., None] * np.array([1.0, 0.97, 0.85], np.float32), 0, None)
    return out

def chroma_split(spr, px):
    """Chromatic aberration: shift R left and B right by px (premultiplied sprite)."""
    if px < 0.5: return spr
    out = spr.copy()
    M1 = np.float32([[1, 0, -px], [0, 1, 0]]); M2 = np.float32([[1, 0, px], [0, 1, 0]])
    h, w = spr.shape[:2]
    out[..., 0] = cv2.warpAffine(spr[..., 0], M1, (w, h))
    out[..., 2] = cv2.warpAffine(spr[..., 2], M2, (w, h))
    out[..., 3] = np.maximum(spr[..., 3], np.maximum(cv2.warpAffine(spr[..., 3], M1, (w, h)), cv2.warpAffine(spr[..., 3], M2, (w, h))))
    return out

# ---------------------------------------------------------------- picture card (PIP)
def card(img_rgb, width, border=10, radius=26, shadow=28):
    """White-bordered rounded card with drop shadow from an RGB float image."""
    h0, w0 = img_rgb.shape[:2]
    iw = width - 2 * border
    ih = int(h0 * iw / w0)
    im = cv2.resize(img_rgb, (iw, ih), interpolation=cv2.INTER_AREA)
    H, W = ih + 2 * border, width
    P = shadow * 2
    out = np.zeros((H + 2 * P, W + 2 * P, 4), np.float32)
    rr = np.zeros((H + 2 * P, W + 2 * P), np.float32)
    cv2.rectangle(rr, (P + radius, P), (P + W - radius, P + H), 1, -1)
    cv2.rectangle(rr, (P, P + radius), (P + W, P + H - radius), 1, -1)
    for cx, cy in [(P + radius, P + radius), (P + W - radius, P + radius), (P + radius, P + H - radius), (P + W - radius, P + H - radius)]:
        cv2.circle(rr, (cx, cy), radius, 1, -1, lineType=cv2.LINE_AA)
    rr = cv2.GaussianBlur(rr, (0, 0), 0.8)
    sh = cv2.GaussianBlur(np.roll(rr, 18, 0), (0, 0), shadow) * 0.6
    out[..., 3] = sh
    out[..., :3] = out[..., :3] * (1 - rr[..., None]) + 1.0 * rr[..., None]
    out[..., 3] = out[..., 3] * (1 - rr) + rr
    ir = np.zeros_like(rr)
    r2 = max(4, radius - border)
    x0, y0, x1, y1 = P + border, P + border, P + border + iw, P + border + ih
    cv2.rectangle(ir, (x0 + r2, y0), (x1 - r2, y1), 1, -1); cv2.rectangle(ir, (x0, y0 + r2), (x1, y1 - r2), 1, -1)
    for cx, cy in [(x0 + r2, y0 + r2), (x1 - r2, y0 + r2), (x0 + r2, y1 - r2), (x1 - r2, y1 - r2)]:
        cv2.circle(ir, (cx, cy), r2, 1, -1, lineType=cv2.LINE_AA)
    ir = cv2.GaussianBlur(ir, (0, 0), 0.7)
    full = np.zeros((H + 2 * P, W + 2 * P, 3), np.float32); full[y0:y1, x0:x1] = im
    out[..., :3] = out[..., :3] * (1 - ir[..., None]) + full * ir[..., None]
    return out

# ---------------------------------------------------------------- check icon
def check_icon(size, progress, color=GREEN):
    """Filled circle with a checkmark drawn progressively (progress 0..1)."""
    S = size * 2
    im = np.zeros((S, S), np.float32)
    cv2.circle(im, (S // 2, S // 2), int(S * 0.46), 1.0, -1, lineType=cv2.LINE_AA)
    ck = np.zeros((S, S), np.float32)
    p1, p2, p3 = (0.28, 0.52), (0.44, 0.68), (0.74, 0.34)
    pts = [np.array(p) * S for p in (p1, p2, p3)]
    l1 = np.linalg.norm(pts[1] - pts[0]); l2 = np.linalg.norm(pts[2] - pts[1])
    L = (l1 + l2) * np.clip(progress, 0, 1)
    th = max(2, int(S * 0.09))
    if L > 0:
        e = pts[0] + (pts[1] - pts[0]) * min(1, L / l1)
        cv2.line(ck, tuple(pts[0].astype(int)), tuple(e.astype(int)), 1.0, th, lineType=cv2.LINE_AA)
        if L > l1:
            e2 = pts[1] + (pts[2] - pts[1]) * min(1, (L - l1) / l2)
            cv2.line(ck, tuple(pts[1].astype(int)), tuple(e2.astype(int)), 1.0, th, lineType=cv2.LINE_AA)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    ck = cv2.resize(ck, (size, size), interpolation=cv2.INTER_AREA)
    col = np.array(color, np.float32) / 255
    out = np.zeros((size, size, 4), np.float32)
    out[..., :3] = col * im[..., None]; out[..., 3] = im
    out[..., :3] = out[..., :3] * (1 - ck[..., None]) + ck[..., None]
    return out

def ease_out_back(t, s=1.70158):
    t = np.clip(t, 0, 1) - 1
    return t * t * ((s + 1) * t + s) + 1

def ease_out(t, p=3):
    return 1 - (1 - np.clip(t, 0, 1)) ** p

def ease_in(t, p=3):
    return np.clip(t, 0, 1) ** p

def ease_io(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

# ---------------------------------------------------------------- brand end card
def anchor_sprite(height=560):
    """Gold anchor logo (from multi-frame fused hoarding capture) with bevel shading."""
    med = np.load('/tmp/edit/work/anchor_med.npy')
    a = np.clip((cv2.GaussianBlur(med, (0, 0), 1.0) - 0.16) / 0.30, 0, 1)
    ys, xs = np.nonzero(a > 0.05)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = height / a.shape[0]
    a = cv2.resize(a, (int(a.shape[1] * s), height), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    a = np.clip((a - 0.5) * 1.6 + 0.5, 0, 1)              # crisper edges after resize
    a = pad(a, 40)
    col = gold_face(a, hi=np.array([1.0, 0.87, 0.52], np.float32), mid=np.array([0.93, 0.70, 0.27], np.float32),
                    lo=np.array([0.60, 0.38, 0.09], np.float32))
    h, w = a.shape
    out = np.zeros((h, w, 4), np.float32)
    g = cv2.GaussianBlur(a, (0, 0), 16) * 0.35
    out[..., :3] = np.array([1.0, 0.72, 0.25], np.float32) * g[..., None]; out[..., 3] = g
    out[..., :3] = out[..., :3] * (1 - a[..., None]) + col * a[..., None]
    out[..., 3] = out[..., 3] * (1 - a) + a
    return out, a

def flat_text(text, size, fnt, tracking=0, color=(1, 1, 1), gold=False, shadow=0.5):
    m = pad(text_mask(text, font(fnt, size), tracking), 24)
    h, w = m.shape
    out = np.zeros((h, w, 4), np.float32)
    if shadow > 0:
        sh = cv2.GaussianBlur(np.roll(m, 3, 0), (0, 0), 6) * shadow
        out[..., 3] = sh
    col = gold_face(m) if gold else np.broadcast_to(np.array(color, np.float32), (h, w, 3))
    out[..., :3] = out[..., :3] * (1 - m[..., None]) + col * m[..., None]
    out[..., 3] = out[..., 3] * (1 - m) + m
    return out, m

def brand_background(w=1080, h=1920):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w * 0.5) / w) ** 2 + ((yy - h * 0.40) / (h * 0.75)) ** 2)
    c0 = np.array([0.050, 0.205, 0.150], np.float32); c1 = np.array([0.008, 0.050, 0.036], np.float32)
    t = np.clip(r / 0.75, 0, 1)[..., None] ** 1.2
    bg = c0 * (1 - t) + c1 * t
    rng = np.random.default_rng(7)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h // 4, w // 4)).astype(np.float32), (0, 0), 1.2)
    bg = bg * (1 + 0.04 * cv2.resize(noise, (w, h))[..., None])
    return np.clip(bg, 0, 1)
