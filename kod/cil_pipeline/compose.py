"""Main compositor: shots -> grade -> framing -> transitions -> motion graphics -> H.264."""
import sys, os, json, math, time, subprocess, argparse
import numpy as np, cv2
sys.path.insert(0, '/tmp/edit/pipeline')
from config import *
from core import Reader, grade, auto_levels, luma
from gfx import *
import timeline as TL

FPS_SRC = 60000 / 1001
BEAT = TL.BEAT
def B(k): return TL.T_DROP1 + k * BEAT
T_END = TL.TOTAL

G_SPK = dict(contrast=1.32, sat=1.45, vib=0.4, sky=0.55, sky_extent=0.30, hl_rolloff=0.6,
             gain=[1.035, 1.0, 0.965], lift=[0.0, 0.004, 0.012], clarity=0.20, sharpen=0.45)
G_DRN = dict(contrast=1.38, sat=1.62, vib=0.45, sky=0.0, hl_rolloff=1.6,
             gain=[1.025, 1.0, 0.975], lift=[0.0, 0.004, 0.012], clarity=0.30, sharpen=0.32)
def drn(**kw):
    d = dict(G_DRN); d.update(kw); return d

# src: intermediate; mode 'spk' (lip-synced via voice map) | 'broll' | 'card'
# broll timing: src_in (s into intermediate) + speed or ramp [(t_rel, speed)...]; framing z0->z1 (1.0 = full intermediate)
SHOTS_T = [
    dict(n='A1', src='SPK_A', mode='spk', t0=0.0, t1=3.080, z0=1.92, z1=2.02, grade=G_SPK, tin=('pushin', 7)),
    dict(n='A2', src='SPK_A', mode='spk', t0=3.080, t1=B(0), z0=2.42, z1=2.56, grade=G_SPK, tin=('punch', 5)),
    dict(n='B1', src='B3', mode='broll', t0=B(0), t1=B(2), src_in=0.30, speed=1.0, z0=1.08, z1=1.15, grade=drn(sky=0.8, contrast=1.30, lift=[0.022, 0.024, 0.034]), tin=('flashzoom', 7), shake=(0.0, 12, 16)),
    dict(n='B2', src='B2', mode='broll', t0=B(2), t1=B(4), src_in=0.40, speed=1.15, z0=1.08, z1=1.14, grade=drn(sky=0.5), tin=('whip', 4, (-1, 0))),
    dict(n='B3', src='B1', mode='broll', t0=B(4), t1=B(5), src_in=2.20, speed=1.0, z0=1.10, z1=1.14, grade=drn(sky=0.7), tin=('zoom', 3)),
    dict(n='C1', src='C1', mode='broll', t0=B(5), t1=B(7), src_in=0.30, speed=1.0, z0=1.06, z1=1.12, grade=drn(sky=0.5), tin=('whip', 4, (0, -1))),
    dict(n='C2', src='C2', mode='broll', t0=B(7), t1=B(9), src_in=0.30, speed=1.0, z0=1.08, z1=1.13, grade=drn(sky=0.8), tin=('punch', 4)),
    dict(n='C3', src='C3', mode='broll', t0=B(9), t1=B(11), src_in=0.30, speed=1.0, z0=1.06, z1=1.12, grade=drn(sky=0.5), tin=('whip', 4, (1, 0))),
    dict(n='C4', src='C4', mode='broll', t0=B(11), t1=B(14), src_in=0.20, speed=1.0, z0=1.04, z1=1.12, grade=drn(), tin=('zoom', 3)),
    dict(n='D1', src='D1', mode='broll', t0=B(14), t1=B(17), src_in=0.90, speed=1.0, z0=1.04, z1=1.10, grade=drn(), tin=('whip', 4, (-1, 0))),
    dict(n='D2', src='D2', mode='broll', t0=B(17), t1=B(21), src_in=0.50, speed=1.5, z0=1.10, z1=1.18, grade=drn(), tin=('spin', 5, 1), outline=True),
    dict(n='E1', src='SPK_E', mode='spk', t0=B(21), t1=B(23), z0=2.12, z1=2.24, grade=G_SPK, tin=('whip', 4, (1, 0))),
    dict(n='E2', src='E2', mode='broll', t0=B(23), t1=B(28), src_in=4.0, ramp=[(0.0, 3.2), (0.55, 3.0), (1.05, 0.9), (2.42, 0.55)], z0=1.04, z1=1.16, grade=drn(), tin=('zoomout', 5)),
    dict(n='F1', src='F1', mode='broll', t0=B(28), t1=B(29), src_in=0.50, speed=1.5, z0=1.10, z1=1.14, grade=drn(), tin=('flashzoom', 6), shake=(0.0, 12, 14)),
    dict(n='F2', src='F2', mode='broll', t0=B(29), t1=B(30), src_in=0.30, speed=1.2, z0=1.12, z1=1.16, grade=drn(sky=0.6), tin=('whip', 3, (-1, 0))),
    dict(n='F3', src='F3', mode='broll', t0=B(30), t1=B(31), src_in=0.30, speed=1.3, z0=1.08, z1=1.12, grade=drn(), tin=('whip', 3, (1, 0))),
    dict(n='F4', src='F4', mode='broll', t0=B(31), t1=B(32), src_in=0.30, speed=1.2, z0=1.10, z1=1.14, grade=drn(sky=0.5), tin=('whip', 3, (-1, 0))),
    dict(n='F5', src='F5', mode='broll', t0=B(32), t1=B(34), src_in=0.20, ramp=[(0.0, 2.0), (0.4, 5.5), (0.97, 5.5)], z0=1.10, z1=1.02, grade=drn(), tin=('whip', 4, (0, -1))),
    dict(n='F6', src='F6', mode='broll', t0=B(34), t1=B(36), src_in=0.50, ramp=[(0.0, 3.0), (0.97, 1.4)], z0=1.14, z1=1.06, grade=drn(), tin=('spin', 5, -1)),
    dict(n='G', src=None, mode='card', t0=B(36), t1=T_END, tin=('flash', 9)),
]

def src_time(sh, t_rel):
    if 'ramp' in sh:
        pts = sh['ramp']; s = 0.0; prev_t, prev_v = pts[0]
        for (tt, vv) in pts[1:]:
            if t_rel <= tt:
                v_now = prev_v + (vv - prev_v) * (t_rel - prev_t) / max(1e-6, tt - prev_t)
                s += (prev_v + v_now) / 2 * (t_rel - prev_t)
                return sh['src_in'] + s, v_now
            s += (prev_v + vv) / 2 * (tt - prev_t); prev_t, prev_v = tt, vv
        return sh['src_in'] + s + prev_v * (t_rel - prev_t), prev_v
    return sh['src_in'] + sh['speed'] * t_rel, sh['speed']

SPK_START = {'SPK_A': 0.90, 'SPK_E': 19.40}
SIZES = {'SPK_A': (1728, 3072), 'SPK_E': (1728, 3072)}
readers = {}
def reader(name):
    if name not in readers:
        w, h = SIZES.get(name, (DW, DH))
        readers[name] = Reader(INTER + f'{name}.mov', w, h)
    return readers[name]

def fetch(name, t_src, speed):
    """Source frame at intermediate time t_src; blends neighbours for motion blur when sped up."""
    r = reader(name)
    idx = t_src * FPS_SRC
    n = int(np.clip(round(speed * 0.9), 1, 7))
    if n <= 1:
        return r.get(int(round(idx)))
    i0 = int(round(idx - (n - 1) / 2))
    acc = None
    for k in range(n):
        f = r.get(i0 + k)
        acc = f.copy() if acc is None else acc + f
    return acc / n

TRACK_F = '/tmp/edit/work/spk_track.json'
def build_track():
    if os.path.exists(TRACK_F):
        return json.load(open(TRACK_F))
    out = {}
    for name, s0 in SPK_START.items():
        pts = []
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', INTER + f'{name}.mov', '-vf', 'fps=8,scale=432:768',
                              '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
        k = 0
        while True:
            buf = p.stdout.read(432 * 768 * 3)
            if len(buf) < 432 * 768 * 3: break
            im = np.frombuffer(buf, np.uint8).reshape(768, 432, 3)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h, s, v = [hsv[..., i].astype(int) for i in range(3)]
            m = ((h >= 4) & (h <= 22) & (s > 100) & (v > 110)).astype(np.uint8)
            m[:, :60] = 0; m[:, -40:] = 0; m[:150] = 0; m[600:] = 0
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))   # drops the thin pump boom
            ys, xs = np.nonzero(m)
            if len(xs) > 200:
                pts.append([s0 + k / 8.0, float(np.median(xs)) / 432, float(np.percentile(ys, 4)) / 768])
            k += 1
        out[name] = pts
    json.dump(out, open(TRACK_F, 'w'))
    return out

TRACK = None
def head_at(name, s):
    pts = np.array(TRACK[name])
    k = np.exp(-0.5 * ((pts[:, 0] - s) / 0.6) ** 2)      # wide smoothing: no framing jitter
    x = float((pts[:, 1] * k).sum() / k.sum()); vt = float((pts[:, 2] * k).sum() / k.sum())
    return x, vt - 0.055

LEVELS = {}
def shot_levels(sh):
    if sh['n'] in LEVELS: return LEVELS[sh['n']]
    mid = (sh['t0'] + sh['t1']) / 2
    if sh['mode'] == 'spk':
        ts = TL.T_to_src(mid) - SPK_START[sh['src']]
    else:
        ts, _ = src_time(sh, mid - sh['t0'])
    tmp = Reader(INTER + f"{sh['src']}.mov", *SIZES.get(sh['src'], (DW, DH)))
    img = tmp.get(int(ts * FPS_SRC)); tmp.close()
    small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    b, w = auto_levels(small)
    LEVELS[sh['n']] = (b - 0.012, min(1.0, w + 0.015))
    return LEVELS[sh['n']]

def ease_t(sh, T):
    return float(np.clip((T - sh['t0']) / max(1e-6, sh['t1'] - sh['t0']), 0, 1))

def shake_offset(T, t0, amp, frames, seed=3):
    k = (T - t0) * FPS
    if k < 0 or k > frames: return 0.0, 0.0, 0.0
    d = (1 - k / frames) ** 2
    rng = np.random.default_rng(seed + int(k))
    return amp * d * rng.uniform(-1, 1), amp * d * rng.uniform(-1, 1), 0.6 * d * rng.uniform(-1, 1)

def radial_blur(img, strength, k=6):
    if strength < 0.004: return img
    h, w = img.shape[:2]
    acc = img.copy()
    for i in range(1, k):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + strength * i / (k - 1))
        acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return acc / k

def spin_blur(img, angle, k=7):
    if abs(angle) < 0.3: return img
    h, w = img.shape[:2]
    acc = img.copy()
    for i in range(1, k):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle * i / (k - 1), 1.0)
        acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return acc / k

def dir_blur(img, dx, dy):
    L = int(abs(dx) + abs(dy))
    if L < 3: return img
    return cv2.blur(img, (L, 1) if abs(dx) >= abs(dy) else (1, L), borderType=cv2.BORDER_REFLECT)

def trans_params(sh, T, nxt):
    """Entry effects of this shot plus exit effects driven by the next shot's entry type."""
    fx = dict(extra_scale=1.0, rot=0.0, off=[0.0, 0.0], blur_dir=(0, 0), radial=0.0, spin=0.0, flash=0.0)
    kin = (T - sh['t0']) * FPS
    tin = sh.get('tin')
    if tin and kin < tin[1]:
        typ, n = tin[0], tin[1]
        p = 1 - kin / n; e = p ** 2
        if typ == 'whip':
            d = tin[2]; fx['off'] = [-d[0] * e * 0.45 * W, -d[1] * e * 0.45 * H]
            fx['blur_dir'] = (d[0] * e * 170, d[1] * e * 170)
        elif typ == 'zoom':
            fx['extra_scale'] = 1 + 0.25 * e; fx['radial'] = 0.10 * e
        elif typ == 'zoomout':
            fx['extra_scale'] = 1 - 0.18 * e; fx['radial'] = 0.08 * e
        elif typ == 'spin':
            fx['rot'] = -tin[2] * 28 * e; fx['extra_scale'] = 1 + 0.35 * e; fx['spin'] = -tin[2] * 16 * e
        elif typ == 'punch':
            fx['extra_scale'] = 1 + 0.07 * e
        elif typ == 'pushin':
            fx['extra_scale'] = 1 - 0.22 * e; fx['radial'] = 0.07 * e
        elif typ in ('flashzoom', 'flash'):
            fx['flash'] = 0.95 * p ** 1.6
            if typ == 'flashzoom':
                fx['extra_scale'] = 1 + 0.18 * e; fx['radial'] = 0.09 * e
    if nxt and nxt.get('tin'):
        typ, n = nxt['tin'][0], nxt['tin'][1]
        kout = (nxt['t0'] - T) * FPS
        n_out = max(2, n - 1)
        if 0 < kout <= n_out:
            p = 1 - (kout - 1) / n_out; e = p ** 2
            if typ == 'whip':
                d = nxt['tin'][2]; fx['off'] = [d[0] * e * 0.45 * W, d[1] * e * 0.45 * H]
                fx['blur_dir'] = (d[0] * e * 170, d[1] * e * 170)
            elif typ in ('zoom', 'flashzoom'):
                fx['extra_scale'] *= 1 + 0.22 * e; fx['radial'] = max(fx['radial'], 0.10 * e)
            elif typ == 'zoomout':
                fx['extra_scale'] *= 1 - 0.12 * e; fx['radial'] = max(fx['radial'], 0.07 * e)
            elif typ == 'spin':
                fx['rot'] += nxt['tin'][2] * 28 * e; fx['extra_scale'] *= 1 + 0.35 * e; fx['spin'] = nxt['tin'][2] * 16 * e
            elif typ == 'flash':
                fx['flash'] = max(fx['flash'], 0.7 * e); fx['extra_scale'] *= 1 + 0.08 * e
    return fx

OUTLINE = json.load(open('/tmp/edit/work/d2_outline.json'))
OUT_P = np.array(OUTLINE['polys'], np.float32)
def draw_outline(img, T, M, sh, ts):
    """Self-drawing glowing gold outline of the tracked excavation footprint, then a soft fill."""
    t = T - sh['t0']
    p = ease_io(np.clip((t - 0.10) / 0.80, 0, 1))
    if p <= 0: return
    k = (ts - OUTLINE['s0']) * OUTLINE['fps']
    i = int(np.clip(np.floor(k), 0, len(OUT_P) - 2)); a = float(np.clip(k - i, 0, 1))
    P = (1 - a) * OUT_P[i] + a * OUT_P[i + 1]
    Q = P @ M[:, :2].T + M[:, 2]
    pts = np.vstack([Q, Q[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = seg.sum(); target = p * L
    path = [pts[0]]; acc = 0.0
    for j, sl in enumerate(seg):
        if acc + sl <= target:
            path.append(pts[j + 1]); acc += sl
        else:
            path.append(pts[j] + (pts[j + 1] - pts[j]) * (target - acc) / max(sl, 1e-6)); break
    path = np.array(path)
    S = 4
    core = np.zeros((H // 2, W // 2), np.float32)
    cv2.polylines(core, [np.round(path * S / 2).astype(np.int32)], False, 1.0, thickness=3, lineType=cv2.LINE_AA, shift=2)
    fill_a = 0.0
    if p >= 1:
        fill_a = 0.13 * ease_out(np.clip((t - 0.92) / 0.35, 0, 1))
    glow = cv2.GaussianBlur(core, (0, 0), 5) * 1.3 + cv2.GaussianBlur(core, (0, 0), 14) * 0.9
    head = np.zeros_like(core)
    if p < 1:
        cv2.circle(head, tuple(np.round(path[-1] / 2).astype(int)), 5, 1.0, -1, lineType=cv2.LINE_AA)
        head = cv2.GaussianBlur(head, (0, 0), 6) * 3.0
    core = cv2.resize(core, (W, H)); glow = cv2.resize(glow + head, (W, H))
    gold = np.array([1.0, 0.74, 0.26], np.float32)
    if fill_a > 0:
        fm = np.zeros((H, W), np.float32)
        cv2.fillPoly(fm, [np.round(Q * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=2)
        img[:] = img * (1 - fm[..., None] * fill_a) + gold * fm[..., None] * fill_a
    img += glow[..., None] * gold * 0.9
    c = np.clip(core, 0, 1)[..., None]
    img[:] = np.clip(img * (1 - c) + np.array([1.0, 0.93, 0.72], np.float32) * c, 0, 1)

def render_shot(sh, T, nxt):
    u = ease_t(sh, T)
    z = sh['z0'] + (sh['z1'] - sh['z0']) * ease_io(u)
    fx = trans_params(sh, T, nxt)
    if sh['mode'] == 'spk':
        s = TL.T_to_src(T)
        img16 = reader(sh['src']).get(int(round((s - SPK_START[sh['src']]) * FPS_SRC)))
        sw, shh = SIZES[sh['src']]
        hx, hy = head_at(sh['src'], s)
        cx = float(np.clip(hx, 0.5 / z, 1 - 0.5 / z))
        cy = float(np.clip(hy + (0.5 - 0.35) / z, 0.5 / z, 1 - 0.5 / z))   # head at 35% from crop top (room for graphics)
    else:
        ts, spd = src_time(sh, T - sh['t0'])
        img16 = fetch(sh['src'], ts, spd)
        sw, shh = DW, DH
        cx, cy = sh.get('cx', 0.5), sh.get('cy', 0.5)
    ox = oy = orot = 0.0
    if 'shake' in sh:
        ox, oy, orot = shake_offset(T, sh['t0'], sh['shake'][1], sh['shake'][2])
    rot = fx['rot'] + orot
    s = (W / (sw / z)) * fx['extra_scale']
    M = cv2.getRotationMatrix2D((cx * sw, cy * shh), rot, s)
    M[0, 2] += W / 2 - cx * sw + fx['off'][0] + ox; M[1, 2] += H / 2 - cy * shh + fx['off'][1] + oy
    if s < 0.9:
        # pre-shrink with area filtering to avoid aliasing, then finish with a mild affine
        f = s / 0.95
        small = cv2.resize(img16, (int(sw * f), int(shh * f)), interpolation=cv2.INTER_AREA)
        M2 = M.copy(); M2[:, :2] /= f
        img = cv2.warpAffine(small, M2, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    else:
        img = cv2.warpAffine(img16, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    b, w_ = shot_levels(sh)
    g = dict(sh['grade']); g['black'] = b; g['white'] = w_
    img = grade(img, g)
    if sh.get('outline') and sh['mode'] == 'broll':
        draw_outline(img, T, M, sh, ts)
    if fx['blur_dir'] != (0, 0): img = dir_blur(img, *fx['blur_dir'])
    if fx['radial'] > 0: img = radial_blur(img, fx['radial'])
    if fx['spin'] != 0: img = spin_blur(img, fx['spin'])
    if fx['flash'] > 0: img = img + (1 - img) * fx['flash']
    return img

WORDS = TL.WORDS
def wT(prefix, nth=0):
    return [w for w in WORDS if w['w'].startswith(prefix)][nth]

PHRASES = [
    (['çalışmalar', 'hız', 'kesmeden'], 'www'), (['devam', 'ediyor.'], 'ww'),
    (['Temel', 'uygulamalarımızın'], 'gw'), (['önemli', 'aşamalarından'], 'ww'), (['biri', 'olan'], 'ww'),
    (['gerçekleştiriyoruz.'], 'w'), (['ön', 'planda', 'tutarak'], 'www'),
    (["Vista", "Kurtuluş'u"], 'gg'), (['adım', 'adım'], 'ww'), (['geleceğe', 'hazırlıyoruz.'], 'ww'),
    (['Çil', 'Yapı', 'güvencesiyle'], 'ggw'), (['yükselmeye', 'devam', 'ediyoruz.'], 'gww'),
]
def build_phrases():
    out, pos = [], 0
    for words, sty in PHRASES:
        items = []
        for wd, st in zip(words, sty):
            while WORDS[pos]['w'] != wd: pos += 1       # next occurrence in speaking order
            wrec = WORDS[pos]; pos += 1
            items.append((wd, st, wrec['T'], wrec['Te']))
        out.append(items)
    hard_stops = [B(0)]                  # captions never straddle the GROBETON title slam
    res = []
    for i, it in enumerate(out):
        start = it[0][2] - 0.04
        end = it[-1][3] + 0.40
        if i + 1 < len(out): end = min(end, out[i + 1][0][2] - 0.04)
        for hs in hard_stops:
            if start < hs < end: end = hs
        res.append((it, start, end))
    return res

SPR = {}
def word_sprite(wd, st):
    if (wd, st) not in SPR:
        SPR[(wd, st)] = caption_word(wd, GOLD_TXT if st == 'g' else (255, 255, 255), size=64,
                                     glow=(255, 160, 30) if st == 'g' else None)
    return SPR[(wd, st)]

CAP_Y = 1235
def draw_captions(frame, T, phrases):
    for items, start, end in phrases:
        if not (start <= T < end): continue
        sprs = [word_sprite(wd, st) for wd, st, _, _ in items]
        space = 18
        widths = [s.shape[1] - 60 for s in sprs]
        total = sum(widths) + space * (len(sprs) - 1)
        scale = min(1.0, 860 / total)   # keep clear of the Reels action column
        x = W / 2 - total * scale / 2
        fade_out = np.clip((end - T) / 0.08, 0, 1)
        for (wd, st, t0, _), spr, wdt in zip(items, sprs, widths):
            k = (T - (t0 - 0.04)) * FPS
            if k >= 0:
                p = np.clip(k / 5.0, 0, 1)
                place(frame, spr, x + wdt * scale / 2, CAP_Y + 10 * (1 - ease_out(p)),
                      scale=scale * (1.28 - 0.28 * ease_out(p)), blur=5 * (1 - p), opacity=min(1.0, k / 2.5) * fade_out)
            x += (wdt + space) * scale

G = {}
def init_gfx():
    G['t1'] = title_3d(['VİSTA', 'KURTULUŞ'], [196, 98], ['Montserrat-Black.ttf', 'Montserrat-ExtraBold.ttf'], [6, 18], depth=14)
    G['t2'] = title_3d(['GROBETON', 'DÖKÜMÜ'], [136, 108], ['Montserrat-Black.ttf', 'Montserrat-Black.ttf'], [4, 10], depth=12, face='white', glow=(255, 190, 80))
    ren = cv2.imread('/tmp/edit/assets/render_raw.png')[42:, 12:-24, ::-1].astype(np.float32) / 255
    ren = grade(ren, dict(black=0.10, white=0.98, contrast=1.25, sat=1.35, clarity=0.15, sharpen=0.4, gain=[1.02, 1.0, 0.98]))
    G['card'] = card(ren, 780)
    G['anchor'] = anchor_sprite(520)
    G['hdr'] = flat_text('HER AŞAMADA', 48, 'Montserrat-Bold.ttf', 16, gold=True)
    G['items'] = [flat_text(t, 70, 'Poppins-Bold.ttf', 1, color=(1, 1, 1), shadow=0.6) for t in ['SAĞLAMLIK', 'KALİTE', 'DOĞRU UYGULAMA']]
    G['brand'] = flat_text('ÇİL', 170, 'Montserrat-Black.ttf', 14, gold=True)
    G['sub'] = flat_text('YAPI & İNŞAAT', 54, 'Montserrat-SemiBold.ttf', 16, color=(0.97, 0.92, 0.80))
    G['project'] = flat_text('VİSTA | I   ·   KURTULUŞ', 38, 'Montserrat-Medium.ttf', 8, color=(0.92, 0.94, 0.92))
    G['bg'] = brand_background()

def draw_title(frame, T, spr_m, t_in, t_out, cy, shine_t=None):
    spr, m = spr_m
    if not (t_in <= T < t_out + 0.2): return
    k = (T - t_in) * FPS
    p = np.clip(k / 6.0, 0, 1)
    sc = 2.1 - 1.1 * ease_out(p, 4); bl = 10 * (1 - p); op = min(1.0, k / 2.0)
    cs = 14 * (1 - p) + (3 if k < 10 else 0)
    if T > t_out:
        q = (T - t_out) / 0.2; sc = 1 + 0.15 * q; bl = 8 * q; op = 1 - q
    elif k > 6:
        sc = 1.0 + 0.012 * (k - 6) / FPS
    s2 = spr
    if shine_t is not None and shine_t[0] <= T <= shine_t[1]:
        s2 = shine(spr, m, -0.2 + 1.4 * (T - shine_t[0]) / (shine_t[1] - shine_t[0]), width=0.09, strength=0.9)
    place(frame, chroma_split(s2, cs), W / 2, cy, scale=sc, blur=bl, opacity=op)

def draw_card(frame, T, t_in, t_out, cy=560):
    if not (t_in <= T < t_out): return
    k = (T - t_in) * FPS
    p = np.clip(k / 9.0, 0, 1)
    sc = 0.55 + 0.45 * ease_out_back(p, 2.2); ang = -7 + 4 * ease_out(p); op = min(1.0, k / 3.0)
    q = np.clip((t_out - T) / 0.17, 0, 1)
    if q < 1: sc *= 0.85 + 0.15 * q; op *= q
    place(frame, G['card'], W / 2, cy - 12 * (T - t_in), scale=sc, angle=ang, opacity=op, blur=2 * (1 - p))

def frosted_panel(frame, x0, y0, x1, y1, op, radius=34):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x0 = max(0, x0); x1 = min(W, x1)
    roi = frame[y0:y1, x0:x1]
    h, w = roi.shape[:2]
    bl = cv2.resize(cv2.GaussianBlur(cv2.resize(roi, (w // 4, h // 4)), (0, 0), 5), (w, h))
    m = np.zeros((h, w), np.float32)
    cv2.rectangle(m, (radius, 0), (w - radius, h), 1, -1); cv2.rectangle(m, (0, radius), (w, h - radius), 1, -1)
    for cx, cy in [(radius, radius), (w - radius, radius), (radius, h - radius), (w - radius, h - radius)]:
        cv2.circle(m, (cx, cy), radius, 1, -1, lineType=cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    edge = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)[..., None] * 0.45 * op
    m = m[..., None] * op
    roi[:] = roi * (1 - m) + (bl * 0.40 + np.array([0.015, 0.045, 0.035], np.float32)) * m
    roi[:] = roi * (1 - edge) + np.array([1.0, 0.86, 0.55], np.float32) * edge

def draw_checklist(frame, T):
    t_in, t_out = wT('Her')['T'] - 0.10, B(14) - 0.03
    if not (t_in <= T < t_out): return
    k = (T - t_in) * FPS
    p = ease_out(np.clip(k / 7.0, 0, 1)); q = np.clip((t_out - T) / 0.2, 0, 1); op = p * q
    px0, px1, py0, py1 = 110, 970, 610, 1110
    slide = (1 - p) * 60 - (1 - q) * 80
    frosted_panel(frame, px0 + slide, py0, px1 + slide, py1, op)
    place(frame, G['hdr'][0], W / 2 + slide, py0 + 70, opacity=op)
    times = [wT('sağlamlık')['T'], wT('kalite')['T'], wT('doğru')['T']]
    for i, (t_i, (spr, _)) in enumerate(zip(times, G['items'])):
        if T < t_i - 0.05: continue
        ki = (T - (t_i - 0.05)) * FPS; pi = np.clip(ki / 7.0, 0, 1)
        y = py0 + 190 + i * 118
        place(frame, check_icon(86, np.clip(ki / 8.0, 0, 1)), px0 + 100 + slide, y, scale=0.6 + 0.4 * ease_out_back(pi, 2.5), opacity=op)
        place(frame, spr, px0 + 152 + spr.shape[1] / 2 - 24 + slide - 40 * (1 - ease_out(pi)), y + 4, opacity=op * min(1, ki / 3), blur=4 * (1 - pi))

BOKEH = np.random.default_rng(11).uniform(0, 1, (26, 5))
def draw_endcard(frame, T):
    t = T - B(36)
    frame[:] = G['bg']
    ov = np.zeros_like(frame); am_ = np.zeros(frame.shape[:2], np.float32)
    for x, y, r, sp, a in BOKEH:
        yy = (y * H - (10 + 30 * sp) * t) % H
        cv2.circle(am_, (int(x * W), int(yy)), int(6 + 20 * r), float(0.04 + 0.08 * a), -1, lineType=cv2.LINE_AA)
    am_ = cv2.GaussianBlur(am_, (0, 0), 3) * min(1, t / 0.6)
    frame[:] = frame * (1 - am_[..., None]) + np.array([0.95, 0.75, 0.35], np.float32) * am_[..., None]
    an, am = G['anchor']
    pa = np.clip(t / 0.55, 0, 1)
    sc = 1.25 - 0.25 * ease_out_back(pa, 1.6)
    s_an = shine(an, am, -0.25 + 1.5 * (t - 0.55) / 0.9, width=0.10, strength=0.85) if 0.55 <= t <= 1.45 else an
    hit = 4 * BEAT                           # last musical hit (bar after logo downbeat)
    pulse = float(np.exp(-((t - hit) / 0.12) ** 2))
    place(frame, s_an, W / 2, 690, scale=sc * (1 + 0.02 * pulse), opacity=min(1, t / 0.18), blur=8 * (1 - pa))
    for key, cy, td in [('brand', 1122, 0.30), ('sub', 1252, 0.50), ('project', 1395, hit - 0.05)]:
        spr = G[key][0]
        pt = np.clip((t - td) / 0.40, 0, 1)
        if pt > 0:
            place(frame, spr, W / 2, cy + 36 * (1 - ease_out(pt)), opacity=ease_out(pt), blur=5 * (1 - pt))
    pl = np.clip((t - (hit - 0.25)) / 0.35, 0, 1)
    if pl > 0:
        half = 190 * ease_out(pl)
        cv2.line(frame, (int(W / 2 - half), 1330), (int(W / 2 + half), 1330), (0.86, 0.68, 0.32), 2, cv2.LINE_AA)
    fo = np.clip((T - (T_END - 0.55)) / 0.55, 0, 1)
    if fo > 0: frame *= (1 - fo)

_VIG = None
def finish(img, T):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _VIG = (1 - 0.16 * np.clip((r - 0.55) / 0.85, 0, 1) ** 1.6)[..., None].astype(np.float32)
    img = img * _VIG
    gr = np.random.default_rng(int(T * 1000)).normal(0, 0.010, (H // 2, W // 2)).astype(np.float32)
    return np.clip(img + cv2.resize(gr, (W, H))[..., None], 0, 1)

def shot_at(T):
    for i, sh in enumerate(SHOTS_T):
        if sh['t0'] <= T < sh['t1']: return i, sh
    return len(SHOTS_T) - 1, SHOTS_T[-1]

def render_frame(fi, phrases):
    T = fi / FPS
    i, sh = shot_at(T)
    nxt = SHOTS_T[i + 1] if i + 1 < len(SHOTS_T) else None
    if sh['mode'] == 'card':
        frame = np.zeros((H, W, 3), np.float32)
        draw_endcard(frame, T)
        kin = (T - sh['t0']) * FPS
        if kin < 9: frame = frame + (1 - frame) * 0.95 * (1 - kin / 9) ** 1.6
        return finish(frame, T)
    frame = render_shot(sh, T, nxt)
    draw_title(frame, T, G['t1'], wT('Vista')['T'] - 0.06, 1.18, 330, shine_t=(0.62, 1.10))
    draw_card(frame, T, 1.36, 3.02, cy=300)
    draw_title(frame, T, G['t2'], B(0), B(4) - 0.22, 700, shine_t=(B(0) + 0.25, B(0) + 0.85))
    draw_checklist(frame, T)
    draw_captions(frame, T, phrases)
    return finish(frame, T)

def main():
    global TRACK
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=str, default='')
    ap.add_argument('--out', type=str, default='/tmp/edit/out/video_nosound.mp4')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=None)
    a = ap.parse_args()
    TRACK = build_track(); init_gfx(); phrases = build_phrases()
    if a.frames:
        os.makedirs('/tmp/edit/out/frames', exist_ok=True)
        for f in sorted(int(round(float(x) * FPS)) for x in a.frames.split(',')):
            img = render_frame(f, phrases)
            cv2.imwrite(f'/tmp/edit/out/frames/f_{f:04d}.png', (img[..., ::-1] * 255 + 0.5).astype(np.uint8))
            print('frame', f, flush=True)
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n0 = int(round(a.start * FPS)); n1 = int(round((a.end or T_END) * FPS))
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', a.out],
                           stdin=subprocess.PIPE)
    t_start = time.time(); rng = np.random.default_rng(1)
    for f in range(n0, n1):
        img = render_frame(f, phrases)
        dith = rng.uniform(-0.5, 0.5, (H, W, 1)).astype(np.float32) / 255.0
        enc.stdin.write(((img + dith) * 255 + 0.5).clip(0, 255).astype(np.uint8).tobytes())
        if f % 30 == 0: print(f'{f}/{n1} {time.time() - t_start:.0f}s', flush=True)
    enc.stdin.close(); enc.wait()
    print('DONE', a.out, f'{time.time() - t_start:.0f}s', flush=True)

if __name__ == '__main__':
    main()
