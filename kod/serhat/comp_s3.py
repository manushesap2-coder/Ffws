"""Serhat Insaat - Inci Projesi: compositor (1080x1920, 30 fps)."""
import sys, os, json, time, subprocess, argparse
import numpy as np, cv2
sys.path.insert(0, '/tmp/jobs/serhat/lib')
from core import Reader
from gfx import composite, place, ease_out, ease_in, ease_io, ease_out_back
from sgfx import *
from sgrade import grade_s
from sgrade2 import grade_int2
import cam2
import tl_s3 as TL
import callouts as CO
from tl_s3 import W, H, FPS, FPS_SRC, T_DROP, T_VO, T_END, B

INTER = '/tmp/jobs/serhat/inter/'
G_INT = dict(ev=0.45, shadows=0.42, hl=0.9, contrast=1.10, sat=1.12, vib=0.35, clarity=0.18, sharpen=0.25, gain=[1.025, 1.0, 0.97])
G_EXT = dict(ev=0.05, shadows=0.12, hl=1.2, contrast=1.16, sat=1.18, vib=0.40, sky=0.55, clarity=0.22, sharpen=0.25, gain=[1.02, 1.0, 0.975])
def mix_grade(a, b, k):
    out = {}
    for key in set(a) | set(b):
        va, vb = a.get(key, 0.0), b.get(key, 0.0)
        out[key] = (np.array(va) * (1 - k) + np.array(vb) * k).tolist() if isinstance(va, list) else va * (1 - k) + vb * k
    return out

# ---------------------------------------------------------------- sources
readers = {}
def rd(name):
    if name not in readers: readers[name] = Reader(INTER + name + '.mp4', 1296, 2304)
    return readers[name]
def drone_loc(t):
    if t < 38.1: return 'D_A', t
    if t < 116.1: return 'D_B', t - 38.0
    return 'D_C', t - 116.0
def fetch(name, t, speed):
    r = rd(name); idx = t * FPS_SRC
    n = int(np.clip(round(abs(speed) * 0.85), 1, 8))
    if n <= 1: return r.get(int(round(idx)))
    i0 = int(round(idx - (n - 1) / 2)); acc = None
    for k in range(n):
        f = r.get(i0 + k)
        acc = f.copy() if acc is None else acc + f
    return acc / n
def src_frame(kind, src, t, v):
    if src == 'D':
        nm, lt = drone_loc(t); return fetch(nm, lt, v), nm, lt
    return fetch(src, t, v), src, t

# ---------------------------------------------------------------- rev2: 4K interior one-take
D4R_S0 = 1.001 + 1739 / FPS_SRC   # D4R.mp4 frame 0 = D4 frame 1739; source time runs backwards
D4_T0 = 1.001            # D4.mp4 starts at this source second (denoised, LUT applied, native 2160x3840)
class R16:
    """Sequential uint16 RGB reader (no float conversion; crop first, convert later)."""
    def __init__(self, path, w, h, fps=FPS_SRC):
        self.path, self.w, self.h, self.fps = path, w, h, fps
        self.proc = None; self.pos = -1; self.cur = None; self._open(0)
    def _open(self, idx):
        if self.proc: self.proc.kill()
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{idx / self.fps:.5f}', '-i', self.path, '-f', 'rawvideo',
                                      '-pix_fmt', 'rgb48le', '-'], stdout=subprocess.PIPE, bufsize=10 ** 8)
        self.pos = idx - 1; self.cur = None
    def get(self, idx):
        idx = max(0, int(idx))
        if idx < self.pos or idx > self.pos + 90: self._open(idx)
        n = self.w * self.h * 6
        while self.pos < idx:
            buf = self.proc.stdout.read(n)
            if len(buf) < n: return self.cur
            self.pos += 1
            if self.pos == idx: self.cur = np.frombuffer(buf, np.uint16).reshape(self.h, self.w, 3)
        return self.cur
_R4 = None
def fetch4k(s, v, box):
    global _R4
    if _R4 is None: _R4 = R16(INTER + 'D4.mp4', 2160, 3840)
    x0, y0, x1, y1 = box; idx = (s - D4_T0) * FPS_SRC
    n = int(np.clip(round(abs(v) * 0.85), 1, 8))
    if n <= 1: return _R4.get(int(round(idx)))[y0:y1, x0:x1].astype(np.float32) / 65535.0
    i0 = int(round(idx - (n - 1) / 2)); acc = None
    for k in range(n):
        f = _R4.get(i0 + k)[y0:y1, x0:x1].astype(np.float32)
        acc = f if acc is None else acc + f
    return acc / (n * 65535.0)
def plate4k(s, v, z, cx, cy):
    """Render the output plate from the 4K take; framing given in D_A coords (DA = 0.6 x 4K, pixel-centre aligned)."""
    so = (W / 1296.0) * z; s4 = so * 0.6
    c4x, c4y = (cx + 0.5) / 0.6 - 0.5, (cy + 0.5) / 0.6 - 0.5
    hw, hh = (W / 2) / s4 + 12, (H / 2) / s4 + 12
    x0, x1 = int(max(0, np.floor(c4x - hw))), int(min(2160, np.ceil(c4x + hw)))
    y0, y1 = int(max(0, np.floor(c4y - hh))), int(min(3840, np.ceil(c4y + hh)))
    crop = fetch4k(s, v, (x0, y0, x1, y1))
    ox, oy = W / 2 - s4 * (c4x - x0), H / 2 - s4 * (c4y - y0)
    if s4 < 0.97:
        nw, nh = max(2, int(round((x1 - x0) * s4 / 0.97))), max(2, int(round((y1 - y0) * s4 / 0.97)))
        fx_, fy_ = nw / (x1 - x0), nh / (y1 - y0)
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        M = np.array([[s4 / fx_, 0, ox + s4 * (0.5 / fx_ - 0.5)], [0, s4 / fy_, oy + s4 * (0.5 / fy_ - 0.5)]])
    else:
        M = np.array([[s4, 0, ox], [0, s4, oy]])
    return cv2.warpAffine(crop, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

# ---------------------------------------------------------------- tracking data
PL = json.load(open('/tmp/jobs/serhat/track/planes.json'))
HF = {int(k): np.array(v) for k, v in PL['H']['floor'].items()}
HC = {int(k): np.array(v) for k, v in PL['H']['ceil'].items()}
KMIN, KMAX = min(HF), max(HF)
ROOF = json.load(open('/tmp/jobs/serhat/track/roof.json'))
RP = {int(k): v for k, v in ROOF['pts'].items()}
def hproj(Hm, P):
    P = np.atleast_2d(np.asarray(P, float)); q = np.c_[P, np.ones(len(P))] @ Hm.T
    return q[:, :2] / q[:, 2:3]
def interp_H(D, kf):
    k0 = int(np.clip(np.floor(kf), KMIN, KMAX - 1)); a = float(np.clip(kf - k0, 0, 1))
    return D[k0], D[k0 + 1], a
def floor_pts(kf, P):
    A, Bm, a = interp_H(HF, kf); return hproj(A, P) * (1 - a) + hproj(Bm, P) * a
def ceil_pts(kf, P):
    A, Bm, a = interp_H(HC, kf); return hproj(A, P) * (1 - a) + hproj(Bm, P) * a
BASES = np.array([[270.0, 1335.0], [1170.0, 1340.0]])
def facade_pts(kf, P):
    """Back facade is fronto-parallel: similarity from the two tracked pillar bases."""
    Q = floor_pts(kf, BASES); P = np.atleast_2d(np.asarray(P, float))
    d0 = BASES[1] - BASES[0]; d1 = Q[1] - Q[0]
    s = np.linalg.norm(d1) / np.linalg.norm(d0)
    ang = np.arctan2(d1[1], d1[0]) - np.arctan2(d0[1], d0[0]); c, sn = np.cos(ang), np.sin(ang)
    R = np.array([[c, -sn], [sn, c]]) * s
    return (P - BASES[0]) @ R.T + Q[0]

# ---------------------------------------------------------------- framing
def take_zoom(T):
    if T < 4.5: z = 1.0
    elif T < 13.0: z = 1.0 + 0.16 * ease_io((T - 4.5) / 8.5)
    elif T < 17.75: z = 1.16
    elif T < T_DROP: z = 1.16 - 0.16 * ease_io((T - 17.75) / (T_DROP - 17.75))
    else: z = 1.0 + 0.05 * ease_io(np.clip((T - T_DROP) / 12.0, 0, 1))
    cx, cy = (680.0, 1150.0) if T < T_DROP else (648.0, 1152.0)
    if 17.75 <= T < T_DROP:
        k = ease_io((T - 17.75) / (T_DROP - 17.75)); cx, cy = 680 + (648 - 680) * k, 1150 + 2 * k
    return z, cx, cy
def affine(z, cx, cy):
    s = (W / 1296.0) * z
    return np.array([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy]], np.float64)
def to_out(M, P):
    P = np.atleast_2d(P); return P @ M[:, :2].T + M[:, 2]

# ---------------------------------------------------------------- transitions
def trans_fx(sh, T, nxt):
    fx = dict(scale=1.0, off=[0.0, 0.0], blur=(0, 0), radial=0.0, flash=0.0)
    tin = sh.get('tin'); kin = (T - sh['t0']) * FPS
    if tin and kin < tin[1]:
        typ, n = tin[0], tin[1]; e = (1 - kin / n) ** 2
        if typ == 'whip':
            d = tin[2]; fx['off'] = [-d[0] * e * 0.42 * W, -d[1] * e * 0.42 * H]; fx['blur'] = (d[0] * e * 160, d[1] * e * 160)
        elif typ == 'zoom': fx['scale'] = 1 + 0.22 * e; fx['radial'] = 0.09 * e
        elif typ == 'zoomout': fx['scale'] = 1 - 0.16 * e; fx['radial'] = 0.07 * e
    if nxt and nxt.get('tin'):
        typ, n = nxt['tin'][0], nxt['tin'][1]; kout = (nxt['t0'] - T) * FPS; n2 = max(2, n - 1)
        if 0 < kout <= n2:
            e = (1 - (kout - 1) / n2) ** 2
            if typ == 'whip':
                d = nxt['tin'][2]; fx['off'] = [d[0] * e * 0.42 * W, d[1] * e * 0.42 * H]; fx['blur'] = (d[0] * e * 160, d[1] * e * 160)
            elif typ == 'zoom': fx['scale'] *= 1 + 0.2 * e; fx['radial'] = max(fx['radial'], 0.09 * e)
            elif typ == 'zoomout': fx['scale'] *= 1 + 0.12 * e; fx['radial'] = max(fx['radial'], 0.06 * e)
    return fx
def radial_blur(img, strength, k=6):
    if strength < 0.004: return img
    acc = img.copy()
    for i in range(1, k):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + strength * i / (k - 1))
        acc += cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / k
def dir_blur(img, dx, dy):
    L = int(abs(dx) + abs(dy))
    if L < 3: return img
    return cv2.blur(img, (L, 1) if abs(dx) >= abs(dy) else (1, L), borderType=cv2.BORDER_REFLECT)
def shake(T, t0, amp, dur, seed=5):
    k = (T - t0) / dur
    if k < 0 or k > 1: return 0.0, 0.0
    d = (1 - k) ** 2; r = np.random.default_rng(seed + int((T - t0) * FPS))
    return amp * d * r.uniform(-1, 1), amp * d * r.uniform(-1, 1)

def shot_at(T):
    for i, s in enumerate(TL.SHOTS):
        if s['t0'] <= T < s['t1']: return i, s
    return len(TL.SHOTS) - 1, TL.SHOTS[-1]

def render_plate(T):
    """Returns (graded frame, M affine intermediate->out, info)."""
    i, sh = shot_at(T); nxt = TL.SHOTS[i + 1] if i + 1 < len(TL.SHOTS) else None
    fx = trans_fx(sh, T, nxt)
    info = dict(shot=sh['n'])
    if sh['kind'] == 'take':
        s, v = TL.take_src(T)
        if T < T_DROP:
            z, cx, cy = cam2.cam(T)
            raw = plate4k(s, v, z, cx, cy)
            out = grade_int2(raw, sharp=0.4 + 0.3 * (z - 1))
            kx = float(np.clip((T - (T_DROP - 0.3)) / 0.3, 0, 1))
            if kx > 0: out = out * (1 - kx) + grade_s(raw, G_EXT) * kx
            kf = (T - T_DROP) * FPS
            if -2 <= kf < 6:
                a = 0.85 * (1 - (kf + 2) / 8) ** 1.5
                out = out + (1 - out) * a
            info.update(src=s, v=v, name='D4', lt=s - D4_T0, kf=s * FPS_SRC, z=z)
            return out, affine(z, cx, cy), info
        img, nm, lt = src_frame('take', 'D', s, v)
        z, cx, cy = take_zoom(T)
        g = G_EXT
        info.update(src=s, v=v, name=nm, lt=lt, kf=s * FPS_SRC)
    elif sh['kind'] == 'rev':
        s, v = TL.rev_src(T)
        lt = D4R_S0 - s
        raw_src = fetch('D4R', lt, v)
        u = T - sh['t0']; z0, z1 = sh['zoom']
        z = (z0 + (z1 - z0) * ease_io(np.clip(u / (sh['t1'] - sh['t0']), 0, 1))) * fx['scale']
        cx, cy = sh['center']
        M = affine(z, cx, cy); M[0, 2] += fx['off'][0]; M[1, 2] += fx['off'][1]
        raw = cv2.warpAffine(raw_src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        kx = float(np.clip((s - 28.3) / (29.35 - 28.3), 0, 1))
        out = grade_int2(raw, sharp=0.4 + 0.3 * (z - 1))
        if kx > 0: out = out * (1 - kx) + grade_s(raw, G_EXT) * kx
        if fx['blur'] != (0, 0): out = dir_blur(out, *fx['blur'])
        out = radial_blur(out, fx['radial'])
        info.update(src=s, v=v, name='D4R', lt=lt)
        return out, M, info
    else:
        u = (T - sh['t0']); s = sh['s0'] + sh['v'] * u; v = sh['v']
        img, nm, lt = src_frame('lin', sh['src'], s, v)
        z0, z1 = sh.get('zoom', (1.02, 1.08))
        z = z0 + (z1 - z0) * ease_io(np.clip(u / (sh['t1'] - sh['t0']), 0, 1))
        cx, cy = sh.get('center', (648.0, 1152.0))
        g = G_INT if sh['grade'] == 'int' else G_EXT
        info.update(src=s, v=v, name=nm, lt=lt)
    z *= fx['scale']
    M = affine(z, cx, cy)
    M[0, 2] += fx['off'][0]; M[1, 2] += fx['off'][1]
    sx, sy = shake(T, T_DROP, 16, 0.45)
    M[0, 2] += sx; M[1, 2] += sy
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = grade_s(out, g)
    if fx['blur'] != (0, 0): out = dir_blur(out, *fx['blur'])
    out = radial_blur(out, fx['radial'])
    # exit flash
    kf = (T - T_DROP) * FPS
    if -2 <= kf < 6:
        a = 0.85 * (1 - (kf + 2) / 8) ** 1.5
        out = out + (1 - out) * a
    return out, M, info

# ================================================================ graphics
SPR = {}
LOGO_BB = (445, 1481, 1715, 2177)
def init_gfx():
    SPR['tag_w'] = tag_spr('GENİŞLİK', '300 m²')
    SPR['tag_l'] = tag_spr('UZUNLUK', '30 m')
    SPR['tag_h'] = tag_spr('TAVAN YÜKSEKLİĞİ', '4,5 m')
    SPR['st1'] = text_spr('Bir işletmenin gücü', 'Montserrat-SemiBold.ttf', 50)
    SPR['st2a'] = grad_text_spr('KONUMLA', 'Montserrat-Black.ttf', 92, tracking=1)
    SPR['st2b'] = text_spr('BAŞLAR', 'Montserrat-Black.ttf', 92, tracking=1)
    SPR['ti1'] = text_spr('SERHAT İNŞAAT', 'Montserrat-Bold.ttf', 42, color=WH, tracking=12)
    SPR['ti2'] = text_spr('İNCİ PROJESİ', 'Cinzel-VF.ttf', 112, wght=700, tracking=4)
    init_callouts()
    SPR['pin'] = shadowed(solid(icon_mask('pin', 120), OR), blur=8, off=(0, 6), strength=0.6, pad=20)
    SPR['pin_lbl'] = chip_spr('star', 'ÇARŞININ TAM MERKEZİNDE', 34, dark=0.82)
    SPR['traffic'] = chip_spr('car', 'YOĞUN YAYA & ARAÇ TRAFİĞİ', 44)
    SPR['c1'] = chip_spr('area', 'GENİŞ KULLANIM ALANI', 46)
    SPR['c2'] = chip_spr('height', 'YÜKSEK TAVAN', 46)
    SPR['c3'] = chip_spr('stairs', 'ASMA KAT İMKÂNI', 46)
    SPR['tg1'] = text_spr('FARKLI İŞ KOLLARINA UYGUN', 'Montserrat-ExtraBold.ttf', 54, tracking=1)
    SPR['tg2'] = grad_text_spr('FERAH & FONKSİYONEL', 'Montserrat-Black.ttf', 76, tracking=1)
    SPR['a1'] = chip_spr('parking', 'OTOPARK AVANTAJI', 46)
    SPR['a2'] = chip_spr('tree', 'SOSYAL ALANLARA YAKIN', 46)
    SPR['a3'] = chip_spr('gov', 'RESMİ KURUMLARA YAKIN', 46)
    SPR['g1a'] = grad_text_spr('GÜÇLÜ', 'Montserrat-Black.ttf', 100, tracking=1)
    SPR['g1b'] = text_spr('KONUM', 'Montserrat-Black.ttf', 100, tracking=1)
    SPR['g2a'] = grad_text_spr('DEĞERLİ', 'Montserrat-Black.ttf', 100, tracking=1)
    SPR['g2b'] = text_spr('FIRSAT', 'Montserrat-Black.ttf', 100, tracking=1)
    SPR['f_proj'] = text_spr('İNCİ PROJESİ', 'Cinzel-VF.ttf', 84, wght=700, tracking=8)
    SPR['f_t1'] = text_spr('Çarşının merkezinde,', 'Montserrat-Medium.ttf', 50)
    SPR['f_t2'] = text_spr('ticaretin tam kalbinde.', 'Montserrat-SemiBold.ttf', 50)
    lg = cv2.imread('/tmp/jobs/serhat/work/logo.png', cv2.IMREAD_UNCHANGED)
    x0, y0, x1, y1 = LOGO_BB
    lg = lg[y0 - 6:y1 + 6, x0 - 6:x1 + 6]
    wl = 780; hl = int(lg.shape[0] * wl / lg.shape[1])
    lg = cv2.resize(lg, (wl, hl), interpolation=cv2.INTER_AREA).astype(np.float32) / (65535.0 if lg.dtype == np.uint16 else 255.0)
    a = lg[..., 3:4]; rgb = lg[..., 2::-1]
    SPR['logo'] = np.concatenate([rgb * a, a], -1)
    SPR['logo_split'] = int(hl * 0.80)      # skyline above, wordmark below

def fade_io(T, t_in, t_out, d_in=0.25, d_out=0.3):
    return float(np.clip((T - t_in) / d_in, 0, 1) * np.clip((t_out - T) / d_out, 0, 1))

def pop(frame, spr, cx, cy, T, t_in, t_out, rise=40, back=1.6, d_out=0.3, scale0=0.85):
    if T < t_in or T > t_out: return
    p = np.clip((T - t_in) / 0.32, 0, 1); q = np.clip((t_out - T) / d_out, 0, 1)
    sc = scale0 + (1 - scale0) * ease_out_back(p, back)
    place(frame, spr, cx, cy + rise * (1 - ease_out(p)) - 20 * (1 - q), scale=sc, opacity=float(min(1, p * 1.8) * q), blur=4 * (1 - p))

def slide_in(frame, spr, x_left, cy, T, t_in, t_out, dist=160):
    if T < t_in or T > t_out: return
    p = ease_out(np.clip((T - t_in) / 0.38, 0, 1), 3); q = np.clip((t_out - T) / 0.3, 0, 1)
    h, w = spr.shape[:2]
    composite(frame, spr * float(min(1, p * 1.5) * q), x_left - dist * (1 - p) - 60 * (1 - q), cy - h / 2)

# ---- measurements (reference frame src 9.0)
W_P = np.array([[275.0, 652.0], [1165.0, 652.0]])
H_P = np.array([[1205.0, 585.0], [1205.0, 1335.0]])
L_P = np.array([[215.0, 20.0], [433.0, 575.0]])   # ceiling, near (top) -> far (beam)
def draw_measures(frame, T, info, M):
    if not (6.8 <= T <= 12.9): return
    kf = info['kf']
    fo = float(np.clip((12.75 - T) / 0.4, 0, 1))
    # width on the beam (facade)
    wp = to_out(M, facade_pts(kf, W_P)); pw = ease_out(np.clip((T - 6.85) / 0.6, 0, 1), 3)
    draw_dim(frame, wp[0], wp[1], pw, alpha=fo, mode='center')
    if T >= 7.38:
        m = (wp[0] + wp[1]) / 2; spr = SPR['tag_w']
        pop(frame, spr, m[0], m[1] - spr.shape[0] / 2 - 4, T, 7.38, 12.75, rise=30)
    # length on the floor (near -> far)
    lp = to_out(M, ceil_pts(kf, L_P)); pl = ease_out(np.clip((T - 8.30) / 0.65, 0, 1), 3)
    draw_dim(frame, lp[0], lp[1], pl, alpha=fo, mode='fwd')
    if T >= 8.88:
        spr = SPR['tag_l']; th, tw = spr.shape[:2]
        # tag sits up-left of the line's near end, clamped inside the frame
        cx = float(np.clip(lp[0][0] - tw / 2 + 24, tw / 2 - 18, W - tw / 2)); cy = float(np.clip(lp[0][1] - th / 2 + 18, th / 2 + 40, H))
        pop(frame, spr, cx, cy, T, 8.88, 12.75, rise=30)
    # ceiling height on the right pillar, top -> bottom
    hp = to_out(M, facade_pts(kf, H_P)); ph = ease_out(np.clip((T - 9.88) / 0.62, 0, 1), 3)
    draw_dim(frame, hp[0], hp[1], ph, alpha=fo, mode='fwd')
    if T >= 10.40:
        spr = SPR['tag_h']; th, tw = spr.shape[:2]
        # dimension text sits on its own line (line runs behind the tag), clear of the speaker
        y = hp[0][1] + (hp[1][1] - hp[0][1]) * 0.46
        cx = float(min(hp[0][0], W - tw / 2 + 22))
        pop(frame, spr, cx, y, T, 10.40, 12.75, rise=30)

# ================================================================ v3 callouts (customer markings)
ST = json.load(open('/tmp/jobs/serhat/v3/strokes.json'))
TRK = {}
def trk_H(name, s):
    if name not in TRK:
        d = json.load(open(f'/tmp/jobs/serhat/v3/{name}.json'))
        TRK[name] = (d['off'], {int(k): np.array(v) for k, v in d['H'].items()})
    off, Hs = TRK[name]; kf = (s - off) * FPS_SRC
    k0 = int(np.clip(np.floor(kf), min(Hs), max(Hs) - 1)); a = float(np.clip(kf - k0, 0, 1))
    return Hs[k0], Hs[k0 + 1], a
def tpts(name, s, P):
    A, Bm, a = trk_H(name, s)
    return hproj(A, P) * (1 - a) + hproj(Bm, P) * a
XC = np.array([650.0, 1185.0])                                   # intersection centre (ref frame, Yol)
ROAD_A = np.array([[512.0, 1174.0], [792.0, 1196.0]])            # cross street (customer's X, stroke 1, extended)
ROAD_B = np.array([[596.0, 1290.0], [706.0, 1068.0]])            # main road (stroke 2, extended)
SCHOOL = CO.chaikin(np.array(ST['school_poly']), True, 2)
PARK = CO.chaikin(np.array(ST['park_poly']), True, 2)
STREET = np.array(ST['street'])
MEYDAN = np.array(ST['meydan_line'])[1]                          # lower end of the customer's pointer = the square
BELED = np.array(ST['belediye_line'])
def init_callouts():
    SPR['t_cross'] = CO.tag(['YOĞUN ARAÇ VE YAYA', 'TRAFİĞİNİN KESİŞİMİNDE'], 'car', 36, pointer='up')
    SPR['t_school'] = CO.tag(['ŞEHİT RAHMİ BEY İLKOKULU'], 'school', 34, pointer='down')
    SPR['t_park'] = CO.tag(['OTOPARK ALANI'], 'parking', 40, pointer=None)
    SPR['t_street'] = CO.street_text('Ş. RAHMİ BEY CAD.', 36)
    SPR['t_mey'] = CO.tag(['CUMHURİYET MEYDANI'], 'tree', 36, pointer='down')
    SPR['t_bel'] = CO.tag(['BELEDİYE BİNASI'], 'gov', 36, pointer='up')
def grow_both(P0, P1, C, p):
    """Segment P0-P1 grown outward from C by fraction p."""
    return C + (P0 - C) * p, C + (P1 - C) * p
def draw_callouts(frame, T, info, M):
    sh = info['shot']; s = info['src']
    if sh == 'YOL':
        t0, t1 = TL.SHOTS[1]['t0'], TL.SHOTS[1]['t1']; fo = float(np.clip((t1 - 0.04 - T) / 0.18, 0, 1))
        X = to_out(M, tpts('trk_yol', s, np.vstack([ROAD_A, ROAD_B, [XC]])))
        c = X[4]; pr = ease_out(np.clip((T - 30.28) / 0.50, 0, 1), 3)
        if pr > 0:
            for P0, P1 in ((X[0], X[1]), (X[2], X[3])):
                a, b = grow_both(P0, P1, c, pr)
                d0 = (a - c) / max(np.linalg.norm(a - c), 1e-6); d1 = (b - c) / max(np.linalg.norm(b - c), 1e-6)
                CO.glow_path(frame, np.array([a, b]), fo, width=6, arrows=((a, d0), (b, d1)) if pr > 0.35 else ())
        CO.ring(frame, c, T, 30.52, fo)
        CO.show_tag(frame, SPR['t_cross'], (c[0], c[1] + 40), T, 30.62, t1 - 0.04)
        sp = to_out(M, tpts('trk_yol', s, SCHOOL)); ps = ease_io(np.clip((T - 31.22) / 0.62, 0, 1))
        CO.draw_outline(frame, sp, ps, fill=ease_out(np.clip((T - 31.80) / 0.35, 0, 1)), alpha=fo)
        top = sp[np.argsort(sp[:, 1])[:6]].mean(0)
        CO.show_tag(frame, SPR['t_school'], (top[0], sp[:, 1].min() - 10), T, 31.80, t1 - 0.04)
    elif sh == 'PARK':
        t1 = TL.SHOTS[3]['t1']; fo = float(np.clip((t1 - 0.04 - T) / 0.18, 0, 1))
        pp = to_out(M, tpts('trk_park', s, PARK)); pr = ease_io(np.clip((T - 41.80) / 0.52, 0, 1))
        CO.draw_outline(frame, pp, pr, fill=ease_out(np.clip((T - 42.25) / 0.30, 0, 1)), alpha=fo)
        cen = pp.mean(0)
        CO.show_tag(frame, SPR['t_park'], (cen[0] + 20, cen[1] - 10), T, 42.36, t1 - 0.04, center=True)
        st = to_out(M, tpts('trk_park', s, STREET)); ps = ease_io(np.clip((T - 42.72) / 0.45, 0, 1))
        if ps > 0: CO.glow_path(frame, CO.partial(st, False, ps), fo, width=6, head=ps < 1)
        ang = np.degrees(np.arctan2(st[1][1] - st[0][1], st[1][0] - st[0][0]))
        mid = (st[0] + st[1]) / 2; nrm = np.array([np.sin(np.radians(ang)), -np.cos(np.radians(ang))])
        CO.show_rotated(frame, SPR['t_street'], mid + nrm * 36, ang, T, 42.98, t1 - 0.04)
    elif sh == 'MEY':
        t1 = TL.SHOTS[4]['t1']; fo = float(np.clip((t1 - 0.04 - T) / 0.18, 0, 1))
        P = to_out(M, tpts('trk_mey', s, np.vstack([[MEYDAN], BELED])))
        CO.ring(frame, P[0], T, 44.40, fo, r0=10, r1=46)
        CO.pin(frame, P[0], T, 44.18, t1 - 0.04)
        CO.show_tag(frame, SPR['t_mey'], (P[0][0], P[0][1] - 66), T, 44.52, t1 - 0.04)
        pb = ease_io(np.clip((T - 45.18) / 0.42, 0, 1))
        if pb > 0: CO.glow_path(frame, CO.partial(P[1:], False, pb), fo, width=6, head=pb < 1)
        bm = (P[1] + P[2]) / 2
        CO.show_tag(frame, SPR['t_bel'], (bm[0], bm[1] + 16), T, 45.58, t1 - 0.04)

def draw_captions(frame, T):
    ch = TL.CHUNKS
    for i, c in enumerate(ch):
        end = min(c[-1]['Te'] + 0.35, ch[i + 1][0]['T'] - 0.04) if i + 1 < len(ch) else c[-1]['Te'] + 0.45
        if c[0]['T'] - 0.05 <= T <= end:
            draw_caption(frame, T, c, cy=1585)

def draw_pin(frame, T, info, M):
    t_in = TL.vo('çarşının')['T'] - 0.25; t_out = B(18) - 0.02
    if not (t_in <= T <= t_out) or info.get('name') != 'D_B': return
    k = int(round(info['lt'] * FPS_SRC)); k = min(max(k, min(RP)), max(RP))
    x, y, _ = RP[k]; P = to_out(M, np.array([[x, y]]))[0]
    q = np.clip((t_out - T) / 0.3, 0, 1)
    p = np.clip((T - t_in) / 0.45, 0, 1)
    drop = (1 - ease_out_back(p, 2.0)) * 260
    # ground ring pulse
    ring_t = (T - t_in - 0.35) % 1.1
    if T > t_in + 0.35:
        r = 20 + 70 * ease_out(ring_t / 1.1, 2); a = (1 - ring_t / 1.1) * 0.9 * q
        lay = np.zeros((H, W), np.float32)
        cv2.ellipse(lay, (int(P[0] * 4), int(P[1] * 4)), (int(r * 4), int(r * 1.5)), 0, 0, 360, 1.0, 12, cv2.LINE_AA, shift=2)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.2) * a
        frame[:] = frame * (1 - lay[..., None]) + OR * lay[..., None]
    spr = SPR['pin']; h, w = spr.shape[:2]
    composite(frame, spr * float(min(1, p * 2) * q), P[0] - w / 2, P[1] - h + 28 - drop)
    lb = SPR['pin_lbl']; lh, lw = lb.shape[:2]
    lx = float(np.clip(P[0] - lw / 2, -20, W - lw + 20)); ly = P[1] - h - lh / 2 - 6 - drop
    slide_in(frame, lb, lx, ly, T, t_in + 0.35, t_out, dist=0)

_SCRIM = None
def scrim(frame, k, depth=0.42, extent=820):
    global _SCRIM
    if k <= 0: return
    if _SCRIM is None:
        yy = np.arange(H, dtype=np.float32)
        _SCRIM = np.clip(1 - yy / extent, 0, 1) ** 1.6
    frame *= (1 - depth * k * _SCRIM)[:, None, None]

def draw_titles(frame, T):
    v0 = TL.vo('Bir')['T']
    ks = fade_io(T, v0 - 0.2, TL.vo('çarşının')['T'] + 0.1, 0.4, 0.5) + fade_io(T, TL.vo('farklı')['T'] - 0.3, TL.vo('Otopark')['T'] - 0.8, 0.4, 0.4)
    scrim(frame, min(1.0, ks)); t_out1 = TL.vo('Serhat')['T'] - 0.35
    if v0 - 0.1 <= T <= t_out1:
        pop(frame, SPR['st1'], W / 2, 430, T, v0, t_out1)
        ka = TL.vo('konumla')['T']; kb = TL.vo('başlar')['T']
        a, b = SPR['st2a'], SPR['st2b']; gap = 28; tw = a.shape[1] + b.shape[1] - 80 + gap
        pop(frame, a, W / 2 - tw / 2 + a.shape[1] / 2 - 20, 540, T, ka - 0.08, t_out1, rise=50, back=2.2)
        pop(frame, b, W / 2 + tw / 2 - b.shape[1] / 2 + 20, 540, T, kb - 0.06, t_out1, rise=50, back=2.2)
    t1 = TL.vo('Serhat')['T'] - 0.05; t2 = TL.vo('çarşının')['T'] - 0.1
    if t1 <= T <= t2:
        q = np.clip((t2 - T) / 0.35, 0, 1)
        p1 = ease_out(np.clip((T - t1) / 0.6, 0, 1), 3)
        s = SPR['ti1']; place(frame, s, W / 2, 330 - 20 * (1 - q), scale=1.12 - 0.12 * p1, opacity=float(p1 * q))
        pl = ease_io(np.clip((T - t1 - 0.3) / 0.5, 0, 1))
        if pl > 0:
            half = 230 * pl; y = 372 - 20 * (1 - q)
            cv2.line(frame, (int(W / 2 - half), int(y)), (int(W / 2 + half), int(y)), tuple(float(c) * float(q) + 0 for c in OR), 3, cv2.LINE_AA)
        ti = TL.vo('İnci')['T'] - 0.12
        if T >= ti:
            p2 = ease_out(np.clip((T - ti) / 0.5, 0, 1), 3); s2 = SPR['ti2']; h2, w2 = s2.shape[:2]
            yb = 385 - 20 * (1 - q)                      # reveal from under the line (mask)
            y_top = yb + (1 - p2) * h2 * 0.9
            tmp = np.zeros((h2, w2, 4), np.float32); vis = int(np.clip(h2 - (y_top - yb), 0, h2))
            if vis > 0:
                tmp[:vis] = s2[:vis]
                composite(frame, tmp * float(q), W / 2 - w2 / 2, y_top - 6)

def draw_chips(frame, T):
    t_a, t_b, t_c = TL.vo('Geniş')['T'], TL.vo('yüksek')['T'], TL.vo('asma')['T']
    end = TL.vo('farklı')['T'] - 0.25
    for spr, t, y in [(SPR['c1'], t_a, 330), (SPR['c2'], t_b, 460), (SPR['c3'], t_c, 590)]:
        slide_in(frame, spr, 50, y, T, t - 0.08, end)
    t1, t2 = TL.vo('farklı')['T'] - 0.05, TL.vo('ferah')['T'] - 0.05; e2 = TL.vo('Otopark')['T'] - 0.9
    pop(frame, SPR['tg1'], W / 2, 420, T, t1, e2)
    pop(frame, SPR['tg2'], W / 2, 520, T, t2, e2, back=2.0)
    g1, g2 = TL.vo('güçlü')['T'] - 0.1, TL.vo('değerli')['T'] - 0.1; e4 = B(61) - 0.12
    for (a, b), t, y in [((SPR['g1a'], SPR['g1b']), g1, 700), ((SPR['g2a'], SPR['g2b']), g2, 850)]:
        tw = a.shape[1] + b.shape[1] - 80 + 26
        pop(frame, a, W / 2 - tw / 2 + a.shape[1] / 2 - 20, y, T, t, e4, back=2.0)
        pop(frame, b, W / 2 + tw / 2 - b.shape[1] / 2 + 20, y, T, t + 0.12, e4, back=2.0)

def draw_finale(frame, T):
    t0 = B(61)
    if T < t0: return
    # darkening band for legibility
    k = ease_io(np.clip((T - t0 - 0.2) / 1.0, 0, 1))
    if k > 0:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        dark = 0.30 + 0.38 * np.exp(-((yy - 0.47) / 0.30) ** 2)
        frame *= (1 - dark * k)
        frame += (dark * k) * CH * 0.6
    ts = TL.vo('Serhat', 1)['T'] - 0.15
    lg = SPR['logo']; hl, wl = lg.shape[:2]; sp = SPR['logo_split']
    cx, cy = W / 2, 800
    if T >= ts:
        p = np.clip((T - ts) / 0.95, 0, 1); pe = ease_io(p)
        top = lg[:sp].copy()
        # upward wipe with a glowing edge
        edge = sp * (1 - pe)
        yy = np.arange(sp, dtype=np.float32)[:, None]
        m = np.clip((yy - edge) / 14.0, 0, 1)
        glow = np.exp(-((yy - edge) / 10.0) ** 2) * (1 - p ** 3)
        spr = top * m[..., None]
        spr[..., :3] += (glow[..., None] * OR_L * 0.9) * (top[..., 3:4] > 0.05)
        composite(frame, spr, cx - wl / 2, cy - hl / 2)
        pw = np.clip((T - ts - 0.7) / 0.5, 0, 1)
        if pw > 0:
            place(frame, lg[sp:], cx, cy - hl / 2 + sp + (hl - sp) / 2 + 14 * (1 - ease_out(pw)), opacity=float(pw))
        # shine sweep across the logo
        ssw = (T - ts - 3.9) / 0.9
        if 0 <= ssw <= 1:
            xs = np.arange(wl, dtype=np.float32)[None, :]; ys = np.arange(hl, dtype=np.float32)[:, None]
            band = np.exp(-(((xs + ys * 0.35) - (-200 + ssw * (wl + 500))) / 60) ** 2)
            sh = lg[..., 3] * band * 0.55
            reg_x, reg_y = int(cx - wl / 2), int(cy - hl / 2)
            frame[reg_y:reg_y + hl, reg_x:reg_x + wl] += sh[..., None] * np.array([1.0, 0.95, 0.85], np.float32)
    tp = TL.vo('İnci', 1)['T'] - 0.1
    y_proj = cy + hl / 2 + 95
    pop(frame, SPR['f_proj'], W / 2, y_proj, T, tp, T_END + 1, rise=30, back=1.4, scale0=0.9)
    pl = ease_io(np.clip((T - tp - 0.25) / 0.5, 0, 1))
    if pl > 0:
        half = 170 * pl; y = int(y_proj + 72)
        cv2.line(frame, (int(W / 2 - half), y), (int(W / 2 + half), y), tuple(float(c) for c in OR), 3, cv2.LINE_AA)
    t1, t2 = TL.vo('çarşının', 1)['T'] - 0.08, TL.vo('ticaretin')['T'] - 0.08
    pop(frame, SPR['f_t1'], W / 2, y_proj + 150, T, t1, T_END + 1, rise=24, back=1.2, scale0=0.95)
    pop(frame, SPR['f_t2'], W / 2, y_proj + 222, T, t2, T_END + 1, rise=24, back=1.2, scale0=0.95)
    fo = np.clip((T - (T_END - 0.55)) / 0.55, 0, 1)
    if fo > 0: frame *= (1 - fo)

_VIG = None
def finish(img, T):
    global _VIG
    if _VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _VIG = (1 - 0.14 * np.clip((r - 0.6) / 0.8, 0, 1) ** 1.6)[..., None].astype(np.float32)
    return np.clip(img * _VIG, 0, 1)          # rev2: no synthetic grain (read as dust after WhatsApp compression)

def render_frame(fi):
    T = fi / FPS
    frame, M, info = render_plate(T)
    if info['shot'] == 'TAKE':
        draw_measures(frame, T, info, M)
        draw_pin(frame, T, info, M)
    elif info['shot'] in ('YOL', 'PARK', 'MEY'):
        draw_callouts(frame, T, info, M)
    draw_captions(frame, T)
    draw_titles(frame, T)
    draw_chips(frame, T)
    draw_finale(frame, T)
    return finish(np.clip(frame, 0, 1), T)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=''); ap.add_argument('--out', default='/tmp/jobs/serhat/out/v_nosound.mp4')
    ap.add_argument('--start', type=float, default=0.0); ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--outdir', default='/tmp/jobs/serhat/out/frames3')
    a = ap.parse_args(); init_gfx()
    if a.frames:
        os.makedirs(a.outdir, exist_ok=True)
        for f in sorted(int(round(float(x) * FPS)) for x in a.frames.split(',')):
            img = render_frame(f)
            cv2.imwrite(f'{a.outdir}/f_{f:04d}.png', (img[..., ::-1] * 255 + 0.5).astype(np.uint8)); print('frame', f, flush=True)
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n0 = int(round(a.start * FPS)); n1 = int(round((a.end or T_END) * FPS))
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    t0 = time.time(); rng = np.random.default_rng(1)
    for f in range(n0, n1):
        img = render_frame(f)
        d = rng.uniform(-0.5, 0.5, (H, W, 1)).astype(np.float32) / 255.0
        enc.stdin.write(((img + d) * 255 + 0.5).clip(0, 255).astype(np.uint8).tobytes())
        if f % 30 == 0: print(f'{f}/{n1} {time.time() - t0:.0f}s', flush=True)
    enc.stdin.close(); enc.wait(); print('DONE', a.out, f'{time.time() - t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
