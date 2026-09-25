"""Reels effects: building drop onto a tracked top-down drone plate (perspective fall, shadow, dust, shake),
facade fly-through into an interior render, and small sprites (pill, clock icon, logo). See README "Efekt modülleri"."""
import sys, json, numpy as np, cv2
sys.path.insert(0, '/tmp/edit/pipeline')
from gfx import *
W, H = 1080, 1920

def premul(rgb, a):
    return np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)

def over(dst, spr):
    dst *= (1 - spr[..., 3:4]); dst += spr[..., :3]

# ------------------------------------------------------------------ dust burst
class Dust:
    """Soft debris cloud spawned along a screen-space footprint at impact; lives in quarter-res for speed."""
    def __init__(self, poly, seed=5, n=420, dur=1.9, color=(0.80, 0.74, 0.64)):
        rng = np.random.default_rng(seed)
        P = np.asarray(poly, np.float32); c = P.mean(0)
        edges = np.roll(P, -1, 0) - P; L = np.linalg.norm(edges, axis=1); cum = np.concatenate([[0], np.cumsum(L)])
        d = rng.uniform(0, cum[-1], n); i = np.clip(np.searchsorted(cum, d) - 1, 0, len(P) - 1)
        pts = P[i] + edges[i] * ((d - cum[i]) / np.maximum(L[i], 1e-6))[:, None]
        out = pts - c; out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
        ang = rng.normal(0, 0.45, n); ca, sa = np.cos(ang), np.sin(ang)
        out = np.stack([out[:, 0] * ca - out[:, 1] * sa, out[:, 0] * sa + out[:, 1] * ca], 1)
        self.p0, self.v = pts, out * rng.uniform(260, 900, (n, 1))
        self.r0, self.gr = rng.uniform(8, 18, n), rng.uniform(40, 110, n)
        self.a0 = rng.uniform(0.45, 0.85, n); self.life = rng.uniform(0.7, 1.0, n) * dur
        self.col = np.array(color, np.float32); self.dur = dur
    def draw(self, img, t):
        if t < 0 or t > self.dur: return
        tau = 0.42
        p = self.p0 + self.v * tau * (1 - np.exp(-t / tau))
        r = self.r0 + self.gr * (1 - np.exp(-t / 0.5))
        a = self.a0 * np.clip(1 - t / self.life, 0, 1) ** 1.5 * np.clip(t / 0.05, 0, 1)
        q = 4
        buf = np.zeros((H // q, W // q), np.float32)
        for (x, y), rr, aa in zip(p, r, a):
            if aa > 0.005:
                cv2.circle(buf, (int(x / q), int(y / q)), max(1, int(rr / q)), float(aa), -1, lineType=cv2.LINE_AA)
        buf = cv2.GaussianBlur(buf, (0, 0), 3.0)
        m = np.clip(cv2.resize(buf, (W, H), interpolation=cv2.INTER_LINEAR) * 1.3, 0, 0.9)[..., None]
        img[:] = img * (1 - m) + self.col * m

# ------------------------------------------------------------------ building drop
class BuildingDrop:
    """Roofs from the top-down render warped onto the tracked plate; falling = shrinking toward the ground (top-down camera)."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.blocks = []
        for b in cfg['blocks']:
            rgba = cv2.imread(b['sprite'], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            rgb, a = rgba[..., 2::-1], rgba[..., 3]
            rgb = self.match(rgb)
            Hb = np.array(b['H'], np.float64)                                          # sprite px -> plate px at t_ref
            if 'scale' in b:                                                           # enlarge about a ground anchor (plate coords)
                k, (ax, ay) = b['scale'], b['anchor']
                Hb = np.array([[k, 0, ax * (1 - k)], [0, k, ay * (1 - k)], [0, 0, 1]]) @ Hb
            self.blocks.append(dict(spr=premul(rgb, a), alpha=a, H=Hb,
                                    t0=b['t0'], t1=b['t1'], poly=np.array(b['poly'], np.float32)))
        self.track = [np.array(h, np.float64) for h in json.load(open(cfg['track']))['H']]            # plate(ref) -> plate(k)
        self.kref = cfg['kref']
        self.dust = {}
    @staticmethod
    def match(rgb):
        # render roofs are cool/flat; warm them up and add contrast to sit in the sunny plate
        x = np.clip((rgb - 0.06) / 0.88, 0, 1)
        x = x * np.array([1.06, 1.02, 0.93], np.float32)
        Y = (x * [0.2126, 0.7152, 0.0722]).sum(-1, keepdims=True)
        x = Y + (x - Y) * 1.15
        x = 0.5 + (x - 0.5) * 1.12
        return np.clip(cv2.GaussianBlur(x, (0, 0), 0.8), 0, 1).astype(np.float32)
    def H_at(self, k):
        k = int(np.clip(round(k), 0, len(self.track) - 1))
        return self.track[k]
    def draw(self, img, T, k_plate, A):
        """img: output frame; k_plate: plate frame index; A: 3x3 plate->output affine."""
        Ht = self.H_at(k_plate)
        shake = np.zeros(2)
        for j, b in enumerate(self.blocks):
            if T < b['t0']: continue
            G = A @ Ht @ b['H']                                    # sprite -> output at ground level
            poly = cv2.perspectiveTransform(b['poly'][None], G)[0]
            c = poly.mean(0)
            u = np.clip((T - b['t0']) / (b['t1'] - b['t0']), 0, 1)
            def S(uu):
                z = 0.70 * (1 - uu ** 2)                           # height as fraction of camera altitude (gravity: z ~ 1 - u^2)
                return 1.0 / (1.0 - z)
            s = S(u)
            # shadow on the ground: offset along the sun direction, softer and lighter the higher the roof
            hgt = (s - 1.0) / (S(0) - 1.0)
            sh = cv2.warpPerspective(b['alpha'], G, (W, H), flags=cv2.INTER_LINEAR)
            off = np.array([-22.0, 16.0]) * (1 + 7 * hgt)
            Msh = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
            sh = cv2.warpAffine(sh, Msh, (W, H))
            sig = 3 + 40 * hgt
            sh = cv2.GaussianBlur(sh, (0, 0), sig) * (0.55 - 0.35 * hgt) * min(1.0, (T - b['t0']) / 0.12)
            img[:] = img * (1 - sh[..., None])
            # roof with motion blur (sub-frame samples of the fall)
            n = 1 if u >= 1 else 5
            acc = None
            for i in range(n):
                uu = max(0.0, u - i * (1.0 / 30) / (b['t1'] - b['t0']) / n)
                ss = S(uu)
                Sm = np.array([[ss, 0, c[0] * (1 - ss)], [0, ss, c[1] * (1 - ss)], [0, 0, 1]])
                w_ = cv2.warpPerspective(b['spr'], Sm @ G, (W, H), flags=cv2.INTER_LINEAR)
                acc = w_ if acc is None else acc + w_
            spr = acc / n
            spr *= min(1.0, (T - b['t0']) / 0.08)
            over(img, spr)
            # impact
            ti = T - b['t1']
            if ti >= 0:
                if j not in self.dust: self.dust[j] = Dust(poly, seed=11 + j)
                self.dust[j].draw(img, ti)
                if ti < 0.4:
                    amp = 20 * np.exp(-ti / 0.12)
                    rng = np.random.default_rng(int(T * 1000))
                    shake += rng.uniform(-1, 1, 2) * amp
                if ti < 0.16:
                    img[:] = img + (1 - img) * 0.28 * (1 - ti / 0.16)
        return shake

# ------------------------------------------------------------------ door fly-through
class FlyThrough:
    """Push into the lit wood-slat column; the slat gaps open up and the lobby appears behind as we pass through."""
    def __init__(self, facade, lobby, portal, facade_anchor0, lobby_anchor, grade_fn):
        self.F = grade_fn(cv2.imread(facade)[..., ::-1].astype(np.float32) / 255.0)
        self.L = grade_fn(cv2.imread(lobby)[..., ::-1].astype(np.float32) / 255.0)
        x0, y0, x1, y1 = portal
        Y = (self.F * [0.2126, 0.7152, 0.0722]).sum(-1)
        gap = np.clip((0.30 - Y) / 0.15, 0, 1)                   # dark gaps between slats
        rect = np.zeros(Y.shape, np.float32); cv2.rectangle(rect, (x0, y0), (x1, y1), 1.0, -1)
        rect = cv2.GaussianBlur(rect, (0, 0), 14)
        self.hole = (rect * (0.45 + 0.55 * cv2.GaussianBlur(gap, (0, 0), 1.2))).astype(np.float32)
        self.pc = np.array([(x0 + x1) / 2, (y0 + y1) / 2]); self.pw, self.ph = x1 - x0, y1 - y0
        self.a0 = np.array(facade_anchor0, np.float64); self.la = np.array(lobby_anchor, np.float64)
        self.s0 = H / self.F.shape[0]
        self.s1 = max(W / self.pw, H / self.ph) * 1.10
        self.sl_end = max(W / self.L.shape[1], H / self.L.shape[0]) * 1.03
    def draw(self, T, Ta, Tb, Tc, Td):
        if T < Tc:
            if T < Tb:
                u = (T - Ta) / (Tb - Ta); lf = np.log(self.s0) + np.log(1.16) * ease_io(u); ua = 0.25 * ease_io(u); uf = 0.0
            else:
                u = (T - Tb) / (Tc - Tb); e = u ** 2.4; uf = u
                lf = np.log(self.s0 * 1.16) + (np.log(self.s1) - np.log(self.s0 * 1.16)) * e; ua = 0.25 + 0.75 * ease_io(u)
            sf = float(np.exp(lf)); anc = self.a0 + (self.pc - self.a0) * ua
            sl = self.sl_end * (sf / self.s1) ** 0.55
            Mf = np.float32([[sf, 0, W / 2 - anc[0] * sf], [0, sf, H / 2 - anc[1] * sf]])
            pcs = Mf @ np.array([self.pc[0], self.pc[1], 1.0])
            Ml = np.float32([[sl, 0, pcs[0] - self.la[0] * sl], [0, sl, pcs[1] - self.la[1] * sl]])
            lob = cv2.warpAffine(self.L, Ml, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            fac = cv2.warpAffine(self.F, Mf, (W, H), flags=cv2.INTER_CUBIC if sf < 3 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            h = np.clip((uf - 0.05) / 0.55, 0, 1) ** 1.3             # slats open during the fast push
            hole = cv2.warpAffine(self.hole, Mf, (W, H), flags=cv2.INTER_LINEAR)
            fa = np.clip(1 - h * hole, 0, 1) * (1 - np.clip((uf - 0.72) / 0.28, 0, 1))
            fa = fa[..., None]
            img = fac * fa + lob * (1 - fa)
            speed = 0.0 if T < Tb else 0.15 * (2.4 * u ** 1.4)
            return img, min(speed, 0.20)
        u = (T - Tc) / (Td - Tc)
        sl = self.sl_end * (1.0 + 0.10 * ease_out(u, 2))
        Ml = np.float32([[sl, 0, W / 2 - self.la[0] * sl], [0, sl, H / 2 - self.la[1] * sl]])
        return cv2.warpAffine(self.L, Ml, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), 0.0

# ------------------------------------------------------------------ small sprites
def clock_icon(size, color=(1.0, 0.82, 0.35)):
    S = size * 2; im = np.zeros((S, S), np.float32)
    cv2.circle(im, (S // 2, S // 2), int(S * 0.42), 1.0, max(2, S // 12), lineType=cv2.LINE_AA)
    cv2.line(im, (S // 2, S // 2), (S // 2, int(S * 0.22)), 1.0, max(2, S // 12), lineType=cv2.LINE_AA)
    cv2.line(im, (S // 2, S // 2), (int(S * 0.70), S // 2), 1.0, max(2, S // 14), lineType=cv2.LINE_AA)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return premul(np.broadcast_to(np.array(color, np.float32), (size, size, 3)), im)

def pill(content, pad_x=34, pad_y=18, fill=(0.05, 0.06, 0.07), fill_a=0.78, border=(1.0, 0.82, 0.35)):
    """Rounded translucent pill behind a premultiplied sprite."""
    h, w = content.shape[:2]
    Hh, Ww = h + 2 * pad_y, w + 2 * pad_x
    m = np.zeros((Hh, Ww), np.float32); r = Hh // 2
    cv2.rectangle(m, (r, 0), (Ww - r, Hh), 1, -1)
    cv2.circle(m, (r, r), r, 1, -1, lineType=cv2.LINE_AA); cv2.circle(m, (Ww - r, r), r, 1, -1, lineType=cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    edge = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
    out = premul(np.broadcast_to(np.array(fill, np.float32), (Hh, Ww, 3)), m * fill_a)
    over(out[..., :3], premul(np.broadcast_to(np.array(border, np.float32), (Hh, Ww, 3)), edge * 0.9))
    out[..., 3] = np.maximum(out[..., 3], edge * 0.9)
    sub = out[pad_y:pad_y + h, pad_x:pad_x + w]
    sub[..., :3] = sub[..., :3] * (1 - content[..., 3:4]) + content[..., :3]
    sub[..., 3] = sub[..., 3] * (1 - content[..., 3]) + content[..., 3]
    return out

def hstack_sprites(sprites, gap=16):
    h = max(s.shape[0] for s in sprites); w = sum(s.shape[1] for s in sprites) + gap * (len(sprites) - 1)
    out = np.zeros((h, w, 4), np.float32); x = 0
    for s in sprites:
        y = (h - s.shape[0]) // 2; out[y:y + s.shape[0], x:x + s.shape[1]] = s; x += s.shape[1] + gap
    return out

def logo_sprite(path, width):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    a = im[..., 3]; ys, xs = np.nonzero(a > 0.02)
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = width / im.shape[1]
    im = cv2.resize(im, (width, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return premul(im[..., 2::-1], im[..., 3])
