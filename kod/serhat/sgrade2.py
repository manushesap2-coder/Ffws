"""Rev2 interior look: local tone mapping (guided-filter base/detail) + clean blacks + luma sharpening. No grain."""
import numpy as np, cv2
def luma(x): return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
def _box(x, r): return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
def guided_self(I, r, eps):
    m = _box(I, r); v = _box(I * I, r) - m * m
    a = v / (v + eps); b = m - a * m
    return _box(a, r) * I + _box(b, r)
def smooth_curve(x, contrast, pivot):
    x = np.clip(x, 0, 1)
    return np.where(x < pivot, pivot * (x / pivot) ** contrast, 1 - (1 - pivot) * ((1 - x) / (1 - pivot)) ** contrast)
P_INT2 = dict(ev=0.22, ks=0.50, kh=0.22, mid=0.18, detail=1.28, r=22, eps=0.10, black=0.022, contrast=1.15, pivot=0.40,
              sat=1.10, vib=0.35, gain=[1.02, 1.0, 0.975], sharp=0.55, glare=1.0)
def grade_int2(x, p=P_INT2, sharp=None):
    H, W = x.shape[:2]
    lin = np.clip(x, 0, 1) ** 2.2
    Y = np.maximum(luma(lin), 1e-5)
    L = np.log2(Y)
    Ls = cv2.resize(L, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    base = cv2.resize(guided_self(Ls, p['r'], p['eps']), (W, H), interpolation=cv2.INTER_LINEAR)
    mid = np.log2(p['mid'])
    d = base - mid
    base2 = mid + np.where(d < 0, d * (1 - p['ks']), d * (1 - p['kh']))
    L2 = base2 + (L - base) * p['detail'] + p['ev']
    # sunlit floor patch: bright, warm, lower frame -> pull down (keyed on the smooth base, so no edges)
    g = p.get('glare', 0.0)
    if g > 0:
        bs = cv2.resize(base, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        ls = cv2.resize(lin, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        warm = np.clip(((ls[..., 0] - ls[..., 2]) / np.maximum(ls[..., 1], 1e-4) - 0.04) / 0.16, 0, 1)
        yy = np.linspace(0, 1, H // 8, dtype=np.float32)[:, None]
        m = np.clip((bs - np.log2(0.22)) / 1.0, 0, 1) * warm * np.clip((yy - 0.52) / 0.16, 0, 1)
        m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 6), (W, H), interpolation=cv2.INTER_LINEAR)
        L2 = L2 - g * m
        GL = m
        grade_int2.last_mask = m
    else:
        GL = None
    lin2 = lin * (2.0 ** (L2 - L))[..., None]
    x2 = np.clip(lin2, 0, None) ** (1 / 2.2)
    # soft shoulder for anything pushed above 1
    over = np.maximum(x2 - 0.85, 0); x2 = np.where(x2 > 0.85, 0.85 + over / (1 + over / 0.15), x2)
    x2 = np.clip((x2 - p['black']) / (1 - p['black']), 0, 1)
    x2 = smooth_curve(x2, p['contrast'], p['pivot'])
    Yd = luma(x2)[..., None]; c = x2 - Yd
    chroma = np.sqrt((c ** 2).sum(-1, keepdims=True))
    k = p['sat'] * (1 + p['vib'] * np.clip(0.2 - chroma, 0, 0.2) / 0.2) / (1 + p['vib'])
    x2 = Yd + c * k
    if GL is not None:
        Yd = luma(x2)[..., None]; x2 = Yd + (x2 - Yd) * (1 - 0.35 * GL[..., None])
    x2 = x2 * np.array(p['gain'], np.float32)
    amt = p['sharp'] if sharp is None else sharp
    if amt > 0:
        Yd = luma(x2); det = Yd - cv2.GaussianBlur(Yd, (0, 0), 0.9)
        m = np.clip((np.abs(det) - 0.002) / 0.008, 0, 1)
        x2 = x2 + (amt * det * m)[..., None]
    return np.clip(x2, 0, 1).astype(np.float32)
