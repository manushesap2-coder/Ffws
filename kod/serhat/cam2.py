"""Rev2 virtual camera for the interior one-take: close on the presenter while she talks, wide for the measurements."""
import json, numpy as np, sys
sys.path.insert(0, '/tmp/jobs/serhat/lib')
import tl_s as TL
_d = [(t, r) for t, r in json.load(open('/tmp/jobs/serhat/rev2/woman.json')) if r]
_S = np.array([t for t, _ in _d])
_H = np.array([r['bot'] - r['top'] for _, r in _d], float)
_CX = np.array([r['cx'] for _, r in _d], float)
_CY = np.array([(r['top'] + r['bot']) / 2 for _, r in _d], float)
def _smooth(v, sig=0.8):
    out = np.empty_like(v)
    for i, s in enumerate(_S):
        k = np.exp(-0.5 * ((_S - s) / sig) ** 2); out[i] = (v * k).sum() / k.sum()
    return out
_Hs, _CXs, _CYs = np.exp(_smooth(np.log(_H))), _smooth(_CX, 1.0), _smooth(_CY)
def woman_at(s):
    return (float(np.interp(s, _S, _Hs)), float(np.interp(s, _S, _CXs)), float(np.interp(s, _S, _CYs)))
def sstep(a, b, t):
    u = float(np.clip((t - a) / (b - a), 0, 1)); return u * u * (3 - 2 * u)
def z_wide(T):
    if T < 4.5: return 1.0
    if T < 13.0: return 1.0 + 0.16 * sstep(4.5, 13.0, T)
    return 1.16
T_OUT1, T_IN1, T_IN2, T_EXIT0, T_EXIT1 = 6.15, 6.95, 12.75, 17.6, 21.1
def cam(T):
    """-> (z, cx, cy) in D_A intermediate coords (1296x2304), z=1 is the full frame width."""
    s, _ = TL.take_src(T)
    h, wx, wy = woman_at(s)
    if T < 9.5:
        zt = float(np.clip(0.56 * 1920 / (0.8333 * h), 1.2, 2.0))
    else:
        zt = float(np.clip(0.40 * 1920 / (0.8333 * h), 1.2, 2.1))
    if T < T_EXIT0:
        w = 1 - sstep(T_OUT1, T_IN1, T) + sstep(T_IN2, T_IN2 + 0.8, T)
        zw, cw = z_wide(T), (680.0, 1150.0)
    else:
        w = 1 - sstep(T_EXIT0, T_EXIT1, T)
        zw, cw = 1.0, (648.0, 1152.0)
    z = float(np.exp((1 - w) * np.log(zw) + w * np.log(zt)))
    so = 0.8333 * z
    tx, ty = wx, wy + (960 - 860) / so
    cx = (1 - w) * cw[0] + w * tx; cy = (1 - w) * cw[1] + w * ty
    hw, hh = 648 / z, 1152 / z
    return z, float(np.clip(cx, hw, 1296 - hw)), float(np.clip(cy, hh, 2304 - hh))
if __name__ == '__main__':
    for T in [0, 1, 3, 5, 6.15, 6.5, 6.95, 9, 12.75, 13.2, 13.55, 15, 17, 17.6, 18.5, 19.5, 20.5, 21.1, 21.3]:
        z, cx, cy = cam(T); s, _ = TL.take_src(T); h, wx, wy = woman_at(s)
        print(f'T {T:5.2f} src {s:5.2f} z {z:4.2f} c ({cx:6.1f},{cy:6.1f})  woman h_out {0.8333*z*h/1920*100:4.1f}%')
