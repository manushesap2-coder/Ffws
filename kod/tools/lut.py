import numpy as np
def load_cube(path):
    size = None; rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith('#'): continue
        if ln.startswith('LUT_3D_SIZE'): size = int(ln.split()[1]); continue
        if ln[0].isalpha(): continue
        rows.append([float(v) for v in ln.split()[:3]])
    a = np.array(rows, np.float32).reshape(size, size, size, 3)  # [b][g][r] order in file (r fastest)
    return a
def apply_cube(img, lut):
    """img float32 HxWx3 RGB in [0,1]; trilinear."""
    n = lut.shape[0]
    x = np.clip(img, 0, 1) * (n - 1)
    i0 = np.floor(x).astype(np.int32); i0 = np.minimum(i0, n - 2); f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    def L(b, g, r): return lut[b, g, r]
    c000 = L(b0, g0, r0); c100 = L(b0, g0, r0 + 1); c010 = L(b0, g0 + 1, r0); c110 = L(b0, g0 + 1, r0 + 1)
    c001 = L(b0 + 1, g0, r0); c101 = L(b0 + 1, g0, r0 + 1); c011 = L(b0 + 1, g0 + 1, r0); c111 = L(b0 + 1, g0 + 1, r0 + 1)
    c00 = c000 * (1 - fr) + c100 * fr; c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr; c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg; c1 = c01 * (1 - fg) + c11 * fg
    return c0 * (1 - fb) + c1 * fb
