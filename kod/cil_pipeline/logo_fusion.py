"""Multi-frame logo extraction from footage: segment a bright warm (gold) mark per frame, rectify it to a
canonical rhombus from its extreme points, ECC-align all frames to a reference and take the median.
Usage: python logo_fusion.py "<16-bit png glob>" out.npy [--roi y0,y1,x0,x1] [--ref IDX]"""
import glob, argparse, numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument('frames'); ap.add_argument('out')
ap.add_argument('--roi', default='1500,2800,500,2100')
ap.add_argument('--ref', type=int, default=-1)
a = ap.parse_args()
y0, y1, x0, x1 = map(int, a.roi.split(','))
Wc = Hc = 1000
hgt = 900.0; wid = hgt * 0.93; cx = Wc / 2; ty = 50; fy = ty + 0.55 * hgt
dst = np.float32([[cx, ty], [cx, ty + hgt], [cx - wid / 2, fy], [cx + wid / 2, fy]])

def score_map(im):
    b, g, r = im[..., 0], im[..., 1], im[..., 2]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    return np.clip((lum - 0.38) / 0.25, 0, 1) * np.clip(((r - b) - 0.05) / 0.15, 0, 1)

warps = []
for f in sorted(glob.glob(a.frames)):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    sc = score_map(im)
    roi = np.zeros_like(sc); roi[y0:y1, x0:x1] = 1
    m = ((sc * roi) > 0.35).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((25, 25), np.uint8)), 8)
    cand = [(st[i][4], i) for i in range(1, n) if 200 < st[i][2] < 700 and 250 < st[i][3] < 700]
    if not cand:
        print('skip', f); continue
    _, i = max(cand); x, y, w, h, _ = st[i]
    keep = (lab == i).astype(np.float32) * m
    ys, xs = np.nonzero(keep[y:y + h, x:x + w])
    pts = np.float32([(xs[ys == ys.min()].mean() + x, ys.min() + y), (xs[ys == ys.max()].mean() + x, ys.max() + y),
                      (xs.min() + x, ys[xs == xs.min()].mean() + y), (xs.max() + x, ys[xs == xs.max()].mean() + y)])
    M = cv2.getPerspectiveTransform(pts, dst)
    warps.append(cv2.warpPerspective(sc * cv2.dilate(keep, np.ones((5, 5), np.uint8)), M, (Wc, Hc), flags=cv2.INTER_CUBIC))
ri = a.ref if a.ref >= 0 else len(warps) // 2
ref = warps[ri]
aligned = [ref]
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
for k, w_ in enumerate(warps):
    if k == ri: continue
    try:
        cc, Wm = cv2.findTransformECC(cv2.GaussianBlur(ref, (0, 0), 2), cv2.GaussianBlur(w_, (0, 0), 2),
                                      np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        if cc > 0.6:   # drop frames that fail to register (motion blur, occlusion)
            aligned.append(cv2.warpPerspective(w_, Wm, (Wc, Hc), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP))
    except cv2.error:
        pass
np.save(a.out, np.median(np.stack(aligned), 0))
print('frames used', len(aligned))
