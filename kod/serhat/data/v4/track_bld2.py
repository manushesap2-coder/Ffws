"""Stream-track a point (building) through a 1296x2304 intermediate with a local partial-affine LK model.
usage: track_bld.py NAME LT0 LT1 X Y OUT [BOX]   (LT1 < LT0 -> backward, loads the range into memory)"""
import sys, json, subprocess, numpy as np, cv2
FPS = 60000 / 1001
name, lt0, lt1, px, py, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
BOX = float(sys.argv[7]) if len(sys.argv) > 7 else 360.0
S = 0.5; W, H = 1296, 2304; w, h = int(W * S), int(H * S)
def reader(t0, n):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{t0:.5f}', '-i', f'/tmp/jobs/serhat/inter/{name}.mp4', '-vf', f'scale={w}:{h}',
                          '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], stdout=subprocess.PIPE, bufsize=10 ** 7)
    for _ in range(n):
        b = p.stdout.read(w * h)
        if len(b) < w * h: break
        yield np.frombuffer(b, np.uint8).reshape(h, w)
    p.kill()
lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
BW, BH = 250.0, 110.0   # tight box on the roof slab (full-res px at scale 1)
def step(ga, gb, P, box):
    sc = box / BOX
    m = np.zeros_like(ga); x, y = P * S
    bx, by = max(BW * sc, 90) * S / 2, max(BH * sc, 50) * S / 2
    m[int(max(0, y - by)):int(min(h, y + by)), int(max(0, x - bx)):int(min(w, x + bx))] = 255
    pts = cv2.goodFeaturesToTrack(ga, 300, 0.003, 3, mask=m, blockSize=5)
    if pts is None or len(pts) < 6: return P, 1.0, 0
    nb, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, **lk)
    bk, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, nb, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((bk - pts).reshape(-1, 2), axis=1) < 0.4)
    if ok.sum() < 6: return P, 1.0, int(ok.sum())
    A, inl = cv2.estimateAffinePartial2D(pts[ok], nb[ok], method=cv2.RANSAC, ransacReprojThreshold=0.8)
    q = A @ np.array([x, y, 1.0])
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    return q / S, s, int(inl.sum())
f_ref = int(round(lt0 * FPS)); f_end = int(round(lt1 * FPS)); d = 1 if f_end > f_ref else -1
n = abs(f_end - f_ref) + 1
if d == 1: frames = reader(f_ref / FPS, n)
else: frames = iter(list(reader(f_end / FPS, n))[::-1])
P = np.array([px, py]); sc = 1.0; res = {f_ref: [px, py, 1.0, 999]}; prev = next(frames); k = f_ref; low = []
for g in frames:
    P, s, ni = step(prev, g, P, BOX * sc); sc *= s; k += d; prev = g
    res[k] = [float(P[0]), float(P[1]), float(sc), ni]
    if ni < 12: low.append(k)
json.dump(dict(name=name, fps=FPS, pts={str(k): v for k, v in sorted(res.items())}), open(out, 'w'))
print(out, 'frames', len(res), 'low-inlier frames', len(low), low[:5], 'final', res[k])
