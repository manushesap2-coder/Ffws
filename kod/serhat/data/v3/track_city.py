"""Ground-plane homography chain (ref frame -> every frame) for a drone city shot. Coordinates: 1296x2304 intermediate."""
import numpy as np, cv2, subprocess, json, sys
FPS = 60000 / 1001
t0, t1, tref, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
sky = float(sys.argv[5]) if len(sys.argv) > 5 else 0.26
def loc(t):
    if t < 38.1: return 'D_A', 0.0
    if t < 116.1: return 'D_B', 38.0
    return 'D_C', 116.0
name, off = loc(t0)
S = 0.5; W, H = 1296, 2304; w, h = int(W * S), int(H * S)
f0, f1, fr = int(round((t0 - off) * FPS)), int(round((t1 - off) * FPS)), int(round((tref - off) * FPS))
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{f0 / FPS:.5f}', '-i', f'/tmp/jobs/serhat/inter/{name}.mp4', '-vf', f'scale={w}:{h}',
                      '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], stdout=subprocess.PIPE)
frames = []
while len(frames) < f1 - f0 + 1:
    b = p.stdout.read(w * h)
    if len(b) < w * h: break
    frames.append(np.frombuffer(b, np.uint8).reshape(h, w))
p.kill()
mask = np.zeros((h, w), np.uint8); mask[int(sky * h):h - 6, 6:w - 6] = 255
lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
def step(a, b):
    pts = cv2.goodFeaturesToTrack(a, 900, 0.008, 7, mask=mask, blockSize=7)
    nb, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **lk)
    bk, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nb, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((bk - pts).reshape(-1, 2), axis=1) < 0.5)
    Hm, inl = cv2.findHomography(pts[ok], nb[ok], cv2.RANSAC, 1.0)
    return Hm, int(inl.sum())
Sm = np.diag([S, S, 1.0]); Si = np.linalg.inv(Sm)
Hs = {fr: np.eye(3)}; ninl = []
for d in (1, -1):
    acc = np.eye(3); k = fr
    while f0 <= k + d <= f1 and 0 <= k + d - f0 < len(frames):
        Hm, n = step(frames[k - f0], frames[k + d - f0]); ninl.append(n)
        acc = Hm @ acc; k += d; Hs[k] = acc.copy()
json.dump(dict(fps=FPS, off=off, name=name, tref=tref, H={str(k): (Si @ v @ Sm).tolist() for k, v in sorted(Hs.items())}), open(out, 'w'))
print(out, 'frames', len(Hs), 'min/median inliers', min(ninl), int(np.median(ninl)))
