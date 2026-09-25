"""Segment the poured grobeton strip in D2's first used frame and track it (feature homography) through the shot."""
import numpy as np, cv2, subprocess, json, sys
sys.path.insert(0, '/tmp/edit/pipeline')
from config import DW, DH
FPS = 60000 / 1001
SRC = '/tmp/edit/inter/D2.mov'
s0, s1 = 0.40, 3.60                     # source window (s) covering the shot incl. transition margins
w, h = DW // 2, DH // 2
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{s0}', '-i', SRC, '-t', f'{s1 - s0}', '-vf', f'scale={w}:{h}',
                      '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
frames = []
while True:
    b = p.stdout.read(w * h * 3)
    if len(b) < w * h * 3: break
    frames.append(np.frombuffer(b, np.uint8).reshape(h, w, 3))
f0 = frames[0]
hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
Hh = hsv[..., 0].astype(int)
m = ((Hh >= 98) & (Hh <= 116) & (S > 45) & (S < 95) & (V > 62) & (V < 115)).astype(np.uint8)   # wet concrete reads blue-grey in log
m[:h // 3] = 0; m[int(h * 0.80):] = 0; m[:, :int(w * 0.55)] = 0
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
i = 1 + np.argmax(st[1:, 4])
blob = (lab == i).astype(np.uint8)
cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea)
poly = np.float32([[34, 652], [572, 424], [614, 432], [638, 866], [40, 882]])   # excavation footprint (manual, first frame)
print('area', st[i, 4], 'poly', poly.tolist())
vis = f0.copy(); cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 200, 255), 3)
# track: features outside the moving boom, homography to first frame
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
mask = np.full_like(g0, 255); mask[:int(h * 0.30)] = 0
pts0 = cv2.goodFeaturesToTrack(g0, 800, 0.01, 8, mask=mask)
Hs = [np.eye(3).tolist()]
prev, pprev = g0, pts0
Hacc = np.eye(3)
for k in range(1, len(frames)):
    g = cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY)
    nxt, stt, _ = cv2.calcOpticalFlowPyrLK(prev, g, pprev, None, winSize=(21, 21), maxLevel=3)
    good0 = pprev[stt[:, 0] == 1]; good1 = nxt[stt[:, 0] == 1]
    Hk, inl = cv2.findHomography(good0, good1, cv2.RANSAC, 2.0)
    Hacc = Hk @ Hacc
    Hs.append(Hacc.tolist())
    prev = g
    pprev = cv2.goodFeaturesToTrack(g, 800, 0.01, 8, mask=mask) if k % 10 == 0 else good1[inl[:, 0] == 1].reshape(-1, 1, 2)
polys = []
for Hk in Hs:
    pp = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), np.array(Hk)).reshape(-1, 2) * 2.0   # to intermediate coords
    polys.append(pp.tolist())
json.dump({'s0': s0, 'fps': FPS, 'polys': polys}, open('/tmp/edit/work/d2_outline.json', 'w'))
lastp = (np.array(polys[-1]) / 2).astype(np.int32)
vis2 = frames[-1].copy(); cv2.polylines(vis2, [lastp], True, (0, 200, 255), 3)
cv2.imwrite('/tmp/hoplite/workspace/.scratch/d2_track.jpg', np.hstack([vis, vis2])[::2, ::2])
print('frames', len(frames))
