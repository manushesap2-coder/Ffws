"""Lock overlays to the ground of a drone plate: LK optical flow + RANSAC homography per frame pair, chained from a
reference frame. Writes {"H": [3x3 per frame: reference-frame coords -> frame k coords], "kref": k}.
Usage: python plate_track.py plate.mov WIDTH HEIGHT KREF out.json   (tracking runs at half resolution)"""
import json, subprocess, sys
import numpy as np, cv2

src, W, H, kref, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
w, h = W // 2, H // 2
proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-vf', f'scale={w}:{h}', '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
                        stdout=subprocess.PIPE)
frames = []
while True:
    buf = proc.stdout.read(w * h)
    if len(buf) < w * h:
        break
    frames.append(np.frombuffer(buf, np.uint8).reshape(h, w))


def pair(a, b):
    pts = cv2.goodFeaturesToTrack(frames[a], 1500, 0.01, 7)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(frames[a], frames[b], pts, None, winSize=(25, 25), maxLevel=4)
    Hk, inliers = cv2.findHomography(pts[st[:, 0] == 1], nxt[st[:, 0] == 1], cv2.RANSAC, 1.5)
    return Hk, int(inliers.sum())


n = len(frames)
Hs = [None] * n
Hs[kref] = np.eye(3)
worst = 10 ** 9
for k in range(kref + 1, n):
    Hk, ni = pair(k - 1, k)
    Hs[k] = Hk @ Hs[k - 1]
    worst = min(worst, ni)
for k in range(kref - 1, -1, -1):
    Hk, ni = pair(k + 1, k)
    Hs[k] = Hk @ Hs[k + 1]
    worst = min(worst, ni)
S = np.diag([2.0, 2.0, 1.0])
json.dump({'H': [(S @ m @ np.linalg.inv(S)).tolist() for m in Hs], 'kref': kref}, open(out, 'w'))
print('frames', n, 'min RANSAC inliers', worst)   # hundreds or more = solid lock; low values mean drift
