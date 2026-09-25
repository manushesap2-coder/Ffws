import subprocess, numpy as np, cv2, json
FPS = 60000 / 1001
def seg_frames(src_t0, n, w=1296, h=2304):
    name, off = ('D_A', 0.0) if src_t0 < 38.1 else (('D_B', 38.0) if src_t0 < 116.1 else ('D_C', 116.0))
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{src_t0 - off:.4f}', '-i', f'/tmp/jobs/serhat/inter/{name}.mp4', '-frames:v', str(n),
                        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
    k = len(p) // (w * h * 3)
    return np.frombuffer(p[:k * w * h * 3], np.uint8).reshape(k, h, w, 3)
sift = cv2.SIFT_create(4000)
out = {}
for n, x0, tc in (('Adsiz', 0, 59.0), ('isaret2', 16, 136.83), ('Yol', 0, 84.17)):
    shot = cv2.imread(f'/tmp/jobs/serhat/src2/isaret/{n}.png')[:, x0:]
    sg = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(sg); mask[:870] = 255                     # exclude player UI
    ks, ds = sift.detectAndCompute(sg, mask)
    fr = seg_frames(tc - 0.6, int(1.2 * FPS))
    best = None
    for i in range(0, len(fr), 2):
        g = cv2.cvtColor(cv2.resize(fr[i], (648, 1152), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        kf, dfe = sift.detectAndCompute(g, None)
        m = cv2.BFMatcher().knnMatch(ds, dfe, k=2)
        good = [a for a, b in m if a.distance < 0.72 * b.distance]
        if len(good) < 12: continue
        P = np.float32([ks[a.queryIdx].pt for a in good]); Q = np.float32([kf[a.trainIdx].pt for a in good]) * 2.0
        A, inl = cv2.estimateAffinePartial2D(P, Q, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        ni = int(inl.sum()); err = float(np.median(np.linalg.norm((np.c_[P, np.ones(len(P))] @ A.T - Q)[inl.ravel() == 1], axis=1)))
        if best is None or (ni, -err) > (best[0], -best[1]): best = (ni, err, i, A)
    ni, err, i, A = best
    t = tc - 0.6 + i / FPS
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    print(f'{n}: src t {t:.3f}  inliers {ni}  median err {err:.2f}px  scale {s:.4f}  A={np.round(A, 3).tolist()}')
    cv2.imwrite(f'/tmp/jobs/serhat/v3/ref_{n}.png', fr[i])
    out[n] = dict(t=t, x0=x0, A=A.tolist())
json.dump(out, open('/tmp/jobs/serhat/v3/refs.json', 'w'), indent=1)
