import subprocess, numpy as np, cv2, json
SW, SH = 90, 160
def prep(g):
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.sqrt(gx ** 2 + gy ** 2)
    return m
shots = {}
for n, x0 in (('Adsiz', 0), ('isaret2', 16), ('Yol', 0)):
    im = cv2.imread(f'/tmp/jobs/serhat/src2/isaret/{n}.png', cv2.IMREAD_GRAYSCALE)[:, x0:]
    h, w = im.shape
    full_h = int(round(w * 16 / 9))                      # video frame height at this width
    fr = np.zeros((full_h, w), np.uint8); fr[:min(h, full_h)] = im[:min(h, full_h)]
    g = cv2.resize(fr, (SW, SH), interpolation=cv2.INTER_AREA)
    shots[n] = (prep(g), int(SH * 850 / full_h))         # use rows above the player UI only
res = {n: [] for n in shots}
for name, off, dur in (('D_A', 0.0, 38.2), ('D_B', 38.0, 78.2), ('D_C', 116.0, 80.0)):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', f'/tmp/jobs/serhat/inter/{name}.mp4', '-vf', f'fps=6,scale={SW}:{SH}:flags=area',
                          '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], stdout=subprocess.PIPE)
    k = 0
    while True:
        b = p.stdout.read(SW * SH)
        if len(b) < SW * SH: break
        f = prep(np.frombuffer(b, np.uint8).reshape(SH, SW))
        t = off + k / 6.0
        for n, (s, rows) in shots.items():
            a = s[:rows].ravel(); c = f[:rows].ravel()
            a = a - a.mean(); c = c - c.mean()
            res[n].append((float((a * c).sum() / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9)), t))
        k += 1
for n in res:
    r = sorted(res[n], reverse=True)
    best = []
    for sc, t in r:
        if all(abs(t - b) > 3 for _, b in best): best.append((sc, t))
        if len(best) == 4: break
    print(n, [(round(t, 2), round(sc, 3)) for sc, t in best])
json.dump({n: sorted(res[n], reverse=True)[:40] for n in res}, open('/tmp/jobs/serhat/v3/match_coarse.json', 'w'))
