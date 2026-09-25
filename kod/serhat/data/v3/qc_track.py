import json, numpy as np, cv2, subprocess
S = json.load(open('/tmp/jobs/serhat/v3/strokes.json'))
def proj(Hm, P):
    P = np.atleast_2d(np.asarray(P, float)); q = np.c_[P, np.ones(len(P))] @ np.asarray(Hm).T; return q[:, :2] / q[:, 2:3]
tiles = []
for trk, keys, times in (('trk_yol', ['school_poly', 'x_a', 'x_b'], [82.3, 83.2, 84.17, 85.1, 86.1]),
                         ('trk_park', ['park_poly', 'street'], [57.6, 58.3, 58.93, 59.6, 60.3]),
                         ('trk_mey', ['meydan_line', 'belediye_line'], [135.4, 136.2, 136.93, 137.8, 138.6])):
    d = json.load(open(f'/tmp/jobs/serhat/v3/{trk}.json')); FPS = d['fps']; off = d['off']
    Hs = {int(k): v for k, v in d['H'].items()}
    row = []
    for t in times:
        k = int(round((t - off) * FPS)); k = min(max(k, min(Hs)), max(Hs))
        b = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{k / FPS:.5f}', '-i', f"/tmp/jobs/serhat/inter/{d['name']}.mp4", '-frames:v', '1',
                            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
        f = np.frombuffer(b, np.uint8).reshape(2304, 1296, 3).copy()
        for kk in keys:
            Q = np.round(proj(Hs[k], S[kk])).astype(np.int32)
            cv2.polylines(f, [Q], kk.endswith('poly'), (0, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(f, f'{t:.2f}', (30, 120), 0, 3.2, (0, 0, 255), 8)
        row.append(cv2.resize(f, (259, 461), interpolation=cv2.INTER_AREA))
    tiles.append(np.hstack(row))
    # motion speed of a target point (px/s in intermediate coords)
    c = np.mean(np.array(S[keys[0]]), 0)
    ks = sorted(Hs); pts = np.array([proj(Hs[k], c)[0] for k in ks])
    v = np.linalg.norm(np.diff(pts, axis=0), axis=1) * FPS
    print(trk, 'target speed px/s median', round(float(np.median(v)), 1), 'max', round(float(v.max()), 1), 'range x', pts[:, 0].min().round(), pts[:, 0].max().round(), 'y', pts[:, 1].min().round(), pts[:, 1].max().round())
cv2.imwrite('/tmp/hoplite/workspace/.scratch/v3_trackqc.jpg', np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
