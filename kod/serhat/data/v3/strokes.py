import numpy as np, cv2, json
R = json.load(open('/tmp/jobs/serhat/v3/refs.json'))
def load(n):
    im = cv2.imread(f'/tmp/jobs/serhat/src2/isaret/{n}.png')[:, R[n]['x0']:]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return im, hsv[..., 0].astype(int) * 2, hsv[..., 1].astype(int), hsv[..., 2].astype(int)
def to_frame(n, pts):
    A = np.array(R[n]['A']); pts = np.asarray(pts, np.float64)
    return pts @ A[:, :2].T + A[:, 2]
def pix(mask, box=None):
    m = mask.copy()
    if box is not None:
        x0, y0, x1, y1 = box; k = np.zeros_like(m); k[y0:y1, x0:x1] = 1; m &= k.astype(bool)
    ys, xs = np.nonzero(m); return np.c_[xs, ys].astype(np.float64)
out = {}
# --- Adsiz: red outline + orange street line
im, h, s, v = load('Adsiz')
red = (((h < 12) | (h > 348)) & (s > 150) & (v > 150))
org = ((h > 18) & (h < 40) & (s > 170) & (v > 200))
P = pix(red, (30, 650, 400, 880))
cnts, _ = cv2.findContours((red[:880] * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(cnts, key=cv2.contourArea)
# inner boundary of the drawn loop = lot edge the customer traced; use the loop's centre line: erode-free approach -> polygon of the stroke's outer contour, shrunk by half the stroke width
mk = np.zeros(red.shape, np.uint8); cv2.drawContours(mk, [c], -1, 255, -1)
holes = cv2.findContours(((mk > 0) & ~red).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
inner = max(holes, key=cv2.contourArea)
out['park_poly'] = to_frame('Adsiz', cv2.approxPolyDP(inner, 3.0, True).reshape(-1, 2)).tolist()
O = pix(org, (220, 590, 500, 730))
mu = O.mean(0); u = np.linalg.svd(O - mu)[2][0]; tt = (O - mu) @ u
out['street'] = to_frame('Adsiz', [mu + u * tt.min(), mu + u * tt.max()]).tolist()
# --- Yol: dark red school outline + black X
im, h, s, v = load('Yol')
dred = (((h < 12) | (h > 340)) & (s > 150) & (v > 80) & (v < 230))
cnts, _ = cv2.findContours((dred[:880] * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = [c for c in cnts if cv2.boundingRect(c)[1] > 430 and cv2.boundingRect(c)[0] < 320]
c = max(cnts, key=cv2.contourArea)
mk = np.zeros(dred.shape, np.uint8); cv2.drawContours(mk, [c], -1, 255, -1)
holes = cv2.findContours(((mk > 0) & ~dred).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
inner = max(holes, key=cv2.contourArea)
out['school_poly'] = to_frame('Yol', cv2.approxPolyDP(inner, 3.0, True).reshape(-1, 2)).tolist()
blk = (v < 40)
X = pix(blk, (240, 515, 370, 590))
# split into two strokes by angle clustering with Hough on the stroke mask
m8 = np.zeros(blk.shape, np.uint8); m8[515:590, 240:370] = blk[515:590, 240:370]
lines = cv2.HoughLinesP(m8 * 255, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=6)
segs = [l for l in np.asarray(lines).reshape(-1, 4)]
def ang(sg): return np.degrees(np.arctan2(sg[3] - sg[1], sg[2] - sg[0])) % 180
g1 = [sg for sg in segs if ang(sg) < 30 or ang(sg) > 150]; g2 = [sg for sg in segs if 30 <= ang(sg) <= 150]
def span(group):
    pts = np.array([[sg[0], sg[1]] for sg in group] + [[sg[2], sg[3]] for sg in group], float)
    mu = pts.mean(0); u = np.linalg.svd(pts - mu)[2][0]; tt = (pts - mu) @ u
    return [mu + u * tt.min(), mu + u * tt.max()]
out['x_a'] = to_frame('Yol', span(g1)).tolist(); out['x_b'] = to_frame('Yol', span(g2)).tolist()
# --- isaret2: green pointer line + yellow underline
im, h, s, v = load('isaret2')
grn = ((h > 100) & (h < 160) & (s > 120) & (v > 90))
ylw = ((h > 40) & (h < 60) & (s > 150) & (v > 180))
G = pix(grn, (330, 360, 430, 425))
mu = G.mean(0); u = np.linalg.svd(G - mu)[2][0]; tt = (G - mu) @ u
ends = [mu + u * tt.min(), mu + u * tt.max()]
ends.sort(key=lambda p: p[1])                                     # lower end = the square
out['meydan_line'] = to_frame('isaret2', ends).tolist()
Y = pix(ylw, (360, 425, 510, 465))
mu = Y.mean(0); u = np.linalg.svd(Y - mu)[2][0]; tt = (Y - mu) @ u
out['belediye_line'] = to_frame('isaret2', [mu + u * tt.min(), mu + u * tt.max()]).tolist()
json.dump(out, open('/tmp/jobs/serhat/v3/strokes.json', 'w'), indent=1)
for k, vv in out.items(): print(k, np.round(np.array(vv), 1).tolist()[:8], '...' if len(vv) > 8 else '')
# --- overlays on the reference frames (intermediate coords) for checking
vis = []
for n, keys in (('Yol', ['school_poly', 'x_a', 'x_b']), ('Adsiz', ['park_poly', 'street']), ('isaret2', ['meydan_line', 'belediye_line'])):
    f = cv2.imread(f'/tmp/jobs/serhat/v3/ref_{n}.png')
    for k in keys:
        P = np.round(np.array(out[k])).astype(np.int32)
        cv2.polylines(f, [P], k.endswith('poly'), (0, 255, 255), 5, cv2.LINE_AA)
        for p in P: cv2.circle(f, tuple(p), 7, (255, 0, 255), -1)
    vis.append(cv2.resize(f, (540, 960), interpolation=cv2.INTER_AREA))
cv2.imwrite('/tmp/hoplite/workspace/.scratch/v3_strokes.jpg', np.hstack(vis), [cv2.IMWRITE_JPEG_QUALITY, 90])
