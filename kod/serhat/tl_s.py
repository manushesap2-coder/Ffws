"""Serhat timeline: speech words, VO words, beat grid, shots and source-time mapping."""
import json, numpy as np
FPS = 30
W, H = 1080, 1920
FPS_SRC = 60000 / 1001
BAR = 4 * 0.48497; BEAT = 0.48497
T_DROP = 11 * BAR                          # music drop (track 15.595) lands here
T_VO = T_DROP + 0.30                       # voice-over start
def B(k): return T_DROP + k * BEAT
T_END = T_VO + 34.35 + 2.45

# ---- girl speech (MMS alignment on the cleaned Pocket take; output T = source t)
_al = json.load(open('/tmp/jobs/serhat/work/p77_align.json'))
def _w(i): return _al[i]
SPEECH = [dict(w=a['w'], T=a['s'], Te=a['e']) for a in _al]
def merge(i, j, text):
    return dict(w=text, T=SPEECH[i]['T'], Te=SPEECH[j]['Te'])
def word(i, text=None):
    d = dict(SPEECH[i]); d['w'] = text or d['w']; return d
# index map: 0 Burada 1 ne 2 iş 3 yapılır 4 diye 5 düşünüyorsanız 6 yanlış 7 soruyu 8 soruyorsunuz. 9 Asıl 10 soru 11 burada
# 12 ne 13 iş 14 yapılmaz. 15 Yaklaşık 16 üç 17 yüz 18 metre 19 genişliği, 20 otuz 21 metre 22 uzunluğu 23 ve 24 dört 25 buçuk
# 26 metre 27 tavan 28 yüksekliğiyle 29 işletmenize 30 gerçekten 31 geniş 32 bir 33 kullanım 34 alanı 35 ve 36 asma 37 kat 38 imkanı 39 ile 40 ekstra 41 alan 42 sunuyor.
CHUNKS = [
    [word(0), word(1), word(2), word(3)],
    [word(4), word(5)],
    [word(6), word(7), word(8)],
    [word(9), word(10)],
    [word(11), word(12), word(13), word(14)],
    [word(15), merge(16, 17, '300'), word(18, 'm²'), word(19)],
    [word(20, '30'), word(21), word(22)],
    [word(23), merge(24, 25, '4,5'), word(26)],
    [word(27), word(28)],
    [word(29), word(30)],
    [word(31), word(32), word(33), word(34)],
    [word(35), word(36), word(37), word(38, 'imkânı'), word(39)],
    [word(40), word(41), word(42)],
]

# ---- voice-over words (output T = T_VO + v)
_vo = json.load(open('/tmp/jobs/serhat/work/vo_align.json'))
VO = [dict(w=a['w'], T=T_VO + a['s'], Te=T_VO + a['e']) for a in _vo]
def vo(word_prefix, nth=0):
    hits = [w for w in VO if w['w'].lower().startswith(word_prefix.lower())]
    return hits[nth]

# ---- one-take drone speed curve (0 .. B(25)); tuned so the building exit (src 29.35) lands on the drop
T_TAKE_END = B(25)
SRC_EXIT = 29.35
def _speed_table(vmax):
    keys = [(0.0, 1.0), (17.75, 1.0), (T_DROP, vmax), (T_DROP + 1.2, 1.5), (29.6, 1.5), (30.2, 2.4), (T_TAKE_END + 0.5, 2.4)]
    ts = np.arange(0, T_TAKE_END + 0.5, 1 / 600)
    v = np.empty_like(ts)
    for i in range(len(keys) - 1):
        (a, va), (b, vb) = keys[i], keys[i + 1]
        m = (ts >= a) & (ts <= b)
        u = (ts[m] - a) / (b - a)
        if i == 1:   s = u ** 2                      # ease-in acceleration into the exit
        elif i == 2: s = 1 - (1 - u) ** 2            # ease-out after the exit
        else:        s = u * u * (3 - 2 * u)
        v[m] = va + (vb - va) * s
    src = 1.55 + np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(ts))])
    return ts, v, src
lo, hi = 2.0, 12.0
for _ in range(50):
    mid = (lo + hi) / 2
    ts_, v_, src_ = _speed_table(mid)
    if np.interp(T_DROP, ts_, src_) < SRC_EXIT: lo = mid
    else: hi = mid
TAKE_T, TAKE_V, TAKE_SRC = _speed_table((lo + hi) / 2)
def take_src(T):
    return float(np.interp(T, TAKE_T, TAKE_SRC)), float(np.interp(T, TAKE_T, TAKE_V))

# ---- shots: (name, t0, t1, kind, params)
SHOTS = [
    dict(n='TAKE', t0=0.0, t1=B(25), kind='take'),
    dict(n='INT1', t0=B(25), t1=B(32), kind='lin', src='P77', s0=0.6, v=1.25, grade='int', tin=('whip', 5, (-1, 0))),
    dict(n='INT2', t0=B(32), t1=B(37), kind='lin', src='D', s0=20.0, v=2.3, grade='int', tin=('whip', 4, (1, 0))),
    dict(n='INT3', t0=B(37), t1=B(42), kind='lin', src='P77', s0=11.5, v=1.8, grade='int', tin=('zoom', 4)),
    dict(n='PARK', t0=B(42), t1=B(45), kind='lin', src='D', s0=50.5, v=1.5, grade='ext', zoom=(1.18, 1.26), center=(700, 1500), tin=('whip', 5, (0, 1))),
    dict(n='CITY', t0=B(45), t1=B(51), kind='lin', src='D', s0=60.0, v=5.0, grade='ext', tin=('zoom', 4)),
    dict(n='WIDE', t0=B(51), t1=B(60), kind='lin', src='D', s0=95.0, v=6.6, grade='ext', tin=('whip', 4, (-1, 0))),
    dict(n='FINAL', t0=B(60), t1=T_END, kind='lin', src='D', s0=150.0, v=4.0, grade='ext', tin=('zoomout', 6)),
]

if __name__ == '__main__':
    print('T_DROP', round(T_DROP, 3), 'T_VO', round(T_VO, 3), 'T_END', round(T_END, 3), 'vmax', round((lo + hi) / 2, 3))
    for T in (0, 6.85, 9.9, 12.5, 17.75, 19.5, T_DROP, T_DROP + 1.2, 24.64, 26.74, 29.94, B(25)):
        print(f'T {T:6.2f} src {take_src(T)[0]:6.2f} v {take_src(T)[1]:4.2f}')
    for s in SHOTS: print(s['n'], round(s['t0'], 3), round(s['t1'], 3))
    for c in CHUNKS: print(' '.join(f"{w['w']}@{w['T']:.2f}" for w in c))
