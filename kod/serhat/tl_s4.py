"""Serhat timeline: speech words, VO words, beat grid, shots and source-time mapping."""
import json, numpy as np
FPS = 30
W, H = 1080, 1920
FPS_SRC = 60000 / 1001
BAR = 4 * 0.48497; BEAT = 0.48497
T_DROP = 11 * BAR                          # music drop (track 15.595) lands here
T_VO = T_DROP + 0.30                       # voice-over start
def B(k): return T_DROP + k * BEAT
VO_SPLIT = 21.917
EXT = 3 * BEAT                               # v4: parking shot held 3 beats longer (customer: more time on the parking area)
VO_PAUSE = 0.40 + EXT                        # breath after 'Otopark avantajı,' grows with it
T_FIN = T_VO + 34.35 + VO_PAUSE + 2.45       # end of the drone/logo finale; the Şule clip is appended after it (sule_s4.py)
T_END = T_FIN
# ---- v4 ending: Şule's CTA (Pocket 0083) cross-dissolved in after the finale
XF = 0.40                                    # xfade length main -> Şule
SULE_S0, SULE_S1, SULE_HOLD = 0.45, 9.62, 1.0  # clip in/out (speech 0.84-8.63) + freeze-frame hold for the end card
D_SULE = SULE_S1 - SULE_S0 + SULE_HOLD
L_MAIN = round(T_FIN * FPS) / FPS
T_SULE = L_MAIN - XF                         # Şule segment start on the final timeline
T_TOT = T_SULE + D_SULE

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
def vo_T(v): return T_VO + v + (VO_PAUSE if v > VO_SPLIT else 0.0)
VO = [dict(w=a['w'], T=vo_T(a['s']), Te=vo_T(a['e'])) for a in _vo]
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
# v4: + parking held longer (EXT); v3: customer markings (Yol / Adsız / işaret2) at the matching VO lines; no Pocket (P77) footage
REF = dict(yol=84.171, park=58.934, mey=136.931)
SHOTS = [
    dict(n='TAKE', t0=0.0, t1=B(18), kind='take'),
    dict(n='YOL', t0=B(18), t1=B(25), kind='lin', src='D', s0=REF['yol'] - (31.75 - B(18)), v=1.0, grade='ext',
         zoom=(1.30, 1.42), center=(580.0, 1130.0), tin=('zoom', 5), trk='trk_yol'),
    dict(n='REVIN', t0=B(25), t1=B(42), kind='rev', grade='int', zoom=(1.0, 1.12), center=(648.0, 1152.0), tin=('whip', 5, (-1, 0))),
    dict(n='PARK', t0=B(42), t1=B(47) + EXT, kind='lin', src='D', s0=REF['park'] - (42.95 - B(42)), v=1.0, grade='ext',
         zoom=(1.08, 1.16), center=(620.0, 1235.0), tin=('zoom', 5), trk='trk_park'),
    dict(n='MEY', t0=B(47) + EXT, t1=B(53) + EXT, kind='lin', src='D', s0=REF['mey'] - (45.45 - B(47)), v=1.0, grade='ext',
         zoom=(1.30, 1.42), center=(778.0, 1000.0), tin=('whip', 5, (1, 0)), trk='trk_mey'),
    dict(n='WIDE', t0=B(53) + EXT, t1=B(61) + EXT, kind='lin', src='D', s0=95.0, v=6.4, grade='ext', tin=('zoom', 5)),
    dict(n='FINAL', t0=B(61) + EXT, t1=T_END, kind='lin', src='D', s0=150.0, v=4.0, grade='ext', tin=('zoomout', 6)),
]
# reversed fly-in (drone re-enters the unit): speed 1.7x -> 1.0x over the first 2.4 s, source runs backwards from REV_S0
REV_S0 = 30.0
def _rev_speed(u):
    k = float(np.clip(u / 2.4, 0, 1)); e = k * k * (3 - 2 * k)
    return 1.7 + (1.0 - 1.7) * e
_RU = np.arange(0, 12.0, 1 / 600.0)
_RV = np.array([_rev_speed(u) for u in _RU])
_RS = np.concatenate([[0], np.cumsum((_RV[1:] + _RV[:-1]) / 2 * np.diff(_RU))])
def rev_src(T):
    """-> (source second, speed) for the reversed fly-in; source decreases with T."""
    u = max(0.0, T - B(25))
    return REV_S0 - float(np.interp(u, _RU, _RS)), float(np.interp(u, _RU, _RV))

if __name__ == '__main__':
    print('T_DROP', round(T_DROP, 3), 'T_VO', round(T_VO, 3), 'T_END', round(T_END, 3), 'vmax', round((lo + hi) / 2, 3))
    for T in (0, 6.85, 9.9, 12.5, 17.75, 19.5, T_DROP, T_DROP + 1.2, 24.64, 26.74, 29.94, B(25)):
        print(f'T {T:6.2f} src {take_src(T)[0]:6.2f} v {take_src(T)[1]:4.2f}')
    for s in SHOTS: print(s['n'], round(s['t0'], 3), round(s['t1'], 3), 's0', round(s.get('s0', 0), 3))
    print('REVIN src', round(rev_src(B(25))[0], 2), '->', round(rev_src(B(42))[0], 2))
    for w in ('Otopark', 'avantajı', 'sosyal', 'resmi', 'işletmeniz', 'güçlü', 'değerli', 'fırsat', 'kalbinde'):
        x = vo(w) if w != 'Serhat' else vo(w, 1); print(' vo', w, round(x['T'], 3), round(x['Te'], 3))
    for c in CHUNKS: print(' '.join(f"{w['w']}@{w['T']:.2f}" for w in c))
