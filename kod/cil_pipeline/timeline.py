"""Single source of truth for timing: voice edit, words, music edit, beats."""
import json, numpy as np

# --- voice: keep ranges on the cleaned take (source seconds); output starts at VOICE_T0
VOICE_T0 = 0.20
KEEPS = [(1.22, 4.10), (4.78, 9.10), (9.36, 10.60), (11.00, 13.40),
         (13.60, 15.86), (16.05, 17.20), (17.48, 19.42), (19.88, 22.76)]

def keep_table():
    tab, t = [], VOICE_T0
    for a, b in KEEPS:
        tab.append((a, b, t, t + (b - a)))
        t += b - a
    return tab

def src_to_T(s):
    for a, b, ta, tb in keep_table():
        if a - 1e-6 <= s <= b + 1e-6:
            return ta + (s - a)
    return None

def T_to_src(T):
    """Map output time to speaker-take time (for lip-synced picture)."""
    tab = keep_table()
    for a, b, ta, tb in tab:
        if ta - 1e-6 <= T < tb:
            return a + (T - ta)
    if T < tab[0][2]:
        return tab[0][0] - (tab[0][2] - T)
    a, b, ta, tb = tab[-1]
    return b + (T - tb)

_al = json.load(open('/tmp/edit/work/tx/v86_align.json'))
WORDS = [dict(w=x['w'], T=src_to_T(x['s']), Te=src_to_T(x['e'])) for x in _al]

# --- music: 'Take this Higher' (Mixkit 1012); beats from librosa
MUSIC = '/tmp/edit/music/cand/1012.mp3'
BEAT = 60.0 / 124.0                                   # kick-calibrated: exactly 124 BPM
TB = 29.0402 + BEAT * np.arange(-60, 150)             # constant grid anchored at the drop downbeat
I_DROP = 60
T_DROP1 = [w['T'] for w in WORDS if w['w'].startswith('grobeton')][0] - 0.005

def tb(i):
    return float(TB[i])

# music edit segments: (track_in, track_out, T_start)
T_JUMP = T_DROP1 + (tb(I_DROP + 24) - tb(I_DROP))       # 6 bars after drop 1
T_DROP2 = T_JUMP + (tb(I_DROP) - tb(I_DROP - 4))        # 1 bar pre-drop again
T_LOGO = T_DROP2 + (tb(I_DROP + 8) - tb(I_DROP))        # 2 bars of montage
I_END = I_DROP + 4 * 31                               # bar at 89.04: one bar, last hit, then the track's fade
M_OFF1 = tb(I_DROP) - T_DROP1
MUSIC_EDIT = [
    (M_OFF1, tb(I_DROP + 24), 0.0),                      # intro groove -> gap -> drop 1 -> 6 bars
    (tb(I_DROP - 4), tb(I_DROP + 8), T_JUMP),            # pre-drop bar -> drop 2 -> 2 bars
    (tb(I_END), tb(I_END) + 12.0, T_LOGO),               # final bars + natural fade-out
]
TOTAL = T_LOGO + 4.0

def beats_T():
    """Output-time beat grid derived through the music edit."""
    out = []
    for tin, tout, Ts in MUSIC_EDIT:
        for b in TB:
            if tin - 1e-3 <= b < tout - 1e-3:
                out.append(Ts + (b - tin))
    return np.array(sorted(out))

if __name__ == '__main__':
    for a, b, ta, tb_ in keep_table():
        print(f"keep src {a:6.2f}-{b:6.2f} -> T {ta:6.3f}-{tb_:6.3f}")
    for w in WORDS:
        print(f"{w['w']:22s} T={w['T']:7.3f} Te={w['Te']:7.3f}")
    print('T_DROP1', round(T_DROP1, 3), 'T_JUMP', round(T_JUMP, 3), 'T_DROP2', round(T_DROP2, 3), 'T_LOGO', round(T_LOGO, 3), 'TOTAL', round(TOTAL, 3))
    print('I_DROP', I_DROP, tb(I_DROP), 'I_END', I_END, tb(I_END))
    print('MUSIC_EDIT', [(round(a, 3), round(b, 3), round(c, 3)) for a, b, c in MUSIC_EDIT])
    bt = beats_T()
    print('beats', [round(x, 3) for x in bt if x < TOTAL])
