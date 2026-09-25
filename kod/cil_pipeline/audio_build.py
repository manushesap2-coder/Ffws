"""Voice edit + processing, beat-aligned music edit with ducking, SFX placement, master to -14 LUFS."""
import sys, subprocess, json, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/edit/pipeline')
import timeline as TL

SR = 48000
OUT = '/tmp/edit/out/'
TOTAL = TL.TOTAL
N = int(TOTAL * SR) + SR

def ff_decode(path, ch=2):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', str(ch), '-ar', str(SR), '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).copy()

def ff_filter(x, af, ch):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ac', str(ch), '-ar', str(SR), '-i', '-', '-af', af,
                        '-f', 'f32le', '-ac', str(ch), '-ar', str(SR), '-'], input=x.astype(np.float32).tobytes(), capture_output=True)
    y = np.frombuffer(p.stdout, np.float32).reshape(-1, ch).copy()
    if len(y) < len(x): y = np.vstack([y, np.zeros((len(x) - len(y), ch), np.float32)])
    return y[:len(x)]

def fade(n, kind='in'):
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.sin(t) ** 2 if kind == 'in' else np.cos(t) ** 2

def db(g): return 10 ** (g / 20)

# ---------------------------------------------------------------- voice
v_src, sr = sf.read('/tmp/edit/work/audio/df/v86_pre.wav', dtype='float32')
assert sr == SR
voice = np.zeros(N, np.float32)
F = int(0.010 * SR)
for a, b, ta, tb in TL.keep_table():
    seg = v_src[int(a * SR):int(b * SR)].copy()
    seg[:F] *= fade(F, 'in'); seg[-F:] *= fade(F, 'out')
    i = int(round(ta * SR)); voice[i:i + len(seg)] += seg
vchain = ('highpass=f=85:poles=2,highpass=f=85:poles=2,'
          'equalizer=f=240:t=q:w=1.1:g=-2.5,equalizer=f=3300:t=q:w=1.0:g=2.8,'
          'highshelf=f=9500:g=2.0,deesser=i=0.35:m=0.5:f=0.5,'
          'acompressor=threshold=-24dB:ratio=3:attack=6:release=110:makeup=5:knee=4,'
          'alimiter=limit=0.7:attack=3:release=60:level=disabled')
voice = ff_filter(voice[:, None], vchain, 1)[:, 0]

# voice activity envelope for ducking (attack 25 ms, release 280 ms)
hop = int(0.005 * SR)
fr = np.abs(voice[:len(voice) // hop * hop]).reshape(-1, hop).max(1)
act = np.clip((20 * np.log10(fr + 1e-6) + 42) / 14, 0, 1)
env = np.zeros_like(act)
a_up, a_dn = 1 - np.exp(-5 / 25), 1 - np.exp(-5 / 280)
for i in range(1, len(act)):
    c = a_up if act[i] > env[i - 1] else a_dn
    env[i] = env[i - 1] + c * (act[i] - env[i - 1])
duck = np.repeat(env, hop)
duck = np.concatenate([duck, np.zeros(N - len(duck), np.float32)])[:N]

# ---------------------------------------------------------------- music
mus = ff_decode(TL.MUSIC, 2)
music = np.zeros((N, 2), np.float32)
X = int(0.018 * SR)                          # crossfade ending exactly on the downbeat
for j, (tin, tout, Ts) in enumerate(TL.MUSIC_EDIT):
    i0 = int(round(Ts * SR)); s0 = int(round(tin * SR))
    pre = X if j > 0 else 0
    L = int(round((tout - tin) * SR)) + (X if j < len(TL.MUSIC_EDIT) - 1 else 0)
    seg = mus[s0 - pre:s0 + L].copy()
    if len(seg) == 0: continue
    if pre: seg[:pre] *= fade(pre, 'in')[:, None]
    if j < len(TL.MUSIC_EDIT) - 1: seg[-X:] *= fade(X, 'out')[:, None]
    k0 = i0 - pre; k1 = min(N, k0 + len(seg))
    music[k0:k1] += seg[:k1 - k0]
t = np.arange(N) / SR
g_db = -11.0 * duck                          # sidechain-style duck under speech
g_db += np.where(t < TL.T_DROP1, -1.0, 0.0)  # slightly softer intro groove
music *= db(g_db)[:, None]
fo = (t > TOTAL - 1.2)
music[fo] *= np.clip((TOTAL - t[fo]) / 1.2, 0, 1)[:, None] ** 2

# ---------------------------------------------------------------- sfx
SFX_DIR = '/tmp/edit/sfx/'
PEAK = {'1491': 0.29, '1143': 0.56, '3005': 0.03, '174': 0.49, '3115': 0.11, '790': 2.54, '788': 2.18, '1492': 1.08,
        '1471': 0.39, '1490': 0.73, '2356': 0.05, '1489': 0.72, '1493': 1.97, '1468': 0.31, '2901': 1.42, '2350': 0.45, '2353': 1.02}
B = lambda k: TL.T_DROP1 + k * TL.BEAT
words = {w['w']: w['T'] for w in TL.WORDS}
EV = [  # (sfx id, time of its peak, gain dB)
    ('1491', 0.12, -9), ('1143', 0.40, -7), ('174', 1.38, -15), ('3005', 1.42, -11), ('3115', 2.96, -17),
    ('790', B(0), -13), ('788', B(0), -5), ('1492', B(0) - 0.03, -11), ('1471', B(4) - 0.18, -17),
    ('1490', B(2), -13), ('3115', B(4), -13), ('1491', B(5), -13), ('174', B(5) + 0.02, -16),
    ('2356', words['sağlamlık,'] - 0.02, -9), ('2356', words['kalite'] - 0.02, -9), ('2356', words['doğru'] - 0.02, -9),
    ('3115', B(7), -19), ('1490', B(9), -14), ('3115', B(11), -14), ('1489', B(14), -11), ('1493', B(17), -10),
    ('1471', B(21), -11), ('1468', B(23), -11), ('790', B(28), -8), ('2901', B(28), -5),
    ('1491', B(29), -12), ('3115', B(30), -12), ('1490', B(31), -12), ('1489', B(32), -11), ('1493', B(34), -10),
    ('788', B(36), -3), ('2350', B(36) + 0.75, -13), ('2353', B(40), -15),
]
sfx = np.zeros((N, 2), np.float32)
cache = {}
for sid, tp, gdb in EV:
    if sid not in cache: cache[sid] = ff_decode(SFX_DIR + f's_{sid}.mp3', 2)
    x = cache[sid]
    start = int(round((tp - PEAK[sid]) * SR))
    a0 = max(0, -start); k0 = max(0, start)
    seg = x[a0:]
    if a0 > 0:
        r = min(len(seg), int(0.004 * SR)); seg = seg.copy(); seg[:r] *= fade(r, 'in')[:, None]
    k1 = min(N, k0 + len(seg))
    sfx[k0:k1] += seg[:k1 - k0] * db(gdb)
sfx = ff_filter(sfx, 'alimiter=limit=0.8:attack=2:release=50:level=disabled', 2)

# ---------------------------------------------------------------- mix & master
v2 = np.stack([voice, voice], 1) * db(-1.0)
mix = v2 + music * db(-3.0) + sfx * db(-1.0)
mix = mix[:int(TOTAL * SR)]
sf.write(OUT + 'mix_pre.wav', mix, SR, subtype='FLOAT')
for name, sig in [('voice', v2), ('music', music), ('sfx', sfx)]:
    sf.write(OUT + f'stem_{name}.wav', sig[:int(TOTAL * SR)], SR, subtype='FLOAT')
# two-pass loudnorm to -14 LUFS / -1.0 dBTP (linear)
m = subprocess.run(['ffmpeg', '-v', 'info', '-i', OUT + 'mix_pre.wav', '-af', 'alimiter=limit=0.89:attack=1:release=40:level=disabled,loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
js = json.loads(m[m.rindex('{'):m.rindex('}') + 1])
af = ('alimiter=limit=0.89:attack=1:release=40:level=disabled,'
      f"loudnorm=I=-14:TP=-1.0:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', OUT + 'mix_pre.wav', '-af', af, '-ar', str(SR), '-c:a', 'pcm_s24le', OUT + 'mix_final.wav'], check=True)
chk = subprocess.run(['ffmpeg', '-v', 'info', '-i', OUT + 'mix_final.wav', '-af', 'ebur128=peak=true', '-f', 'null', '-'], capture_output=True, text=True).stderr
print('measured pre:', js['input_i'], js['input_tp'])
print('\n'.join(l for l in chk.splitlines()[-12:] if l.strip()))
