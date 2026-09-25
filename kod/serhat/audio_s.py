"""Serhat audio: girl voice (denoised) + VO + edited music with ducking + SFX -> loudnorm -14 LUFS / -1 dBTP."""
import sys, subprocess, io, json, numpy as np, soundfile as sf
sys.path.insert(0, '/tmp/jobs/serhat/lib')
import tl_s as TL
SR = 48000
W = '/tmp/jobs/serhat/work/'; SFX = '/tmp/edit/sfx/'
def load(path, af=None):
    cmd = ['ffmpeg', '-v', 'error', '-i', path] + (['-af', af] if af else []) + ['-ac', '1', '-ar', str(SR), '-f', 'wav', '-acodec', 'pcm_f32le', '-']
    x, sr = sf.read(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout), dtype='float32')
    return x
def db(x): return 10 ** (x / 20)
def rms_active(x, thr=-45):
    win = 2400; n = len(x) // win; r = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(1) + 1e-12)
    a = r[20 * np.log10(r) > thr]; return 20 * np.log10(np.sqrt((a ** 2).mean())) if len(a) else -99
N = int(TL.T_END * SR) + SR
mix_v = np.zeros(N, np.float32); mix_m = np.zeros(N, np.float32); mix_s = np.zeros(N, np.float32)
def put(buf, x, t, gain=1.0):
    i = int(round(t * SR)); j = min(N, i + len(x))
    if i < 0: x = x[-i:]; i = 0; j = min(N, len(x))
    buf[i:j] += x[:j - i] * gain

# ---- girl voice
g = load(W + 'df/p77_hp.wav', 'highpass=f=90,equalizer=f=250:t=q:w=1.0:g=-2.5,equalizer=f=3500:t=q:w=0.9:g=2.5,equalizer=f=9000:t=q:w=1.0:g=1.0,'
         'deesser=i=0.35,acompressor=threshold=-24dB:ratio=3:attack=5:release=90:makeup=2')
end = TL.SPEECH[-1]['Te'] + 0.30
g = g[:int(end * SR)].copy(); fl = int(0.25 * SR); g[-fl:] *= np.linspace(1, 0, fl)
g *= db(-20 - rms_active(g))
put(mix_v, g, 0.0)
# ---- voice-over
v = load(W + 'vo_48k.wav', 'highpass=f=70,equalizer=f=180:t=q:w=1.0:g=-1.0,equalizer=f=4200:t=q:w=1.0:g=1.5,acompressor=threshold=-22dB:ratio=2:attack=8:release=120:makeup=1.5')
v *= db(-19.5 - rms_active(v))
put(mix_v, v, TL.T_VO)

# ---- music: Mixkit 623 "Deep Urban", bar-edited so its drop (15.595 s) lands on T_DROP
m = load('/tmp/jobs/music/623.mp3')
BAR = TL.BAR; t0 = 15.595 - 8 * BAR
def seg(a, b): return m[int(round(a * SR)):int(round(b * SR))].copy()
parts = [seg(t0 + BAR, t0 + 4 * BAR), seg(t0, 15.595), seg(15.595, 15.595 + TL.T_END - TL.T_DROP + 1.0)]
xf = int(0.03 * SR); out = parts[0]
for p in parts[1:]:
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    out[-xf:] = out[-xf:] * (1 - ramp) + p[:xf] * ramp
    out = np.concatenate([out, p[xf:]])
music = out[:N] if len(out) >= N else np.pad(out, (0, N - len(out)))
# ducking from voice envelope
env = np.sqrt(np.convolve(mix_v ** 2, np.ones(2400) / 2400, 'same'))
act = (20 * np.log10(env + 1e-9) > -42).astype(np.float32)
k = int(0.35 * SR); act = np.convolve(act, np.ones(k) / k, 'same')      # smooth attack/release
act = np.clip(act * 1.6, 0, 1)
t = np.arange(N) / SR
base = np.full(N, -21.0, np.float32)                     # music level without voice
base[t < 17.7] = -26.0                                   # softer intro under the girl
ramp = (t >= 17.7) & (t < TL.T_DROP); base[ramp] = -26 + 6 * ((t[ramp] - 17.7) / (TL.T_DROP - 17.7))
duck = -12.5 * act
gain_db = base + duck
fo = (t >= TL.T_END - 2.4); gain_db[fo] -= 40 * ((t[fo] - (TL.T_END - 2.4)) / 2.4) ** 2
music_level = rms_active(music[int(22 * SR):int(50 * SR)])
music = music * db(gain_db - music_level)
mix_m += music.astype(np.float32)

# ---- SFX
def sfx(i, t_peak, peak_at, gain_db, rev=False, cut=None):
    x = load(SFX + f's_{i}.mp3')
    if rev:
        x = x[::-1].copy(); peak_at = len(x) / SR - peak_at
    if cut: x = x[:int(cut * SR)]
    put(mix_s, x, t_peak - peak_at, db(gain_db))
sfx(788, TL.T_DROP - 0.02, 2.17, -9, rev=True)            # reversed impact = riser into the exit
sfx(1143, TL.T_DROP + 0.02, 0.63, -6)                     # impact on the drop
for t_line in (6.85, 8.30, 9.88): sfx(1491, t_line + 0.22, 0.30, -20)
for t_tag in (7.38, 8.88, 10.40): sfx(3005, t_tag + 0.03, 0.03, -17)
cuts = [(TL.B(25), 1490), (TL.B(32), 3115), (TL.B(37), 1492), (TL.B(42), 1490), (TL.B(45), 1492), (TL.B(51), 1490), (TL.B(60), 1492)]
peaks = {1490: 0.73, 1492: 1.08, 3115: 0.11}
for tc, i in cuts: sfx(i, tc, peaks[i], -11)
pops = [TL.vo('çarşının')['T'] + 0.2, TL.vo('yoğun')['T'] - 0.05, TL.vo('Geniş')['T'], TL.vo('yüksek')['T'], TL.vo('asma')['T'],
        TL.vo('farklı')['T'], TL.vo('ferah')['T'], TL.vo('Otopark')['T'], TL.vo('sosyal')['T'], TL.vo('resmi')['T'],
        TL.vo('güçlü')['T'], TL.vo('değerli')['T'], TL.vo('Serhat')['T'], TL.vo('İnci')['T']]
for tp in pops: sfx(3005, tp, 0.03, -19)
sfx(2350, TL.vo('Serhat', 1)['T'] + 0.25, 0.45, -12)      # logo wipe sparkle
sfx(1143, TL.vo('Serhat', 1)['T'] + 0.9, 0.63, -16)       # soft low hit when the skyline completes

mix = mix_v + mix_m + mix_s
mix = mix[:int(TL.T_END * SR)]
sf.write('/tmp/jobs/serhat/out/mix_raw.wav', mix, SR, subtype='FLOAT')
# stems for QC
for nm, b in (('voice', mix_v), ('music', mix_m), ('sfx', mix_s)): sf.write(f'/tmp/jobs/serhat/out/stem_{nm}.wav', b[:len(mix)], SR, subtype='FLOAT')
print('raw written', len(mix) / SR, 'peak', 20 * np.log10(np.abs(mix).max()))
