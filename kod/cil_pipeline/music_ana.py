import librosa, numpy as np, json, glob, os, sys
names = {d['url'].split('/')[-1].replace('.mp3',''): d['title'] for d in json.load(open('/tmp/edit/music/tracks.json'))}
res = []
for f in sorted(glob.glob('/tmp/edit/music/cand/*.mp3')):
    tid = os.path.basename(f)[:-4]
    y, sr = librosa.load(f, sr=22050, mono=True)
    dur = len(y)/sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    tempo = float(np.atleast_1d(tempo)[0])
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    # 1s smoothed energy in dB
    win = int(sr/hop)
    sm = np.convolve(rms, np.ones(win)/win, mode='same')
    db = 20*np.log10(sm+1e-6)
    # energy per 2s block
    blocks = [float(np.mean(db[(t>=a)&(t<a+2)])) for a in np.arange(0, min(dur,60), 2)]
    # biggest rises (drops) within first 60s: diff over 1.5 s
    lag = int(1.5*sr/hop)
    rise = db[lag:] - db[:-lag]
    idx = np.argsort(rise)[::-1]
    drops = []
    for i in idx:
        tt = t[i+lag]
        if tt > 60: continue
        if all(abs(tt-d[0])>4 for d in drops):
            drops.append((round(float(tt),2), round(float(rise[i]),1)))
        if len(drops) >= 3: break
    # percussive ratio
    H, P = librosa.effects.hpss(y[:sr*40])
    pr = float(np.sum(P**2)/(np.sum(H**2)+np.sum(P**2)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y[:sr*40], sr=sr)))
    res.append((tid, names.get(tid,'?'), round(dur,1), round(tempo,1), round(pr,2), int(cent), drops, [round(b) for b in blocks]))
    print(f"{tid:5} {names.get(tid,'?')[:22]:22} dur={dur:5.1f} bpm={tempo:6.1f} perc={pr:.2f} cent={cent:5.0f} drops={drops}\n      E2s={[round(b) for b in blocks]}", flush=True)
json.dump(res, open('/tmp/edit/music/analysis.json','w'))
