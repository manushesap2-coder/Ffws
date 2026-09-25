import torch, torchaudio, json, sys, soundfile as sf, numpy as np
from torchaudio.pipelines import MMS_FA as bundle
wav_path, out_path = sys.argv[1], sys.argv[2]
text = sys.argv[3]
words_disp = text.split()
def romanize(w):
    tr = str.maketrans({'ç':'c','ş':'s','ğ':'g','ı':'i','ö':'o','ü':'u','â':'a','î':'i','û':'u','Ç':'c','Ş':'s','Ğ':'g','İ':'i','I':'i','Ö':'o','Ü':'u'})
    w = w.translate(tr).lower()
    return ''.join(ch for ch in w if ch in bundle.get_labels() and ch != "'" )
words = [romanize(w) for w in words_disp]
x, sr = sf.read(wav_path, dtype='float32')
if x.ndim > 1: x = x.mean(1)
wave = torch.from_numpy(x)[None]
if sr != bundle.sample_rate:
    wave = torchaudio.functional.resample(wave, sr, bundle.sample_rate)
model = bundle.get_model(with_star=False)
dictionary = bundle.get_dict(star=None)
with torch.inference_mode():
    emission, _ = model(wave)
tokens = [dictionary[c] for w in words for c in w]
aligned, scores = torchaudio.functional.forced_align(emission, torch.tensor([tokens], dtype=torch.int32), blank=0)
aligned, scores = aligned[0], scores[0].exp()
token_spans = torchaudio.functional.merge_tokens(aligned, scores)
ratio = wave.shape[1] / emission.shape[1] / bundle.sample_rate
out = []; k = 0
for wd, w in zip(words_disp, words):
    spans = token_spans[k:k+len(w)]; k += len(w)
    s = spans[0].start * ratio; e = spans[-1].end * ratio
    sc = float(np.mean([sp.score for sp in spans]))
    out.append({'w': wd, 's': round(s, 3), 'e': round(e, 3), 'score': round(sc, 3)})
json.dump(out, open(out_path, 'w'), ensure_ascii=False, indent=1)
for o in out: print(f"{o['w']:22s} {o['s']:7.3f} {o['e']:7.3f} {o['score']:.2f}")
