"""Settle an unclear spoken line: score candidate transcripts against the audio with a CTC model (torchaudio MMS_FA).
Lower = better fit (0.0 is the best candidate). Use it instead of trusting a doubtful Whisper line.
Usage: python score_lines.py take.wav START END "candidate one" "candidate two" ..."""
import sys
import soundfile as sf, torch, torchaudio
from torchaudio.pipelines import MMS_FA as bundle

path, a, b, cands = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
labels = bundle.get_dict(star=None)
TR = str.maketrans({'ç': 'c', 'ş': 's', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ü': 'u', 'â': 'a', 'î': 'i', 'û': 'u',
                    'Ç': 'c', 'Ş': 's', 'Ğ': 'g', 'İ': 'i', 'I': 'i', 'Ö': 'o', 'Ü': 'u'})


def romanize(text):
    return ''.join(c for c in text.translate(TR).lower() if c in labels and c not in "-'*")


x, sr = sf.read(path, dtype='float32')
if x.ndim > 1:
    x = x.mean(1)
wav = torchaudio.functional.resample(torch.from_numpy(x[int(a * sr):int(b * sr)].copy())[None], sr, bundle.sample_rate)
with torch.inference_mode():
    emission, _ = bundle.get_model(with_star=False).eval()(wav)
logp = torch.log_softmax(emission[0], -1)


def nll(text):
    target = torch.tensor([[labels[c] for c in romanize(text)]], dtype=torch.long)
    return float(torch.nn.functional.ctc_loss(logp[:, None], target, torch.tensor([logp.shape[0]]),
                                              torch.tensor([target.shape[1]]), blank=0, reduction='sum', zero_infinity=True))


scores = sorted((nll(c), c) for c in cands)
for s, c in scores:
    print(f'{s - scores[0][0]:7.1f}  {c}')
