import sys, json, glob, os, time
from faster_whisper import WhisperModel
model_name = sys.argv[1] if len(sys.argv) > 1 else "large-v3"
files = sys.argv[2:] or sorted(glob.glob('/tmp/edit/work/audio/*.16k.wav'))
t0 = time.time()
m = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
print("loaded", time.time()-t0, flush=True)
for f in files:
    b = os.path.basename(f).replace('.16k.wav', '')
    segs, info = m.transcribe(f, language="tr", word_timestamps=True, beam_size=5, vad_filter=False,
                              condition_on_previous_text=False)
    out = []
    for s in segs:
        out.append({"start": s.start, "end": s.end, "text": s.text,
                    "words": [{"w": w.word, "s": w.start, "e": w.end, "p": w.probability} for w in (s.words or [])]})
    json.dump(out, open(f'/tmp/edit/work/tx/{b}.{model_name}.json', 'w'), ensure_ascii=False, indent=1)
    print(f"== {b} ({time.time()-t0:.0f}s)", flush=True)
    for s in out:
        print(f"  [{s['start']:6.2f}-{s['end']:6.2f}] {s['text']}", flush=True)
