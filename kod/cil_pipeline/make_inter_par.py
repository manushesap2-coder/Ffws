import subprocess, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/tmp/edit/pipeline')
from config import *

ENC = ['-c:v', 'prores_ks', '-profile:v', '3', '-pix_fmt', 'yuv422p10le', '-an']
jobs = []
for name, (src, t_in, dur) in SHOTS.items():
    if os.path.exists(INTER + f"{name}.mov"):
        continue
    jobs.append((dur, name, ['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-ss', f'{t_in:.3f}', '-i', src,
                             '-t', f'{dur:.3f}', '-vf', f'scale={DW}:{DH}:flags=lanczos'] + ENC))
for name, t_in, dur in [('SPK_A', 0.90, 7.40), ('SPK_E', 19.40, 2.40)]:
    if os.path.exists(INTER + f"{name}.mov"):
        continue
    jobs.append((dur, name, ['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-ss', f'{t_in:.3f}', '-i', SPEAKER_SRC,
                             '-t', f'{dur:.3f}'] + ENC))
jobs.sort(reverse=True)  # longest first


def run(job):
    dur, name, cmd = job
    tmp = INTER + f"{name}.part.mov"
    r = subprocess.run(cmd + [tmp], capture_output=True, text=True)
    if r.returncode == 0:
        os.rename(tmp, INTER + f"{name}.mov")
    print(name, 'OK' if r.returncode == 0 else 'FAIL ' + r.stderr[-300:], flush=True)


with ThreadPoolExecutor(2) as ex:
    list(ex.map(run, jobs))
print('ALL DONE', flush=True)
