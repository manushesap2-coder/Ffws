"""Person mattes (rembg u2net_human_seg) for putting a title BEHIND the speaker. Writes quarter-resolution masks
m_<frame>.png for source frames FIRST..LAST of the clip.
Usage: python person_matte.py clip.mov out_dir FIRST LAST [SRC_W SRC_H]   (default source size 1728x3072, DJI Pocket)
Compositing: draw the title on a copy, then img = titled * (1 - m) + original * m, with m warped to the output by the
shot's affine scaled x4."""
import os, subprocess, sys
import numpy as np, cv2
from rembg import remove, new_session

clip, out, f0, f1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sw, sh = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (1728, 3072)
w, h = sw // 4, sh // 4
os.makedirs(out, exist_ok=True)
session = new_session('u2net_human_seg')
proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', clip, '-vf', f'select=between(n\\,{f0}\\,{f1}),scale={w}:{h}',
                         '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
k = f0
while True:
    buf = proc.stdout.read(w * h * 3)
    if len(buf) < w * h * 3:
        break
    mask = remove(np.frombuffer(buf, np.uint8).reshape(h, w, 3), session=session, only_mask=True, post_process_mask=True)
    cv2.imwrite(os.path.join(out, f'm_{k:05d}.png'), np.asarray(mask))
    k += 1
print('mattes', k - f0)
