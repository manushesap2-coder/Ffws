import sys, subprocess, io, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
t0, t1, step, cols, tw, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
def loc(t):
    if t < 38.1: return 'D_A', t
    if t < 116.1: return 'D_B', t - 38.0
    return 'D_C', t - 116.0
def grab(t):
    n, lt = loc(t)
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{lt:.3f}', '-i', f'/tmp/jobs/serhat/inter/{n}.mp4', '-frames:v', '1', '-vf', f'scale={tw}:-2',
                        '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True)
    return Image.open(io.BytesIO(p.stdout)).convert('RGB')
ts = list(np.arange(t0, t1, step))
with ThreadPoolExecutor(4) as ex: ims = list(ex.map(grab, ts))
th = ims[0].size[1]; rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * th)); d = ImageDraw.Draw(sheet)
f = ImageFont.truetype('/tmp/jobs/fonts/Poppins-Bold.ttf', max(12, tw // 8))
for k, (im, t) in enumerate(zip(ims, ts)):
    x, y = (k % cols) * tw, (k // cols) * th; sheet.paste(im, (x, y))
    d.text((x + 4, y + 2), f'{t:.1f}', font=f, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
sheet.save(out, quality=86); print(out, sheet.size)
