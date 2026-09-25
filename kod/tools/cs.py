"""Contact sheet: python cs.py video out.jpg t0 t1 step cols thumb_w"""
import sys, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
v, out, t0, t1, step, cols, tw = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
ts = list(np.arange(t0, t1, step))
def grab(t):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', v, '-frames:v', '1', '-vf', f'scale={tw}:-2', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True)
    import io
    try: return Image.open(io.BytesIO(p.stdout)).convert('RGB')
    except Exception: return None
with ThreadPoolExecutor(4) as ex: ims = list(ex.map(grab, ts))
ims = [i for i in ims if i is not None]
th = ims[0].size[1]
rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * th), (0, 0, 0))
f = ImageFont.truetype('/tmp/edit/fonts/Poppins-Bold.ttf', max(12, tw // 9))
d = ImageDraw.Draw(sheet)
for k, (im, t) in enumerate(zip(ims, ts)):
    x, y = (k % cols) * tw, (k // cols) * th
    sheet.paste(im, (x, y))
    d.text((x + 4, y + 2), f'{t:.1f}', font=f, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
sheet.save(out, quality=88)
print(out, sheet.size, len(ims))
