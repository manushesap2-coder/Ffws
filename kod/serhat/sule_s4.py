"""Serhat v4 ending: Şule's CTA clip (Pocket 0083), graded + captions + logo end card -> out/sule_v4.mp4 (local time 0..D_SULE)."""
import sys, json, subprocess, numpy as np, cv2
sys.argv = sys.argv[:1]
import comp_s4 as C
import tl_s4 as TL
from sgrade2 import smooth_curve
W, H, FPS = 1080, 1920, 30
SRC = '/tmp/jobs/serhat/src3/DJI_20260920183210_0083_D.MP4'
C.init_gfx(); SPR = C.SPR
al = json.load(open('/tmp/jobs/serhat/work/s83_align.json'))
WD = [dict(w=a['w'], T=a['s'] - TL.SULE_S0, Te=a['e'] - TL.SULE_S0) for a in al]
CH = [WD[0:4], WD[4:6], WD[6:10], WD[10:13], WD[13:14], WD[14:17], WD[17:20]]
D = TL.D_SULE; NF = int(round(D * FPS)); T_LOGO = WD[18]['T'] - 0.1          # logo lands on "Serhat"
def grade(x):
    x = x * np.array([1.035, 1.0, 0.955], np.float32)                        # overcast -> a touch warmer
    L = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    base = cv2.GaussianBlur(L, (0, 0), 24)
    L2 = np.clip(L + (L - base) * 0.35, 0, 1)                                   # local contrast
    L3 = smooth_curve(L2, 1.12, 0.46)
    x = x * ((L3 + 1e-4) / (L + 1e-4))[..., None]
    m = x.mean(-1, keepdims=True); x = m + (x - m) * 1.10                      # saturation
    bl = cv2.GaussianBlur(x, (0, 0), 1.2); x = x + (x - bl) * 0.35             # gentle sharpening
    return np.clip(x, 0, 1)
def logo_card(frame, T):
    if T < T_LOGO: return
    k = C.ease_io(np.clip((T - T_LOGO) / 0.6, 0, 1))
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    frame *= 1 - 0.72 * k * np.clip(1 - yy / 0.42, 0, 1) ** 1.3                 # top scrim for legibility
    lg = SPR['logo']; s = 0.78
    C.pop(frame, cv2.resize(lg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), W / 2, 300, T, T_LOGO, D + 1, rise=40, back=1.6)
    C.pop(frame, SPR['f_proj'], W / 2, 300 + lg.shape[0] * s / 2 + 70, T, T_LOGO + 0.25, D + 1, rise=24, back=1.3, scale0=0.9)
def main():
    rd = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{TL.SULE_S0:.3f}', '-i', SRC, '-vf', f'scale={W}:{H}:flags=lanczos',
                           '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], stdout=subprocess.PIPE, bufsize=10 ** 8)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart',
                            '/tmp/jobs/serhat/out/sule_v4.mp4'], stdin=subprocess.PIPE)
    fps_src = 60000 / 1001; got = -1; cur = None; rng = np.random.default_rng(3)
    only = [int(x) for x in sys.argv_frames] if hasattr(sys, 'argv_frames') else None
    for i in range(NF):
        u = i / FPS; want = int(round(min(u, TL.SULE_S1 - TL.SULE_S0 - 0.02) * fps_src))
        while got < want:
            b = rd.stdout.read(W * H * 6)
            if len(b) < W * H * 6: break
            cur = np.frombuffer(b, np.uint16).reshape(H, W, 3); got += 1
        img = grade(cur.astype(np.float32) / 65535.0)
        z = 1.0 + 0.06 * (u / D); cx, cy = W / 2, 860
        M = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        for j, c in enumerate(CH):
            end = min(c[-1]['Te'] + 0.35, CH[j + 1][0]['T'] - 0.04) if j + 1 < len(CH) else c[-1]['Te'] + 0.45
            if c[0]['T'] - 0.05 <= u <= end: C.draw_caption(img, u, c, cy=1585)
        logo_card(img, u)
        img = C.finish(np.clip(img, 0, 1), u)
        fo = np.clip((u - (D - 0.6)) / 0.6, 0, 1); img = img * (1 - fo)
        d = rng.uniform(-0.5, 0.5, (H, W, 1)).astype(np.float32) / 255.0
        enc.stdin.write(((img + d) * 255 + 0.5).clip(0, 255).astype(np.uint8).tobytes())
        if i in (15, 120, 230, 290): cv2.imwrite(f'/tmp/jobs/serhat/out/frames4/sule_{i:03d}.png', (img[..., ::-1] * 255).astype(np.uint8))
    rd.kill(); enc.stdin.close(); enc.wait(); print('DONE sule', NF, 'frames', D)
if __name__ == '__main__':
    main()
