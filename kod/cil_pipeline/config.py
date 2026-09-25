"""Shared edit configuration: sources, shots, timeline."""
SRC = "/tmp/edit/src/2.Video Grobeton"
POCKET = SRC + "/Pocket/"
DRONE = SRC + "/Drone/"
INTER = "/tmp/edit/inter/"
FPS = 30
W, H = 1080, 1920
# intermediate scale for drone (native 2160x3840): 0.6 -> 1296x2304 gives 1.2x zoom headroom
DW, DH = 1296, 2304

def drone(n):
    import glob
    return glob.glob(DRONE + f"*_{n:04d}_D.MP4")[0]

# B-roll shots: name -> (source, in_sec, dur_sec)
SHOTS = {
    "B1": (drone(137), 25.3, 3.0),
    "B2": (drone(136), 10.5, 3.0),
    "B3": (drone(139), 19.5, 2.5),
    "C1": (drone(138), 0.0, 3.0),
    "C2": (drone(139), 4.0, 2.5),
    "C3": (drone(137), 1.8, 2.5),
    "C4": (drone(125), 24.5, 2.5),
    "D1": (drone(142), 9.3, 3.2),
    "D2": (drone(131), 8.0, 5.0),
    "E2": (drone(132), 8.0, 14.0),
    "F1": (drone(123), 12.5, 2.5),
    "F2": (drone(137), 29.5, 2.0),
    "F3": (drone(133), 25.0, 2.5),
    "F4": (drone(136), 17.0, 2.0),
    "F5": (drone(122), 3.0, 9.0),
    "F6": (drone(131), 20.0, 6.0),
}
SPEAKER_SRC = POCKET + "DJI_20260921162258_0086_D.MP4"
