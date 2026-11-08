"""フェリス：不死鳥の帽子、背中に広がる炎の羽、黄緑のタンクトップと青いスカート。"""

import numpy as np

from ..body import Character, Head, SIDES, finish
from ..builder import Model
from ..textures import TextureSet
from ..geom import catmull, ellipsoid, tube
from .. import parts, physics

HAT = (0.82, 0.38, 0.45)
HAIR = (0.93, 0.66, 0.38)
BANG = (0.82, 0.84, 0.38)
TOP = (0.83, 0.85, 0.40)
SKIRT = (0.30, 0.42, 0.88)
FLAME = (1.0, 0.55, 0.06)
FLAME_IN = (1.0, 0.85, 0.25)
BOOT = (0.66, 0.32, 0.22)


def build(out_dir):
    m = Model("フェリス", "Feris", "フェリス（非公式・二次創作モデル）", 150)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(1.0, 0.87, 0.79), eye_color=(0.88, 0.48, 0.10), bust=1.18,
                  eye=dict(yaw=0.60, h=0.44, rx=0.0205, ry=0.0175, top=0.92, iris=0.26, iris_in=0.10),
                  mouth=dict(rx=0.0140, ry=0.0055, smile=-1.0))
    m.material("top", "タンクトップ", TOP, toon=0)
    m.material("skirt", "スカート", SKIRT, double=True, toon=0)
    m.material("shorts", "インナー", SKIRT, toon=0)
    m.material("boot", "ブーツ", BOOT, toon=0)
    m.material("sock", "靴下", (0.97, 0.96, 0.94), toon=0)
    c.hair_material("hair", "髪", HAIR)
    c.hair_material("bang", "前髪（黄緑）", BANG)
    m.material("hat", "帽子", HAT, double=True, toon=0)
    m.material("hat_eye", "帽子の目", (0.08, 0.06, 0.06), toon=0)
    m.material("feather", "白い羽", (0.98, 0.98, 0.96), double=True, toon=0)
    m.material("flame", "炎の羽", FLAME, double=True, toon=1)
    m.material("flame_in", "炎の芯", FLAME_IN, double=True, toon=1)

    c.build_body(torso_bands=[(1.0, "skin"), (0.770, "top"), (0.650, "skin"), (0.585, "shorts")],
                 leg_bands=[(1.0, "skin"), (0.160, "sock"), (0.120, "boot")], arm_bands=[(-1.0, "skin")])
    c.build_face(closed_mouth_color=(0.62, 0.30, 0.32))

    torso = c.skin_torso()
    # 肩ひも
    for jp, en, s in SIDES:
        strap = catmull([[s * 0.040, 0.772, -0.040], [s * 0.050, 0.796, -0.012], [s * 0.050, 0.796, 0.020],
                         [s * 0.042, 0.772, 0.044]], 12)
        m.add("top", tube(strap, 0.0055, 0.003, n=8), torso)
    m.add("skirt", parts.skirt(0.592, 0.455, 1.85, pleats=12, depth=0.12), physics.skirt(c, 0.592, 0.455, 1.85))
    for jp, en, s in SIDES:
        m.add("boot", parts.leg_tube(s, 0.120, 0.030, 1.30, steps=4), c.skin_leg(jp))
        m.add("boot", parts.foot(s, r=(0.028, 0.028, 0.064), lift=0.002), m.skin_fixed(jp + "足首"))

    hair(c)
    hat(c)
    wings(c)
    finish(m)
    return m


def hair(c):
    c.hair_cap("hair")
    for th in np.linspace(-0.9, 0.9, 11):
        mat = "bang" if abs(th) < 0.30 else "hair"
        c.bang(mat, th, 0.20 - 0.25 * abs(th), 0.022, e_start=1.15, pointy=True, lift_end=1.10, sway=0.25 * th)
    for th in np.linspace(0.95, np.pi, 8):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            c.drape("hair", sg * th, 0.790, 0.030, flare=0.0, pointy=True, wave=0.004)


def hat(c):
    m = c.m
    h = c.head
    hc = h.c + np.array([0, 0.014, 0.004])
    hr = h.r * np.array([1.20, 1.13, 1.20])

    def keep(d):
        return d[:, 1] > 0.36 - 0.55 * np.clip(d[:, 2], 0, 1)

    m.add("hat", ellipsoid(hc, hr, nu=36, nv=22, keep=keep), m.skin_fixed("頭"))
    for sg in (1, -1):
        d = Head.dir(sg * 0.50, 0.66)
        m.add("hat_eye", ellipsoid(hc + d * hr * 1.005, [0.008, 0.010, 0.008], nu=12, nv=8), m.skin_fixed("頭"))
        # 帽子の横の白い羽
        root = hc + Head.dir(sg * 1.55, 0.45) * hr
        for k, (dy, ln) in enumerate([(0.040, 0.050), (0.018, 0.058), (-0.004, 0.044)]):
            tip = root + np.array([sg * ln, dy + 0.012, 0.012 + 0.008 * k])
            path = catmull([root, (root + tip) / 2 + np.array([0, 0.006, 0]), tip], 10)
            w = np.linspace(0.010, 0.002, len(path))
            c.strand(path, w, 0.003, np.tile([0, 1.0, 0], (len(path), 1)), "feather", skin=m.skin_fixed("頭"))
    # とさか（アホ毛）
    top = hc + np.array([0, hr[1], 0])
    crest = catmull([top - np.array([0, 0.006, 0]), top + np.array([0.0, 0.035, 0.010]),
                     top + np.array([0.0, 0.055, -0.020]), top + np.array([0.0, 0.040, -0.040])], 18)
    c.strand(crest, np.linspace(0.016, 0.003, len(crest)), 0.006, np.tile([1.0, 0, 0], (len(crest), 1)), "hat",
             skin=m.skin_fixed("頭"))


def wings(c):
    """背中から左右に広がる炎の羽。"""
    m = c.m
    skin = m.skin_fixed("上半身2")
    for sg in (1, -1):
        root = np.array([sg * 0.040, 0.750, 0.062])
        tongues = [(0.75, 0.17), (0.45, 0.24), (0.15, 0.29), (-0.15, 0.27), (-0.45, 0.22), (-0.75, 0.15)]
        for k, (ang, ln) in enumerate(tongues):
            d = np.array([sg * np.cos(ang), np.sin(ang), 0.30])
            d /= np.linalg.norm(d)
            perp = np.array([-np.sin(ang) * sg, np.cos(ang), 0.0])
            mid = root + d * ln * 0.55 + perp * 0.025 * np.sin(k * 2.1)
            tip = root + d * ln + perp * 0.04 * np.cos(k * 1.7) + np.array([0, 0.02, 0.03])
            path = catmull([root, mid, tip], 18)
            t = np.linspace(0, 1, len(path))
            prof = np.sin(np.pi * np.clip(t * 0.95 + 0.12, 0, 1)) ** 0.7
            sides = np.tile(perp, (len(path), 1))
            c.strand(path, 0.050 * prof + 0.002, 0.006, sides, "flame", skin=skin)
            inner = path[: int(len(path) * 0.75)] + np.array([0, 0, -0.005])
            t2 = np.linspace(0, 1, len(inner))
            c.strand(inner, 0.026 * np.sin(np.pi * np.clip(t2 * 0.95 + 0.12, 0, 1)) ** 0.7 + 0.001, 0.004,
                     sides[: len(inner)], "flame_in", skin=skin)
