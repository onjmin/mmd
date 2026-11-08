"""蓄音キリコ：若草色のサイドポニー＋片目隠しの斜め前髪、赤褐色の瞳、エスキモー風のファー付きコート。"""

import numpy as np

from ..body import Character, SIDES, finish, joints
from ..builder import Model
from ..textures import TextureSet
from ..geom import catmull, ellipsoid, ring, torus, tube, TAU
from .. import parts, physics

SUIT = (0.13, 0.13, 0.16)
COAT = (0.50, 0.34, 0.22)
FUR = (0.95, 0.88, 0.74)
GLOVE = (0.42, 0.20, 0.13)
HAIR = (0.50, 0.80, 0.24)
TIP = (0.30, 0.74, 0.72)
BOW = (0.58, 0.22, 0.16)


def build(out_dir):
    m = Model("蓄音キリコ", "Chikune Kiriko", "蓄音キリコ（非公式・二次創作モデル）", 158)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(0.98, 0.80, 0.56), eye_color=(0.66, 0.24, 0.12), bust=0.95,
                  mouth=dict(rx=0.0125, ry=0.0040, smile=-0.8),
                  eye=dict(yaw=0.58, h=0.45, rx=0.022, ry=0.016, top=0.80, iris=0.40, iris_in=0.14))
    m.material("suit", "インナー", SUIT, toon=0)
    m.material("coat", "コート", COAT, double=True, toon=0)
    m.material("fur", "ファー", FUR, toon=1)
    m.material("glove", "手袋", GLOVE, toon=0)
    m.material("skirt", "スカート", COAT, double=True, toon=0)
    c.hair_material("hair", "髪", HAIR)
    c.hair_material("tip", "毛先", TIP)
    m.material("bow", "リボン", BOW, toon=0)
    m.material("mouth_in", "口の中", (0.35, 0.10, 0.10), edge=False, toon=0)

    c.build_body(torso_bands=[(1.0, "suit")], leg_bands=[(1.0, "suit")], arm_bands=[(-1.0, "suit")],
                 neck_mat="suit", hand_mat="glove")
    c.build_face(closed_mouth_color=(0.40, 0.11, 0.12))

    torso = c.skin_torso()
    fluffy = lambda a: 1 + 0.12 * np.abs(np.sin(a * 9))
    # 前の開いたショート丈のコート（-90° が正面）
    a0, a1 = np.radians(-90 + 28), np.radians(270 - 28)
    m.add("coat", parts.band_around_torso(0.800, 0.650, lambda y: 1.20 if y < 0.79 else 1.35, a0, a1, steps=8),
          torso)
    m.add("fur", torus([0, 0.800, 0.004], [0, 1, 0], 0.046, 0.017), m.skin_auto(["上半身2", "首"]))
    rx, rz, cz = parts.torso_r(0.650, 1.2)
    hem = ring([0, 0.650, cz], [1, 0, 0], [0, 0, 1], rx, rz, 40, a0, a1, closed=False)
    m.add("fur", tube(hem, 0.009, n=8, rmod=fluffy), torso)
    for a in (a0, a1):
        edge = []
        for y in np.linspace(0.795, 0.652, 8):
            rx, rz, cz = parts.torso_r(y, 1.20 if y < 0.79 else 1.35)
            edge.append([rx * np.cos(a), y, cz + rz * np.sin(a)])
        m.add("fur", tube(edge, 0.009, n=8), torso)
    # スカート
    m.add("skirt", parts.skirt(0.590, 0.470, 1.75, pleats=14, depth=0.10), physics.skirt(c, 0.590, 0.470, 1.75))
    m.add("fur", tube(ring([0, 0.588, 0.006], [1, 0, 0], [0, 0, 1], 0.062, 0.046, 36).tolist() +
                      [ring([0, 0.588, 0.006], [1, 0, 0], [0, 0, 1], 0.062, 0.046, 36)[0].tolist()], 0.007, n=8),
          m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        m.add("coat", parts.sleeve(s, -0.010, 0.255, [0.038, 0.045, 0.050, 0.052, 0.050, 0.042, 0.032]), arm)
        m.add("fur", parts.arm_torus(s, 0.258, 0.032, 0.012), arm)
        # ファー付きブーツ
        leg = c.skin_leg(jp)
        m.add("coat", parts.leg_tube(s, 0.170, 0.030, lambda t: 1.35 + 0.15 * t, steps=6), leg)
        m.add("fur", torus(parts.leg_at(0.172, s)[1], [0, 1, 0], 0.046, 0.014), leg)
        m.add("coat", parts.foot(s, r=(0.034, 0.034, 0.070), lift=0.004), m.skin_fixed(jp + "足首"))

    hair(c)
    finish(m)
    return m


def hair(c):
    m = c.m
    c.hair_cap("hair")
    # 斜めぱっつんの前髪。右目（-X 側）を隠す
    for th in np.linspace(-0.95, 0.95, 15):
        cover = np.clip((0.30 - th) / 0.85, 0, 1)
        e_end = 0.24 - 0.75 * cover
        c.bang("hair", th, e_end, 0.021, e_start=1.25, pointy=False, lift_end=1.07 + 0.03 * cover, sway=-0.12 * cover)
    # あごの長さのボブ：横は口の下、後ろはうなじで切れ、毛先は外へはねる
    for th in np.linspace(0.9, np.pi, 9):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            y_end = 0.850 - 0.014 * (th - 0.9) / (np.pi - 0.9)
            c.drape("hair", sg * th, y_end, 0.032, flare=0.0, pointy=True, splay=0.30, wave=0.003)
    # 右の耳の後ろで結んだサイドポニー。肩の後ろを波打って腰の上まで垂れ、毛先が水色になる
    base = np.array([-0.052, 0.928, 0.050])
    for k, (dx, dz, ln) in enumerate([(0.000, 0.000, 1.00), (-0.012, 0.010, 0.93), (0.010, 0.012, 0.88),
                                       (-0.006, -0.010, 0.82), (0.008, -0.006, 0.76)]):
        ctrl = [base, [-0.098, 0.902, 0.058], [-0.130 + dx, 0.835, 0.060 + dz], [-0.128 + dx, 0.760, 0.066 + dz],
                [-0.140 + 1.4 * dx, 0.690, 0.060 + dz], [-0.124 + 2.0 * dx, 0.600, 0.052 + dz]]
        path = catmull(ctrl, 30)
        path = path[: int(len(path) * ln)]
        n = len(path)
        t = np.linspace(0, 1, n)
        path = path + np.outer(0.010 * np.sin(t * 9 + k * 1.3) * np.clip(t * 3, 0, 1), [1.0, 0, 0.4])
        mats = ["tip" if p[1] < 0.715 else "hair" for p in path]
        c.strand(path, 0.028 * np.sin(np.pi * np.clip(t * 0.9 + 0.1, 0, 1)) ** 0.6 + 0.004,
                 0.012 * (1 - 0.6 * t) + 0.003, np.tile([0, 0, 1.0], (n, 1)), mats,
                 skin=c.hair_chain(path, 0.92, 4, 0.020))
    # 結び目の大きなリボン。頭の後ろから左右に輪が見える
    for mesh in parts.bow([-0.034, 0.940, 0.066], 0.058, 0.034, tilt=0.30, facing=(-0.25, 0.1, 1.0)):
        m.add("bow", mesh, m.skin_fixed("頭"))
