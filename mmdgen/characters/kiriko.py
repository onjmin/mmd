"""蓄音キリコ：若草色のサイドポニー＋片目隠しの斜め前髪、赤褐色の瞳、エスキモー風のファー付きコート。"""

import numpy as np

from ..body import Character, SIDES, finish, joints
from ..geom import loft, smoothstep
from ..builder import Model
from ..textures import TextureSet
from ..geom import catmull, ellipsoid, ring, torus, tube, TAU
from .. import parts, physics

SUIT = (0.13, 0.13, 0.16)
COAT = (0.66, 0.52, 0.38)  # 公式の絵の明るい黄土色
FUR = (0.95, 0.88, 0.74)
GLOVE = (0.42, 0.20, 0.13)
HAIR = (0.50, 0.80, 0.24)
TIP = (0.30, 0.74, 0.72)
BOW = (0.38, 0.15, 0.11)


def fur(path, r, closed=False, bumps=None, n=16):
    """ファーの縁：毛のかたまりが並んだようにふくらみを付けた太い管。bumps はふくらみの数（既定は長さから決める）。"""
    path = np.asarray(path, float)
    if closed:
        path = np.vstack([path, path[:1]])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    k = bumps or max(4, int(round(t[-1] / (r * 1.6))))
    bump = 1 + 0.16 * np.abs(np.sin(np.pi * k * t / t[-1]))
    # 断面にも小さな毛の凹凸を付け、なめらかな筒に見えないようにする
    return tube(path, r * bump, n=n, rmod=lambda a: 1 + 0.06 * np.sin(a * 7))


def build(out_dir):
    m = Model("蓄音キリコ", "Chikune Kiriko", "蓄音キリコ（非公式・二次創作モデル）", 158)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(0.86, 0.58, 0.36), eye_color=(0.66, 0.24, 0.12), bust=0.95,
                  mouth=dict(rx=0.0094, ry=0.0030, smile=-0.6),
                  eye=dict(rx=0.016, ry=0.0125, top=0.88, iris=0.70, iris_h=1.05, iris_y=-0.06, iris_in=0.05))
    m.material("suit", "インナー", SUIT, toon=0)
    m.material("coat", "コート", COAT, double=True, toon=1)  # 陰影を出してしわやたるみを見せる
    m.material("fur", "ファー", FUR, toon=1)
    m.material("glove", "手袋", GLOVE, toon=0)
    m.material("skirt", "スカート", COAT, double=True, toon=1)
    c.hair_material("hair", "髪", HAIR)
    c.hair_material("tip", "毛先", TIP)
    m.material("bow", "リボン", BOW, toon=0)

    c.build_body(torso_bands=[(1.0, "suit")], leg_bands=[(1.0, "suit")], arm_bands=[(-1.0, "suit")],
                 neck_mat="suit", hand_mat="glove")
    c.build_face(closed_mouth_color=(0.40, 0.11, 0.12))

    torso = c.skin_torso()
    # ---- コート（公式の絵）：ゆったりしたショート丈。肩を覆い、前は開いて、裾へ広がる。しわとたるみを入れる ----
    a0, a1 = np.radians(-90 + 26), np.radians(270 - 26)
    rng = np.random.default_rng(5)
    ph = rng.uniform(0, TAU, 4)

    def grow(y):  # 胴からの膨らみ：肩はぴったり、胸の下から裾へ広がる
        return 1.16 + 0.10 * float(np.clip((0.800 - y) / 0.03, 0, 1)) + 0.30 * smoothstep(0.735, 0.645, y)

    def coat_ring(y, n=150, off=0.0):
        rx, rz, cz = parts.torso_r(y, grow(y))
        a = np.linspace(a0, a1, n)
        t = smoothstep(0.80, 0.65, y)  # 下ほど深いしわ
        fold = 1 + (0.012 + 0.040 * t) * (np.sin(a * 7 + ph[0] + y * 30) * 0.6 + np.sin(a * 13 + ph[1] - y * 55) * 0.4)
        pts = np.stack([rx * fold * np.cos(a), np.full(n, y), cz + rz * fold * np.sin(a)], 1)
        return pts + off
    ys = np.linspace(0.828, 0.645, 46)
    rings = []
    for y in ys:
        r = coat_ring(y)
        # 裾は波打たせる（たるみ）
        if y < 0.66:
            a = np.linspace(a0, a1, len(r))
            r[:, 1] += 0.006 * np.sin(a * 5 + ph[2]) * (0.66 - y) / 0.015
        rings.append(r)
    m.add("coat", loft(rings, closed=False, centers=[np.array([0, y, parts.torso_r(y)[2]]) for y in ys]), torso)
    # 首を包んであごの下まで立つファーの襟（前は少し開く）と、肩へ広がるファーのショール
    neck_c = np.array([0.0, 0.0, 0.004])
    for y, r, th in [(0.834, 0.034, 0.016), (0.846, 0.031, 0.014), (0.857, 0.028, 0.012)]:
        arc = ring(neck_c + [0, y, 0], [1, 0, 0], [0, 0, 1], r, r * 0.95, 90, np.radians(-90 + 22), np.radians(270 - 22), closed=False)
        m.add("fur", fur(arc, th, n=16), m.skin_auto(["上半身2", "首"]))
    for y, sc in [(0.822, 1.00), (0.810, 1.06)]:
        rx, rz, cz = parts.torso_r(y, grow(y) * sc)
        arc = ring([0, y + 0.004, cz], [1, 0, 0], [0, 0, 1], rx + 0.004, rz + 0.006, 120, a0 + 0.15, a1 - 0.15, closed=False)
        m.add("fur", fur(arc, 0.013 if y > 0.815 else 0.011, n=16), m.skin_auto(["上半身2", "首"]))
    # 裾と前の縁のファー
    hem = coat_ring(0.645, 160)
    a = np.linspace(a0, a1, 160)
    hem[:, 1] += 0.006 * np.sin(a * 5 + ph[2]) * (0.66 - 0.645) / 0.015
    m.add("fur", fur(hem, 0.011), torso)
    for k in (0, -1):
        edge = np.array([coat_ring(y, 150)[k] for y in np.linspace(0.826, 0.648, 40)])
        m.add("fur", fur(edge, 0.010), torso)

    # ---- スカート：黄土色のひだが 2 段に重なって広がる。腰にファー ----
    sk = physics.skirt(c, 0.590, 0.490, 2.10)
    m.add("skirt", parts.skirt(0.590, 0.530, 1.70, pleats=22, depth=0.12, n=220, steps=12), sk)
    m.add("skirt", parts.skirt(0.556, 0.492, 2.10, pleats=28, depth=0.14, n=260, steps=14), sk)
    m.add("fur", fur(ring([0, 0.588, 0.006], [1, 0, 0], [0, 0, 1], 0.064, 0.048, 120), 0.009, closed=True),
          m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        # 袖：大きくふくらみ、何段もしわが寄って、手首でファーの袖口に絞られる
        knots = [0.044, 0.054, 0.061, 0.064, 0.062, 0.054, 0.036]
        tt = np.linspace(0, 1, 44)
        roll = 1 + 0.07 * np.sin(tt * np.pi * 9 + (0.6 if s > 0 else 1.7)) * np.sin(np.pi * tt) ** 0.7
        radii = np.interp(tt, np.linspace(0, 1, len(knots)), knots) * roll
        m.add("coat", parts.sleeve(s, -0.012, 0.250, radii, n=56,
                                   rmod=lambda a: 1 + 0.035 * np.sin(a * 6) + 0.015 * np.sin(a * 11 + 1.0)), arm)
        cuff = [parts.arm_point(s, 0.255) + v for v in
                ring([0, 0, 0], *_perp(parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)), 0.034, 0.034, 72)]
        m.add("fur", fur(cuff, 0.014, closed=True, bumps=16), arm)
        # ブーツ：ふくらはぎの下までの太いムクルク。大きなファーの縁と丸いつま先
        leg = c.skin_leg(jp)
        m.add("coat", parts.leg_tube(s, 0.128, 0.030, lambda t: 1.55 + 0.25 * t, steps=22, n=56,
                                     rmod=lambda a: 1 + 0.03 * np.sin(a * 5)), leg)
        top = parts.leg_at(0.130, s)[1]
        m.add("fur", fur(ring(top, [1, 0, 0], [0, 0, 1], 0.050, 0.050, 90), 0.019, closed=True, bumps=16), leg)
        m.add("coat", ellipsoid(np.array([joints(s)["足首"][0], 0.032, -0.026]), [0.040, 0.036, 0.076], nu=48, nv=28),
              m.skin_fixed(jp + "足首"))

    hair(c)
    finish(m)
    return m


def _perp(d):
    """向き d に直交する 2 本の単位ベクトル。"""
    d = np.asarray(d, float) / np.linalg.norm(d)
    ref = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def hair(c):
    m = c.m
    c.hair_cap("hair")
    # 原画のように髪は大きなかたまりで描き、細い束は並べない。
    # 斜めぱっつんの前髪：太い束 7 本。右目（-X 側）を隠し、左は眉の上で切れる（目の高さから決める）
    top = c.eye["pitch"] + 0.58
    for th in np.linspace(-0.95, 0.95, 7):
        cover = np.clip((0.30 - th) / 0.85, 0, 1)
        e_end = top - 0.74 * cover
        c.bang("hair", th, e_end, 0.0330, e_start=1.25, pointy=False, lift_end=1.065 + 0.03 * cover, sway=-0.12 * cover)
    # あごの長さのボブ：横は口の下、後ろはうなじで切れ、毛先は外へはねる。目尻より外から始める
    for th in np.linspace(1.35, np.pi, 5):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            y_end = 0.850 - 0.014 * (th - 1.35) / (np.pi - 1.35)
            c.drape("hair", sg * th, y_end, 0.052, flare=0.0, pointy=True, splay=0.30, wave=0.003)
    # 右の耳の後ろで結んだサイドポニー。肩の後ろを波打って腰の上まで垂れ、毛先が水色になる
    base = np.array([-0.052, 0.928, 0.050])
    for k, (dx, dz, ln) in enumerate([(0.000, 0.000, 1.00), (-0.010, 0.010, 0.90), (0.008, -0.008, 0.80)]):
        ctrl = [base, [-0.098, 0.902, 0.058], [-0.130 + dx, 0.835, 0.060 + dz], [-0.128 + dx, 0.760, 0.066 + dz],
                [-0.140 + 1.4 * dx, 0.690, 0.060 + dz], [-0.124 + 2.0 * dx, 0.600, 0.052 + dz]]
        path = catmull(ctrl, 30)
        path = path[: int(len(path) * ln)]
        n = len(path)
        t = np.linspace(0, 1, n)
        path = path + np.outer(0.010 * np.sin(t * 9 + k * 1.3) * np.clip(t * 3, 0, 1), [1.0, 0, 0.4])
        mats = ["tip" if p[1] < 0.715 else "hair" for p in path]
        c.strand(path, 0.036 * np.sin(np.pi * np.clip(t * 0.9 + 0.1, 0, 1)) ** 0.6 + 0.004,
                 0.012 * (1 - 0.6 * t) + 0.003, np.tile([0, 0, 1.0], (n, 1)), mats,
                 skin=c.hair_chain(path, 0.92, 4, 0.020))
    # 頭の上の後ろ寄りに大きなリボン。正面から見ると頭の上の左右に輪が出る（pixiv の全身絵・おんJRPG の立ち絵）
    for mesh in parts.bow([0.0, 0.992, 0.040], 0.062, 0.038, tilt=0.20, facing=(0.0, 0.55, 1.0)):
        m.add("bow", mesh, m.skin_fixed("頭"))
