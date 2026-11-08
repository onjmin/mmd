"""束音ロゼ：黒髪の姫カット＋内側の緑、黒いチャイナ風トップスと大きな袖、炎の裾の赤い腰布。"""

import numpy as np

from ..body import Character, Head, SIDES, finish, joints
from ..builder import Model
from ..textures import TextureSet
from ..geom import catmull, ellipsoid, loft, ring, torus, tube, TAU
from .. import parts

BLACK = (0.15, 0.14, 0.17)
RED = (0.78, 0.14, 0.17)
PINK = (1.0, 0.50, 0.70)
PALE_PINK = (1.0, 0.78, 0.90)
HAIR = (0.11, 0.11, 0.14)
TEAL = (0.16, 0.66, 0.56)
GOLD = (0.78, 0.62, 0.32)


def build(out_dir):
    m = Model("束音ロゼ", "Tabane Roze", "束音ロゼ（非公式・二次創作モデル）", 165)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(1.0, 0.86, 0.76), eye_color=(0.12, 0.62, 0.46), bust=1.12,
                  mouth=dict(smile=-1.2))
    m.material("black", "服", BLACK, toon=0)
    m.material("red", "赤", RED, toon=0)
    m.material("red_d", "腰布", RED, double=True, toon=0)
    m.material("pink_d", "炎", PINK, double=True, toon=0)
    m.material("pale_d", "炎の先", PALE_PINK, double=True, toon=0)
    m.material("white", "白", (0.96, 0.95, 0.93), toon=0)
    m.material("sleeve", "袖", BLACK, double=True, toon=0)
    c.hair_material("hair", "髪", HAIR)
    c.hair_material("teal", "髪（内側）", TEAL)
    m.material("gold", "金具", GOLD, toon=0)
    m.material("gray", "ヘッドセット", (0.32, 0.32, 0.36), toon=0)
    m.material("jade", "髪飾り（緑）", (0.40, 0.80, 0.62), toon=0)

    c.build_body(torso_bands=[(1.0, "black"), (0.655, "skin"), (0.566, "black")],
                 leg_bands=[(1.0, "skin")], arm_bands=[(-1.0, "skin")], neck_mat="black")
    c.build_face()

    torso = c.skin_torso()
    # 胸元の開き・裾と襟の赤い縁取り
    m.add("skin", ellipsoid([0, 0.716, -0.052], [0.016, 0.010, 0.010]), m.skin_fixed("上半身2"))
    m.add("red", parts.band_around_torso(0.662, 0.652, 1.04, steps=2), torso)
    m.add("red", torus([0, 0.828, 0.003], [0, 1, 0], 0.0225, 0.003), m.skin_fixed("首"))
    m.add("red", tube([[0, 0.79, -0.044], [0, 0.745, -0.057]], 0.0025), m.skin_fixed("上半身2"))
    # ベルトとバックル
    m.add("black", parts.band_around_torso(0.578, 0.560, 1.07, steps=3), torso)
    m.add("gold", ellipsoid([0, 0.569, -0.052], [0.016, 0.011, 0.004]), m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        # 大きな袖（肩は出す）
        m.add("sleeve", parts.sleeve(s, 0.055, 0.30, [0.033, 0.042, 0.055, 0.068, 0.078, 0.082]), arm)
        m.add("red", parts.sleeve(s, 0.30, 0.335, [0.083, 0.084, 0.082], n=20), arm)
        m.add("red", parts.arm_torus(s, 0.058, 0.034, 0.004), arm)
        # 腰布：脚の外側を覆い、裾が炎になる
        leg = c.skin_leg(jp)
        a0, a1 = (-1.75, 1.75) if s > 0 else (np.pi - 1.75, np.pi + 1.75)
        radius = lambda y: 0.064 + 0.040 * (0.57 - y) / 0.57  # 裾へ向けて広がる
        for y0, y1, mat, steps in [(0.570, 0.150, "red_d", 10), (0.150, 0.138, "white", 2),
                                   (0.138, 0.070, "pink_d", 4), (0.070, 0.004, "pale_d", 4)]:
            jag = (lambda a: 0.035 * np.abs(np.sin(a * 4.0))) if mat == "pale_d" else None
            grow = lambda t, y0=y0, y1=y1: radius(y0 + (y1 - y0) * t) / parts.leg_at(y0 + (y1 - y0) * t, s)[0]
            m.add(mat, parts.leg_tube(s, y0, y1, grow, steps=steps, n=26, a0=a0, a1=a1, jag=jag), leg)
        m.add("black", parts.foot(s, r=(0.024, 0.022, 0.058)), m.skin_fixed(jp + "足首"))
        # 肩の「OJ」の代わりに赤い印
        m.add("red", ellipsoid(joints(s)["腕"] + np.array([s * 0.020, -0.03, -0.012]), [0.004, 0.009, 0.004]),
              arm)

    hair(c)
    finish(m)
    return m


def hair(c):
    m = c.m
    c.hair_cap("hair")
    # 姫カットの前髪
    for th in np.linspace(-0.95, 0.95, 15):
        side = abs(th) > 0.62
        c.bang("hair", th, -0.55 if side else 0.20, 0.019, e_start=1.25, pointy=False,
               lift_end=1.12 if side else 1.08)
    # サイドの姫毛と、その奥の緑
    for sg in (1, -1):
        for th, y_end, mat_low in [(1.08, 0.66, "hair"), (1.30, 0.60, "teal"), (1.50, 0.56, "teal")]:
            c.drape("hair", sg * th, y_end, 0.020, pointy=False, flare=0.06,
                    mats_fn=lambda p, ml=mat_low: ml if p[1] < 0.79 else "hair")
    # 後ろ髪（腰まで）
    for th in np.linspace(1.70, np.pi, 7):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            c.drape("hair", sg * th, 0.50, 0.034, flare=0.10, pointy=False)
    # 頭の両側の小さな結い髪と陰陽玉の髪飾り
    hc, hr = c.hair_surface()
    for sg in (1, -1):
        d = Head.dir(sg * 1.05, 0.80)
        base = hc + d * hr
        tip = base + np.array([sg * 0.030, 0.022, 0.012])
        m.add("hair", tube(catmull([base - d * 0.01, base + (tip - base) * 0.5, tip], 10), [0.014, 0.016, 0.008, 0.002],
                           n=10), m.skin_fixed("頭"))
        for k, dx in enumerate((0.0, 0.018)):
            ant = catmull([base, base + np.array([sg * (0.012 + dx), 0.028, 0.0]),
                           base + np.array([sg * (0.03 + dx), 0.045 - k * 0.012, 0.01])], 10)
            m.add("hair", tube(ant, np.linspace(0.0035, 0.0008, len(ant)), n=6), m.skin_fixed("頭"))
        orb = base + d * 0.012
        m.add("jade", ellipsoid(orb, [0.0145] * 3, keep=lambda q, sg=sg: q[:, 0] * sg >= -0.05, nu=16, nv=10),
              m.skin_fixed("頭"))
        m.add("white", ellipsoid(orb, [0.0143] * 3, keep=lambda q, sg=sg: q[:, 0] * sg <= 0.05, nu=16, nv=10),
              m.skin_fixed("頭"))
    # ヘッドセット（左耳）とマイク
    ear = c.head.c + np.array([0.084, -0.010, 0.002])
    m.add("gray", ellipsoid(ear, [0.007, 0.016, 0.014]), m.skin_fixed("頭"))
    mic = catmull([ear + np.array([0.002, -0.008, -0.004]), c.head.c + np.array([0.070, -0.050, -0.045]),
                   c.head.c + np.array([0.030, -0.060, -0.078])], 14)
    m.add("gray", tube(mic, 0.0018, n=6), m.skin_fixed("頭"))
    m.add("gray", ellipsoid(mic[-1], [0.005] * 3, nu=10, nv=6), m.skin_fixed("頭"))
