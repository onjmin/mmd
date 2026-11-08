"""束音ロゼ：黒髪の姫カット＋内側の緑、黒いチャイナ風トップスと大きな袖、炎の裾の赤い腰布。"""

import numpy as np

from ..body import ARM_ANGLE, Character, Head, SIDES, finish, joints
from ..builder import Model
from ..textures import TextureSet
from ..geom import MeshData, catmull, ellipsoid, loft, ring, torus, tube, TAU
from .. import parts

BLACK = (0.15, 0.14, 0.17)
RED = (0.78, 0.14, 0.17)
PINK = (1.0, 0.50, 0.70)
PALE_PINK = (1.0, 0.78, 0.90)
HAIR = (0.19, 0.19, 0.24)  # 立ち絵の黒髪は灰色寄りでつやがある
TEAL = (0.16, 0.66, 0.56)
GOLD = (0.78, 0.62, 0.32)


def build(out_dir):
    m = Model("束音ロゼ", "Tabane Roze", "束音ロゼ（非公式・二次創作モデル）", 165)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(1.0, 0.86, 0.76), eye_color=(0.12, 0.62, 0.46), bust=1.12,
                  eye=dict(yaw=0.46, pitch=-0.05, rx=0.0195, ry=0.0152, top=0.78, iris=0.47, iris_in=0.14), mouth=dict(rx=0.0090, ry=0.0009, smile=5.0))
    m.material("black", "服", BLACK, toon=0)
    m.material("red", "赤", RED, toon=0)
    m.material("red_d", "腰布", RED, double=True, toon=0)
    m.material("pink_d", "炎", PINK, double=True, toon=0)
    m.material("pale_d", "炎の先", PALE_PINK, double=True, toon=0)
    m.material("flame", "裾の炎", PINK, double=True, edge=False,
               tex=c.tex.flame("flame.png", (0.90, 0.30, 0.45), PINK, PALE_PINK))
    m.material("white", "白", (0.96, 0.95, 0.93), toon=0)
    m.material("meander", "腰布の雷文", RED, double=True, tex=c.tex.meander("meander.png", RED, (0.97, 0.95, 0.92)))
    m.material("oj", "肩のOJ", RED, edge=False, shadow=False, shade=1.0, tex=c.tex.mark("oj.png", "OJ", (0.80, 0.12, 0.15)),
               toon_tex=c.tex.toon("toon_flat.png", (1, 1, 1)))
    m.material("sleeve", "袖", BLACK, double=True, toon=0)
    c.hair_material("hair", "髪", HAIR)
    c.hair_material("teal", "髪（内側）", TEAL)
    m.material("gold", "金具", GOLD, toon=0)
    m.material("gray", "ヘッドセット", (0.32, 0.32, 0.36), toon=0)
    m.material("jade", "髪飾り（緑）", (0.40, 0.80, 0.62), toon=0)

    c.build_body(torso_bands=[(1.0, "black"), (0.655, "skin"), (0.566, "black")],
                 leg_bands=[(1.0, "black"), (0.538, "skin")], arm_bands=[(-1.0, "skin")],
                 leg_cut=(0.538, 0.030))
    c.build_face()

    torso = c.skin_torso()
    # 襟のすぐ下の横長の開き・裾と襟の赤い縁取り
    m.add("skin", ellipsoid([0, 0.726, -0.052], [0.036, 0.015, 0.012]), m.skin_fixed("上半身2"))
    m.add("red", parts.band_around_torso(0.662, 0.652, 1.04, steps=2), torso)
    # 首の下半分を覆う立て襟と、その縁の赤
    m.add("black", tube([[0, 0.784, 0.008], [0, 0.812, 0.006], [0, 0.836, 0.004]], [0.0255, 0.0225, 0.0222], n=20),
          m.skin_auto(["上半身2", "首"]))
    m.add("red", torus([0, 0.836, 0.004], [0, 1, 0], 0.0222, 0.0025), m.skin_fixed("首"))
    # 襟の金ボタンと、胸の上の赤い飾り結び 2 本
    m.add("gold", ellipsoid([0, 0.822, -0.0235], [0.0035] * 3, nu=10, nv=6), m.skin_fixed("首"))
    for y in (0.772, 0.756):
        z = -torso_z(y)
        m.add("red", tube([[-0.014, y, z + 0.004], [0, y - 0.001, z], [0.014, y, z + 0.004]], 0.0022, n=6),
              m.skin_fixed("上半身2"))
    # ベルトとバックル
    m.add("black", parts.band_around_torso(0.578, 0.560, 1.07, steps=3), torso)
    m.add("gold", ellipsoid([0, 0.569, -0.052], [0.016, 0.011, 0.004]), m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        # 大きな袖（肩は出す）
        sleeve, cuff = balloon_sleeve(s)
        m.add("sleeve", sleeve, arm)
        m.add("red", cuff, arm)
        m.add("red", parts.arm_torus(s, 0.085, 0.029, 0.0035), arm)
        # 腰布：腰の横から垂れて脚の外〜後ろを覆い（前は開く）、裾が炎になる
        leg = c.skin_leg(jp)
        mid = 0.50 if s > 0 else np.pi - 0.50  # 0 が外側、π/2 が後ろ
        a0, a1 = mid - 1.60, mid + 1.60
        radius = lambda y: 0.066 + 0.100 * ((0.57 - y) / 0.57) ** 1.3  # 裾へ向けて広がる
        for y0, y1, mat, steps in [(0.570, 0.200, "red_d", 9), (0.200, 0.150, "meander", 2),
                                   (0.150, 0.004, "flame", 7)]:
            jag = (lambda a: 0.040 * np.abs(np.sin(a * 5.0)) ** 1.5) if mat == "flame" else None
            grow = lambda t, y0=y0, y1=y1: radius(y0 + (y1 - y0) * t) / parts.leg_at(y0 + (y1 - y0) * t, s)[0]
            mesh = parts.leg_tube(s, y0, y1, grow, steps=steps, n=26, a0=a0, a1=a1, jag=jag)
            # 雷文は 1 マスが帯の高さと同じくらいになるよう横にくり返す
            m.add(mat, mesh, leg, uv=mesh.uv * [{"meander": 7.0, "flame": 3.0}.get(mat, 1.0), 1.0])
        # 足元は炎に包まれて見えない：足首から下を、上へ揺らめく炎の輪で囲む
        # 下が太く上へすぼまり、上ほど舌の形に透けて消える炎（靴に見えないよう縁も乱す）
        mesh = parts.leg_tube(s, 0.000, 0.120, lambda t: 2.1 * (1 - 0.35 * t), steps=5, n=40,
                              jag=lambda a: 0.06 * np.abs(np.sin(a * 4.0 + s)) ** 3)
        m.add("flame", mesh, m.skin_fixed(jp + "足首"), uv=mesh.uv * [3.0, 1.0])
    # 左の二の腕の「OJ」
    m.add("oj", shoulder_mark(1.0), c.skin_arm("左"))

    hair(c)
    finish(m)
    return m


def torso_z(y):
    rx, rz, cz = parts.torso_r(y)
    return rz - cz


def balloon_sleeve(s, t0=0.09, t1=0.33, n=22):
    """二の腕から始まる風船袖。下へ垂れて膨らみ、袖口の赤い筒で絞る。返り値は (袖, 袖口)。"""
    d = parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)
    d /= np.linalg.norm(d)
    up = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0.0])
    back = np.array([0, 0, 1.0])
    radii = [0.030, 0.056, 0.078, 0.093, 0.100, 0.098, 0.086, 0.064]
    ts = np.linspace(t0, t1, len(radii))
    sag = lambda r: max(0.0, r - 0.028) * 0.8  # 上は腕に乗り、膨らんだぶん下に垂れる
    rings, centers = [], []
    for t, r in zip(ts, radii):
        c = parts.arm_point(s, t) - up * sag(r)
        rings.append(ring(c, back, up, r * 0.92, r, n))
        centers.append(c)
    sleeve = loft(rings, centers=centers)
    cr, cc = [], []
    end = parts.arm_point(s, t1) - up * sag(radii[-1])
    for k, (dt, r) in enumerate([(0.0, 0.062), (0.030, 0.065), (0.060, 0.063)]):
        c = end + d * dt
        cr.append(ring(c, back, up, r * 0.92, r, n))
        cc.append(c)
    return sleeve, loft(cr, centers=cc)


def shoulder_mark(s, t=0.045, w=0.014, h=0.008, r=0.0275):
    """二の腕の前寄り外側に貼る、文字の向きが正面から読める板。"""
    d = parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)
    d /= np.linalg.norm(d)
    n = np.array([s * 0.6, 0.0, -1.0])
    n -= (n @ d) * d
    n /= np.linalg.norm(n)
    hx = np.array([1.0, 0, 0]) - n[0] * n
    hx /= np.linalg.norm(hx)
    vy = np.cross(hx, n)
    p = parts.arm_point(s, t) + n * r
    verts = np.array([p + hx * a + vy * b for b in (h, -h) for a in (-w, w)])
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    if np.cross(verts[1] - verts[0], verts[2] - verts[0]) @ n < 0:
        faces = faces[:, ::-1]
    return MeshData(verts, faces, uv)


def hair(c):
    m = c.m
    c.hair_cap("hair")
    rng = np.random.default_rng(7)  # 束ごとに太さと長さを少しずつ変え、筋がそろって見えないようにする
    # 姫カットの前髪：太さと毛先の高さをばらし、束どうしの間にすき間が見えるようにする
    for th in np.linspace(-0.66, 0.66, 12):
        c.bang("hair", th + rng.uniform(-0.025, 0.025), 0.15 + rng.uniform(-0.05, 0.05), 0.021 * rng.uniform(0.8, 1.2),
               thick=0.009, e_start=1.25, pointy=False, lift_end=1.12)
    # 頬の横に細く垂れる姫毛（あごの高さまで）
    for sg in (1, -1):
        for th, e_end, w in [(0.86, -0.66, 0.013), (0.96, -0.74, 0.012)]:
            c.bang("hair", sg * th, e_end + rng.uniform(-0.03, 0.03), w, thick=0.004, e_start=1.25, pointy=True,
                   lift_end=1.03)
    # 結い髪から下がる左右の房：玉の下から外へ張り出し、あごの下で外へはねる。1〜2 本は胸まで、奥の緑は腰まで
    hc, hr = c.hair_surface()
    for sg in (1, -1):
        base = hc + Head.dir(sg * 1.05, 0.80) * hr
        for k, (y_end, w, mat, back) in enumerate([(0.840, 0.017, "hair", 0.000), (0.828, 0.015, "hair", 0.006),
                                                    (0.850, 0.014, "hair", 0.012), (0.815, 0.016, "hair", 0.018),
                                                    (0.735, 0.013, "hair", 0.010), (0.760, 0.012, "hair", 0.020),
                                                    (0.600, 0.015, "teal", 0.032), (0.575, 0.016, "teal", 0.040)]):
            j = rng.uniform(-0.006, 0.006, 3)
            out = 0.066 + j[0] - (0.012 if mat == "teal" else 0.0)
            pts = [base + [0, 0, back * 0.3],
                   base + [sg * 0.030, -0.022, 0.006 + back * 0.5],
                   base + [sg * out * 0.85, -0.062 + j[1], 0.012 + back],
                   base + [sg * out, -0.108, 0.014 + back]]
            # 肩より下は肩の外側をまっすぐ下り、毛先だけ小さく波打たせる
            for y in np.linspace(base[1] - 0.15, y_end, 4 if y_end < 0.80 else 2):
                wave = 0.005 * np.sin((base[1] - y) * 60 + k)
                pts.append([sg * (abs(base[0]) + out - 0.004 + wave + j[2]), y, base[2] + 0.012 + back])
            path = catmull(np.array(pts, float), 30)
            tg = np.gradient(path, axis=0)
            sides = np.cross(tg, [0, 0, 1.0])
            n = len(path)
            tt = np.linspace(0, 1, n)
            widths = w * (0.7 + 0.3 * np.minimum(1, tt * 4)) * (1 - 0.9 * np.clip((tt - 0.75) / 0.25, 0, 1))
            c.strand(path, widths, 0.005, sides, mat, skin=c.hair_chain(path, base[1] - 0.03, 3, w * 0.6))
    # 後ろ髪（肩甲骨くらいまで）：前からは房の陰に隠れる範囲だけ
    for th in np.linspace(2.0, np.pi, 7):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            c.drape("hair", sg * th, 0.72 + rng.uniform(-0.03, 0.03), 0.034 * rng.uniform(0.8, 1.25),
                    flare=0.04, pointy=True, bones=3)
    # 頭の両側の小さな結い髪と陰陽玉の髪飾り
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
        orb = base + d * 0.015
        m.add("jade", ellipsoid(orb, [0.0180] * 3, keep=lambda q, sg=sg: q[:, 0] * sg >= -0.05, nu=16, nv=10),
              m.skin_fixed("頭"))
        m.add("white", ellipsoid(orb, [0.0178] * 3, keep=lambda q, sg=sg: q[:, 0] * sg <= 0.05, nu=16, nv=10),
              m.skin_fixed("頭"))
    # ヘッドセット（左耳）とマイク
    ear = c.head.c + np.array([0.084, -0.010, 0.002])
    m.add("gray", ellipsoid(ear, [0.007, 0.016, 0.014]), m.skin_fixed("頭"))
    mic = catmull([ear + np.array([0.002, -0.008, -0.004]), c.head.c + np.array([0.070, -0.050, -0.045]),
                   c.head.c + np.array([0.030, -0.060, -0.078])], 14)
    m.add("gray", tube(mic, 0.0018, n=6), m.skin_fixed("頭"))
    m.add("gray", ellipsoid(mic[-1], [0.005] * 3, nu=10, nv=6), m.skin_fixed("頭"))
