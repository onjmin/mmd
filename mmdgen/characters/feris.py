"""フェリス：橙の短髪、黄色いくちばしととさかの付いた赤い鳥の帽子、胸に「炎」の黄色いタンクトップ、赤いプリーツのミニスカート、背中の橙の鳥の羽。

参考はおんJwiki「フェリス保管庫」の約 200 枚（ref/feris_wiki/、gitignore）。多くの描き手に共通する形を取る。
open2ch の過去スレの色付き 3 枚（ref/feris_extra/）は同じ描き手の別解釈（赤髪・デニム・炎の羽）なので、帽子の形と比率だけ使う。
"""

import numpy as np

from ..body import Character, Head, SIDES, finish
from ..builder import Model
from ..textures import TextureSet
from ..geom import MeshData, catmull, ellipsoid, loft, ring, tube, TAU
from .. import parts, physics
from ..shoes import Last

HAT = (0.78, 0.13, 0.12)
BEAK = (0.96, 0.84, 0.22)
HAIR = (0.96, 0.56, 0.20)
TOP = (0.96, 0.90, 0.42)
SKIRT = (0.84, 0.14, 0.14)
EN = (0.90, 0.22, 0.10)
FLAME = (0.98, 0.52, 0.12)
FLAME_IN = (1.0, 0.80, 0.32)
SHOE = (0.80, 0.14, 0.12)
SOCK = (0.97, 0.96, 0.92)


def build(out_dir):
    m = Model("フェリス", "Feris", "フェリス（非公式・二次創作モデル）", 150)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(1.0, 0.87, 0.79), eye_color=(0.82, 0.12, 0.12), bust=1.18,
                  eye=dict(rx=0.0163, ry=0.0127, top=0.88, iris=0.70, iris_h=1.05, iris_y=-0.06, iris_in=0.05),
                  mouth=dict(rx=0.0100, ry=0.0036, smile=-0.8, fang=True))
    m.material("top", "タンクトップ", TOP, toon=0)
    m.material("top_trim", "タンクトップの縁", tuple(np.array(TOP) * [0.93, 0.80, 0.55]), toon=0, edge=False)
    m.material("skirt", "スカート", SKIRT, double=True, toon=0)
    m.material("shorts", "インナー", SKIRT, toon=0)
    m.material("en", "胸の炎", EN, edge=False, shadow=False, shade=1.0,
               tex=c.tex.mark("en.png", "炎", EN, font="C:/Windows/Fonts/YuGothB.ttc"),
               toon_tex=c.tex.toon("toon_flat.png", (1, 1, 1)))
    m.material("sock", "靴下", SOCK, toon=0)
    m.material("shoe", "靴", SHOE, toon=0)
    m.material("sole", "靴底", (0.30, 0.10, 0.09), toon=0)
    m.material("gold", "金具", (0.92, 0.74, 0.30), toon=0, edge=False)
    m._mat["shoe"].specular, m._mat["shoe"].power = (0.35, 0.30, 0.30), 28.0  # 革のつや
    c.hair_material("hair", "髪", HAIR)
    m.material("hat", "帽子", HAT, double=True, toon=0)
    m.material("beak", "帽子のくちばし", BEAK, toon=0)
    m.material("hat_eye", "帽子の目", (0.08, 0.06, 0.06), toon=0)
    m.material("hat_trim", "帽子の縁", tuple(np.array(HAT) * 0.72), toon=0, edge=False)
    m.material("beak_line", "くちばしの線", tuple(np.array(BEAK) * [0.80, 0.55, 0.30]), toon=0, edge=False)
    m.material("hat_shine", "帽子の目の光", (1.0, 1.0, 1.0), toon=0, edge=False)
    m.material("feather", "白い羽", (0.98, 0.98, 0.96), double=True, toon=0)
    m.material("flame", "羽", FLAME, double=True, toon=1)
    m.material("flame_in", "羽の内側", FLAME_IN, double=True, toon=1)

    c.build_body(torso_bands=[(1.0, "skin"), (0.770, "top"), (0.650, "skin"), (0.585, "shorts")],
                 leg_bands=[(1.0, "shorts"), (0.560, "skin"), (0.300, "sock")], arm_bands=[(-1.0, "skin")])
    c.build_face(closed_mouth_color=(0.62, 0.30, 0.32))

    torso = c.skin_torso()
    # 肩ひも
    for jp, en, s in SIDES:
        m.add("top", tube(strap(c, s * 0.046, 0.772), 0.0055, 0.003, n=12), torso)
    for jp, en, s in SIDES:
        # 白いハイソックスと、赤いストラップシューズ：靴下の足を、履き口の開いた甲革が包む。
        # 靴底にはかかとの段、履き口には縁取り、甲に金の留め具のストラップ
        ank = m.skin_fixed(jp + "足首")
        L = Last(s, lift=0.0065)
        m.add("sock", L.foot(), ank)
        rim = lambda t, L=L: L.lift + float(np.interp(t, [0, 0.10, 0.26, 0.58, 0.69, 0.74], [0.041, 0.041, 0.0240, 0.0215, 0.032, 0.09]))
        upper, loop = L.upper(rim)
        m.add("shoe", upper, ank)
        m.add("shoe", tube(np.vstack([loop, loop[:1]]), 0.0014, n=8), ank)
        m.add("shoe", L.band(0.40), ank)
        outer = 0.0 if s > 0 else np.pi  # 体の外側
        bk = L.point(np.array([0.40]), np.array([outer + (0.30 if s > 0 else -0.30)]), 0.0034)[0]
        m.add("gold", ellipsoid(bk, [0.0016, 0.0034, 0.0034], nu=16, nv=10), ank)
        m.add("sole", L.sole(), ank)
        # 靴下のはき口：ひざ下の折り返しの縁
        r_top, c_top = parts.leg_at(0.296, s)
        m.add("sock", parts.leg_tube(s, 0.300, 0.288, 1.045, steps=4, n=40), c.skin_leg(jp))

    # タンクトップの上下の縁：体に沿う細い帯（リブ編みの縁取り）。肩ひもの付け根にも同じ色の縁
    angs = np.arange(120) / 120 * TAU
    for y0, y1 in ((0.7725, 0.7655), (0.6595, 0.6500)):
        ys = np.linspace(y0, y1, 4)
        rings = [c.torso_ring_point(y, angs, offset=0.0012 + 0.0006 * np.sin(np.pi * k / 3)) for k, y in enumerate(ys)]
        m.add("top_trim", loft(rings, cap0=True, cap1=True, centers=[[0, y, parts.torso_r(y)[2]] for y in ys]), torso)
    # 赤いプリーツスカート：片側へ折った車ひだ（折り山が鋭く立つ）と、腰の帯
    sk = physics.skirt(c, 0.600, 0.480, 1.90)
    m.add("skirt", pleated_skirt(0.600, 0.480, 1.90, pleats=18, depth=0.085), sk)
    # 帯の上の縁は体に付け、下へ向かってスカートの太さまで広げる（上が開いた輪にしない）
    m.add("skirt", parts.band_around_torso(0.608, 0.594, lambda y: 1.012 + 0.058 * min(1.0, (0.608 - y) / 0.004), steps=8, n=96),
          m.skin_fixed("下半身"))
    # 胸の「炎」：胸の前面に沿わせた板
    m.add("en", chest_mark(np.array(m.positions), 0.716, 0.064, 0.032), torso)

    hair(c)
    hat(c)
    wings(c)
    finish(m)
    return m


def pleated_skirt(y0, y1, r1_scale, pleats=18, depth=0.085, steps=14):
    """車ひだのプリーツスカート。parts.skirt と同じ円錐の上に、ゆるく上がって急に落ちる のこぎり形のひだを付ける
    （折り山がはっきり立つ。サイン波のひだだと波板に見える）。ひだは裾へ行くほど深い。"""
    rx0, rz0, cz0 = parts.torso_r(y0, 1.06)
    per = 20
    n = pleats * per
    a = np.arange(n) / n * TAU
    x = (np.arange(n) % per) / per
    saw = np.where(x < 0.85, x / 0.85, (1 - x) / 0.15) - 0.5
    rings, centers = [], []
    for k, y in enumerate(np.linspace(y0, y1, steps)):
        t = k / (steps - 1)
        sx = rx0 + (rx0 * r1_scale - rx0) * t
        sz = rz0 + (rx0 * r1_scale * 0.85 - rz0) * t
        ctr = np.array([0, y, cz0 + 0.01 * t])
        sx, sz = parts.clear_legs(y, sx, sz, ctr[2])
        r = 1 + depth * (0.15 + 0.85 * t) * saw
        rings.append(ctr + np.stack([sx * r * np.cos(a), np.zeros(n), sz * r * np.sin(a)], 1))
        centers.append(ctr)
    return loft(rings, centers=centers)


def hair(c):
    """おんJwiki のフェリス保管庫の絵に多い形：あご〜首までのやわらかいボブ。頭に沿って下り、毛先は少し内へ入って細かくとがる。
    前髪はぎざぎざに分かれて眉の上まで、目と目の間に少し長い束。顔の横の髪は頬からあごまで。"""
    c.hair_cap("hair")
    rng = np.random.default_rng(11)
    # 前髪：先のとがった束 7 本。真ん中の 1 本だけ目と目の間まで長い
    top = c.eye["pitch"] + 0.44
    for k, th in enumerate(np.linspace(-0.78, 0.78, 7)):
        e_end = top - 0.10 * abs(th) + 0.025 * np.sin(th * 11) - (0.10 if k == 3 else 0.0)
        c.bang("hair", th, e_end, 0.0290, e_start=1.18, pointy=True, lift_end=1.045, sway=0.18 * th)
    # 顔の横の髪：目尻の外から頬の横を下り、あごの高さでとがって終わる
    for sg in (1, -1):
        for th, e_end, w in [(1.08, c.eye["pitch"] - 0.42, 0.0200), (1.24, c.eye["pitch"] - 0.50, 0.0230)]:
            c.bang("hair", sg * th, e_end, w, thick=0.0060, e_start=1.20, pointy=True, lift_end=1.035, sway=sg * 0.05)
    # ボブ：頭のいちばん広い所からまっすぐ下ろし、毛先へ向かって少し内へ丸める（肩をよけて外へ広げない）。
    # 横はあごの高さ、後ろはうなじの少し下まで。束を重ねて切れ目のない面にする
    for th in np.linspace(1.40, np.pi, 7):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            y_end = 0.856 - 0.010 * (th - 1.40) / (np.pi - 1.40) + rng.uniform(-0.004, 0.004)
            c.drape("hair", sg * th, y_end, 0.050, thick=0.0075, flare=0.0, pointy=True, wave=0.002,
                    clearance=(0.0, 0.0, 0.0), tuck=0.14)


def hat(c):
    """頭の上にのる小さな赤い鳥の帽子。前に黄色いくちばし、その上に黒い目、横に白い羽。"""
    m = c.m
    h = c.head
    hc = h.c + np.array([0, 0.046, 0.006])
    hr = h.r * np.array([1.12, 0.72, 1.15])  # 髪の外側にかぶせる

    m.add("hat", ellipsoid(hc, hr, nu=36, nv=22, keep=lambda d: d[:, 1] > -0.12), m.skin_fixed("頭"))
    # 頭の上のとさか：後ろへ反る赤い羽 3 本
    top = hc + np.array([0, hr[1] * 0.97, -0.004])
    for k, (dx, ln) in enumerate([(0.0, 0.065), (-0.010, 0.050), (0.010, 0.050)]):
        crest = catmull([top, top + np.array([dx, ln * 0.7, 0.004]), top + np.array([dx * 1.5, ln, 0.020])], 12)
        c.strand(crest, np.linspace(0.016, 0.003, len(crest)), 0.005, np.tile([1.0, 0, 0], (len(crest), 1)), "hat",
                 skin=m.skin_fixed("頭"))
    # くちばし：額の上に張り出す平たい丸
    bc = hc + np.array([0, -0.10 * hr[1], -0.93 * hr[2]])
    br = np.array([0.55 * hr[0], 0.30 * hr[1], 0.40 * hr[2]])
    m.add("beak", ellipsoid(bc, br, nu=48, nv=24), m.skin_fixed("頭"))
    # くちばしの合わせ目（上下の境の線）と、上の小さな鼻の穴
    seam = ring(bc, [1, 0, 0], [0, 0, 1], br[0] * 1.012, br[2] * 1.012, 40, np.pi * 1.04, np.pi * 1.96, closed=False)
    m.add("beak_line", tube(seam, 0.0011, n=6, cap0=True, cap1=True), m.skin_fixed("頭"))
    for sg in (1, -1):
        nd = np.array([sg * 0.20, 0.80, -0.56])
        nd /= np.linalg.norm(nd)
        m.add("beak_line", ellipsoid(bc + nd * br * 1.0, [0.0022, 0.0010, 0.0016], nu=12, nv=8), m.skin_fixed("頭"))
    # 帽子の縁：かぶり口をぐるりと回る太い縫い目
    rim_y = -0.12 * hr[1]
    rim = ring(hc + [0, rim_y, 0], [1, 0, 0], [0, 0, 1], hr[0] * 0.995, hr[2] * 0.995, 96)
    m.add("hat_trim", tube(np.vstack([rim, rim[:1]]), 0.0030, n=10), m.skin_fixed("頭"))
    for sg in (1, -1):
        d = Head.dir(sg * 0.55, 0.42)
        m.add("hat_eye", ellipsoid(hc + d * hr * 1.0, [0.0045, 0.0055, 0.0045], nu=20, nv=12), m.skin_fixed("頭"))
        m.add("hat_shine", ellipsoid(hc + d * hr * 1.0 + d * 0.0040 + [-0.0014, 0.0020, 0], [0.0014, 0.0016, 0.0010], nu=10, nv=6),
              m.skin_fixed("頭"))
        # 帽子の横の白い羽
        root = hc + Head.dir(sg * 1.30, 0.20) * hr
        for k, (dy, ln) in enumerate([(0.026, 0.034), (0.012, 0.040), (-0.002, 0.030)]):
            tip = root + np.array([sg * ln, dy + 0.010, 0.010 + 0.006 * k])
            path = catmull([root, (root + tip) / 2 + np.array([0, 0.005, 0]), tip], 10)
            w = np.linspace(0.009, 0.002, len(path))
            c.strand(path, w, 0.003, np.tile([0, 1.0, 0], (len(path), 1)), "feather", skin=m.skin_fixed("頭"))


def strap(c, x, y0, off=0.0035):
    """肩ひも：x の縦の断面で胴の肌の面（胸のふくらみ込み）に沿い、前の y0 から肩を越えて後ろの y0 まで回る。"""
    angs = np.linspace(-np.pi, np.pi, 1441)
    front, back, y_top = [], [], None
    for y in np.linspace(y0, 0.845, 120):
        P = c.torso_ring_point(y, angs)
        if np.abs(P[:, 0]).max() < abs(x) + 0.002:  # ここより上は肩の面が x まで届かない
            break
        y_top = y
        for side, out in ((P[:, 2] < np.median(P[:, 2]), front), (P[:, 2] >= np.median(P[:, 2]), back)):
            Q = P[side & (np.sign(P[:, 0]) == np.sign(x))]
            k = np.argmin(np.abs(Q[:, 0] - x))
            q = Q[k]
            cz = parts.torso_r(y)[2]
            nrm = np.array([0.0, 0.0, q[2] - cz])
            out.append([x, y, q[2] + np.sign(nrm[2]) * off])
    front, back = np.array(front), np.array(back)
    top = [x, y_top + 0.004, (front[-1][2] + back[-1][2]) / 2]
    return catmull(np.vstack([front[::3], [top], back[::-3]]), 120)


def chest_mark(pos, y, w, h, off=0.004):
    """胸の前面に沿う文字の板。各格子点の奥行きを、その近くの体の頂点の一番前に合わせる。"""
    xs, ys = np.linspace(-w, w, 7), np.linspace(y + h, y - h, 5)
    near = pos[(np.abs(pos[:, 0]) < w + 0.02) & (np.abs(pos[:, 1] - y) < h + 0.02) & (pos[:, 2] < 0)]
    verts, uv = [], []
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            sel = near[np.hypot(near[:, 0] - xx, near[:, 1] - yy) < 0.008]
            z = sel[:, 2].min() if len(sel) else near[:, 2].min()
            verts.append([xx, yy, z - off])
            uv.append([i / (len(xs) - 1), j / (len(ys) - 1)])
    faces = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            a = j * len(xs) + i
            faces += [[a, a + 1, a + len(xs)], [a + 1, a + len(xs) + 1, a + len(xs)]]
    return MeshData(np.array(verts), np.array(faces), np.array(uv, float))


def wings(c):
    """背中の橙の鳥の羽。肩甲骨から肩の高さへ上がって外へ回る前縁（羽の腕）があり、
    そこから長い風切羽が外ほど長く垂れる。付け根は短い雨覆い羽が 2 段重なって覆う。
    羽根は薄く、先がとがり、真ん中に羽軸の筋が通る。"""
    m = c.m
    skin = m.skin_fixed("上半身2")

    def feather(root, tip, bend, width, mat, thick=0.0032, shaft=True):
        path = catmull([root, (root + tip) / 2 + bend, tip], 14)
        t = np.linspace(0, 1, len(path))
        prof = np.sin(np.pi * np.clip(t * 0.93 + 0.07, 0, 1)) ** 0.55 * (1 - 0.35 * t)  # 根元から広がり、先でとがる
        tang = np.gradient(path, axis=0)
        sides = np.cross(tang, [0, 0, 1.0])
        sides /= np.linalg.norm(sides, axis=1, keepdims=True) + 1e-9
        c.strand(path, width * prof + 0.0012, thick, sides, mat, skin=skin)
        if shaft:  # 羽軸：前と後ろの面に細い筋
            for dz in (-thick * 2.6, thick * 0.9):
                m.add("flame_in", tube(path[1:-2] + [0, 0, dz], np.linspace(0.0016, 0.0006, len(path) - 3), n=6), skin)

    for sg in (1, -1):
        root = np.array([sg * 0.036, 0.752, 0.060])
        arch = catmull([root, [sg * 0.10, 0.830, 0.088], [sg * 0.19, 0.858, 0.098], [sg * 0.275, 0.812, 0.090]], 16)
        n_a = len(arch)
        # 前縁（羽の腕）：付け根が太く、先へ細る
        m.add("flame", tube(arch, np.linspace(0.013, 0.0045, n_a), n=14, cap0=True, cap1=True), skin)
        at = lambda u: arch[int(round(u * (n_a - 1)))]
        # 風切羽：内側は短くほぼ真下へ、外ほど長く外へ流れる
        for k in range(11):
            u = 0.10 + 0.90 * k / 10
            a = at(u)
            ln = 0.105 + 0.215 * u ** 1.2
            tip = a + np.array([sg * (0.012 + 0.075 * u), -ln, 0.016 + 0.004 * np.sin(k * 1.7)])
            feather(a + [0, -0.004, 0.002 * (k % 2)], tip, np.array([sg * 0.035 * u, 0.012, 0.0]), 0.026 + 0.006 * u, "flame")
        # 雨覆い羽：前縁から短い羽根が 2 段、風切羽の根元に重なる（明るい色）
        for row, (ln0, w, dz, mat) in enumerate(((0.085, 0.020, -0.007, "flame_in"), (0.050, 0.017, -0.012, "flame"))):
            for k in range(12):
                u = 0.06 + 0.92 * (k + 0.5 * row) / 12
                a = at(min(u, 1.0))
                ln = ln0 * (0.75 + 0.6 * u)
                tip = a + np.array([sg * (0.008 + 0.040 * u), -ln, dz])
                feather(a + [0, -0.002, dz], tip, np.array([sg * 0.012 * u, 0.006, 0.0]), w, mat, thick=0.0026, shaft=row == 0)
            # 後ろから見ても根元が覆われるよう、背中側にも 1 段
        for k in range(10):
            u = 0.06 + 0.92 * k / 9
            a = at(u)
            tip = a + np.array([sg * (0.008 + 0.040 * u), -0.075 * (0.75 + 0.6 * u), 0.030])
            feather(a + [0, -0.002, 0.012], tip, np.array([sg * 0.012 * u, 0.006, 0.004]), 0.020, "flame_in", thick=0.0026)
