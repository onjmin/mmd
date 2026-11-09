"""フェリス：橙の短髪、黄色いくちばしととさかの付いた赤い鳥の帽子、胸に「炎」の黄色いタンクトップ、赤いプリーツのミニスカート、背中の橙の鳥の羽。

参考はおんJwiki「フェリス保管庫」の約 200 枚（ref/feris_wiki/、gitignore）。多くの描き手に共通する形を取る。
open2ch の過去スレの色付き 3 枚（ref/feris_extra/）は同じ描き手の別解釈（赤髪・デニム・炎の羽）なので、帽子の形と比率だけ使う。
"""

import numpy as np

from ..body import Character, Head, SIDES, finish
from ..builder import Model
from ..textures import TextureSet
from ..geom import MeshData, catmull, ellipsoid, tube
from .. import parts, physics

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
    m.material("skirt", "スカート", SKIRT, double=True, toon=0)
    m.material("shorts", "インナー", SKIRT, toon=0)
    m.material("en", "胸の炎", EN, edge=False, shadow=False, shade=1.0,
               tex=c.tex.mark("en.png", "炎", EN, font="C:/Windows/Fonts/YuGothB.ttc"),
               toon_tex=c.tex.toon("toon_flat.png", (1, 1, 1)))
    m.material("sock", "靴下", SOCK, toon=0)
    m.material("shoe", "靴", SHOE, toon=0)
    c.hair_material("hair", "髪", HAIR)
    m.material("hat", "帽子", HAT, double=True, toon=0)
    m.material("beak", "帽子のくちばし", BEAK, toon=0)
    m.material("hat_eye", "帽子の目", (0.08, 0.06, 0.06), toon=0)
    m.material("feather", "白い羽", (0.98, 0.98, 0.96), double=True, toon=0)
    m.material("flame", "羽", FLAME, double=True, toon=1)
    m.material("flame_in", "羽の内側", FLAME_IN, double=True, toon=1)

    c.build_body(torso_bands=[(1.0, "skin"), (0.770, "top"), (0.650, "skin"), (0.585, "shorts")],
                 leg_bands=[(1.0, "shorts"), (0.560, "skin"), (0.300, "sock")], arm_bands=[(-1.0, "skin")])
    c.build_face(closed_mouth_color=(0.62, 0.30, 0.32))

    torso = c.skin_torso()
    # 肩ひも
    for jp, en, s in SIDES:
        m.add("top", tube(strap(s * 0.046, 0.772), 0.0055, 0.003, n=8), torso)
    for jp, en, s in SIDES:
        # 白いハイソックスと赤い靴
        m.add("shoe", parts.leg_tube(s, 0.060, 0.030, 1.20, steps=3), c.skin_leg(jp))
        m.add("shoe", parts.foot(s, r=(0.028, 0.028, 0.064), lift=0.002), m.skin_fixed(jp + "足首"))

    m.add("skirt", parts.skirt(0.600, 0.480, 1.90, pleats=14, depth=0.10), physics.skirt(c, 0.600, 0.480, 1.90))
    # 胸の「炎」：胸の前面に沿わせた板
    m.add("en", chest_mark(np.array(m.positions), 0.716, 0.064, 0.032), torso)

    hair(c)
    hat(c)
    wings(c)
    finish(m)
    return m


def hair(c):
    c.hair_cap("hair")
    # 原画のように髪は大きなかたまりで描き、細い束は並べない。
    # 眉の上まで下りる、毛先のとがった前髪：太い束 7 本（毛先の高さは目から決める）
    top = c.eye["pitch"] + 0.44
    for th in np.linspace(-0.90, 0.90, 7):
        c.bang("hair", th, top - 0.14 * abs(th) + 0.03 * np.sin(th * 9), 0.0300, e_start=1.18, pointy=True,
               lift_end=1.055, sway=0.26 * th)
    # 短いボブ：横は口の高さ、後ろはうなじで切れ、毛先は外へはねる。目尻より外から始める
    for th in np.linspace(1.35, np.pi, 5):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            y_end = 0.858 - 0.016 * (th - 1.35) / (np.pi - 1.35)
            c.drape("hair", sg * th, y_end, 0.056, flare=0.0, pointy=True, splay=0.35, wave=0.003)


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
    m.add("beak", ellipsoid(bc, [0.55 * hr[0], 0.30 * hr[1], 0.40 * hr[2]], nu=24, nv=12), m.skin_fixed("頭"))
    for sg in (1, -1):
        d = Head.dir(sg * 0.55, 0.42)
        m.add("hat_eye", ellipsoid(hc + d * hr * 1.0, [0.0045, 0.0055, 0.0045], nu=12, nv=8), m.skin_fixed("頭"))
        # 帽子の横の白い羽
        root = hc + Head.dir(sg * 1.30, 0.20) * hr
        for k, (dy, ln) in enumerate([(0.026, 0.034), (0.012, 0.040), (-0.002, 0.030)]):
            tip = root + np.array([sg * ln, dy + 0.010, 0.010 + 0.006 * k])
            path = catmull([root, (root + tip) / 2 + np.array([0, 0.005, 0]), tip], 10)
            w = np.linspace(0.009, 0.002, len(path))
            c.strand(path, w, 0.003, np.tile([0, 1.0, 0], (len(path), 1)), "feather", skin=m.skin_fixed("頭"))


def strap(x, y0, off=0.0035):
    """肩ひも：x の縦の断面で胴の表面に沿い、前の y0 から肩を越えて後ろの y0 まで回る。"""
    ys = np.linspace(y0, 0.840, 60)
    rx, rz, cz = parts.torso_r(ys)
    ys, rz, cz, rx = ys[rx > abs(x) + 0.004], rz[rx > abs(x) + 0.004], cz[rx > abs(x) + 0.004], rx[rx > abs(x) + 0.004]
    d = rz * np.sqrt(1 - (x / rx) ** 2) + off
    top = [[x, ys[-1] + 0.6 * d[-1], cz[-1]]]
    front = [[x, y, c - dd] for y, c, dd in zip(ys, cz, d)]
    back = [[x, y, c + dd] for y, c, dd in zip(ys, cz, d)][::-1]
    return catmull(front[::4] + top + back[::4], 6)


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
    """背中の橙の鳥の羽。肩甲骨から肩の高さへ上がり、外へ回って羽根が垂れる。"""
    m = c.m
    skin = m.skin_fixed("上半身2")
    for sg in (1, -1):
        root = np.array([sg * 0.040, 0.750, 0.062])
        # 羽の上の縁（肩から上へ上がって外へ回る弧）から、炎の羽根が外下へ垂れる
        arch = catmull([root, [sg * 0.10, 0.83, 0.09], [sg * 0.19, 0.86, 0.10], [sg * 0.27, 0.81, 0.09]], 8)
        for k in range(8):
            a = arch[min(k, len(arch) - 1)]
            u = k / 7
            ln = 0.10 + 0.20 * u                       # 外ほど長く垂れる
            tip = a + np.array([sg * (0.03 + 0.04 * u), -ln, 0.02 + 0.01 * np.sin(k)])
            mid = (a + tip) / 2 + np.array([sg * 0.05, 0.02, 0.0])
            path = catmull([root, (root + a) / 2 + np.array([0, 0.02, 0.02]), a, mid, tip], 30)
            t = np.linspace(0, 1, len(path))
            prof = np.sin(np.pi * np.clip(t * 0.92 + 0.10, 0, 1)) ** 0.6
            tang = np.gradient(path, axis=0)
            sides = np.cross(tang, [0, 0, 1.0])
            sides /= np.linalg.norm(sides, axis=1, keepdims=True) + 1e-9
            c.strand(path, 0.036 * prof + 0.002, 0.006, sides, "flame", skin=skin)
            inner = path[: int(len(path) * 0.8)] + np.array([0, 0, -0.005])
            t2 = np.linspace(0, 1, len(inner))
            c.strand(inner, 0.018 * np.sin(np.pi * np.clip(t2 * 0.92 + 0.10, 0, 1)) ** 0.6 + 0.001, 0.004,
                     sides[: len(inner)], "flame_in", skin=skin)
