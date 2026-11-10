"""蓄音キリコ：若草色のサイドポニー＋片目隠しの斜め前髪、赤褐色の瞳、エスキモー風のファー付きコート。"""

import numpy as np

from ..body import ARM_ANGLE, Character, SIDES, arm_dir, finish, joints
from ..geom import loft, smoothstep
from ..builder import Model
from ..textures import TextureSet
from ..geom import catmull, ellipsoid, ring, torus, tube, TAU
from .. import parts, physics
from ..shoes import Last

SUIT = (0.13, 0.13, 0.16)
COAT = (0.66, 0.52, 0.38)  # 公式の絵の明るい黄土色
FUR = (0.95, 0.88, 0.74)
GLOVE = (0.42, 0.20, 0.13)
HAIR = (0.50, 0.80, 0.24)
TIP = (0.30, 0.74, 0.72)
BOW = (0.38, 0.15, 0.11)


def _noise(x, scale, seed, octaves=3):
    """なめらかで繰り返さない凹凸（-1..1 程度）。x は長さ、scale はおおよその山の間隔。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for o in range(octaves):
        f = TAU / scale * (1.0 + o * 0.83) * rng.uniform(0.85, 1.15)
        out += np.sin(x * f + rng.uniform(0, TAU)) / (1 + o)
    return out / 1.8


def fur(path, r, closed=False, seed=0, n=18, rv=None, side=None):
    """ファーの縁：毛足のふくらみが不規則に揺らぐ太い管（同じ玉が並んで数珠に見えないようにする）。
    rv を渡すと断面を楕円にする（side 方向が r、直交方向が rv）。"""
    path = np.asarray(path, float)
    if closed:
        path = np.vstack([path, path[:1]])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    rng = np.random.default_rng(seed)
    if closed:  # 継ぎ目でずれないよう、一周でちょうど繰り返す凹凸にする
        k = max(3, int(round(t[-1] / (r * 3.0))))
        x = t / t[-1] * TAU
        bump = 1 + 0.07 * (np.sin(x * k + rng.uniform(0, TAU)) + 0.6 * np.sin(x * (k + 3) + rng.uniform(0, TAU))
                           + 0.4 * np.sin(x * (2 * k - 1) + rng.uniform(0, TAU))) / 2.0
        taper = 1.0
    else:
        bump = 1 + 0.07 * _noise(t, r * 3.0, seed)
        taper = 0.55 + 0.45 * np.clip(np.minimum(t, t[-1] - t) / (r * 2.0), 0, 1)  # 端は細めて丸く終える
    rr = r * bump * taper
    p1, p2 = rng.uniform(0, TAU, 2)
    return tube(path, rr, None if rv is None else rr * (rv / r), n=n, side=side, cap0=not closed, cap1=not closed,
                rmod=lambda a: 1 + 0.035 * np.sin(a * 5 + p1) + 0.02 * np.sin(a * 8 + p2))


def _smooth(y, k=5):
    """1 次元の値をならす（端は伸ばす）。"""
    pad = np.pad(y, k, mode="edge")
    w = np.exp(-np.linspace(-2, 2, 2 * k + 1) ** 2)
    return np.convolve(pad, w / w.sum(), mode="same")[k:-k]


def a_skirt(y0, y1, r1_scale, pleats, depth, seed, n=240, steps=16, bulge=0.0):
    """A ラインのひだスカート。太ももを避ける幅と円錐の幅のうち大きい方を高さ方向にならし、
    腰のすぐ下が筒になって急に折れる形にしない。ひだの深さと幅は 1 本ずつ少し変える。"""
    rng = np.random.default_rng(seed)
    rx0, rz0, cz0 = parts.torso_r(y0, 1.06)
    ys = np.linspace(y0, y1, steps)
    sx, sz, cz = [], [], []
    for y in ys:
        t = (y0 - y) / (y0 - y1)
        x0 = rx0 + (rx0 * r1_scale - rx0) * t ** 0.85 + bulge * np.sin(np.pi * t)
        z0 = rz0 + (rx0 * r1_scale * 0.85 - rz0) * t ** 0.85 + bulge * np.sin(np.pi * t)
        c = cz0 + 0.01 * t
        x1, z1 = parts.clear_legs(y, x0, z0, c)
        sx.append(x1); sz.append(z1); cz.append(c)
    # 外へ押し出した所だけ持ち上げてから少しならす（細くはしない）
    # 腰のいちばん上は元の細さのままにする（ならしたせいで上端まで太ると、腰に棚のような段ができる）
    def ease(v):
        raw = np.maximum.accumulate(np.array(v))
        sm = _smooth(raw, 3)
        return np.maximum.accumulate(np.maximum(raw[0] + (sm - sm[0]), raw * np.clip(np.arange(len(raw)) / 3.0, 0, 1) ** 0.5))
    sx, sz = ease(sx), ease(sz)
    a = np.arange(n) / n * TAU
    ph = a * pleats / 2 + 0.25 * np.sin(a * 3 + rng.uniform(0, TAU))  # ひだの間隔を少しずらす
    amp = 1 + 0.35 * np.sin(a * 2 + rng.uniform(0, TAU)) * np.sin(a * 5 + rng.uniform(0, TAU))
    rings, centers = [], []
    for k, y in enumerate(ys):
        t = k / (steps - 1)
        r = 1 + depth * t * amp * np.abs(np.sin(ph))
        ctr = np.array([0, y, cz[k]])
        rings.append(ctr + np.stack([sx[k] * r * np.cos(a), np.zeros(n), sz[k] * r * np.sin(a)], 1))
        centers.append(ctr)
    return loft(rings, centers=centers)


def puff_sleeve(s, t0, t1, seed, nt=72, na=72):
    """ふくらんだ袖。肩は腕に沿い、ひじから下でふくらんで布が下へ垂れ、袖口の手前で細かく寄せて絞る。
    しわは等間隔の輪にせず、向きと深さの違う数本を斜めに入れる。"""
    rng = np.random.default_rng(seed)
    j = joints(s)
    d = arm_dir(s)
    u = np.array([0, 0, 1.0])
    v = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0])  # 腕に直交する上向き
    tt = np.linspace(0, 1, nt)
    a = np.linspace(0, TAU, na, endpoint=False)
    prof = _smooth(np.interp(tt, [0, 0.12, 0.40, 0.66, 0.82, 0.93, 1.0],
                             [0.038, 0.044, 0.053, 0.059, 0.057, 0.045, 0.033]), 4)
    sag = _smooth(np.interp(tt, [0, 0.2, 0.7, 0.9, 1.0], [0.0, 0.002, 0.010, 0.008, 0.0]), 4)
    T, A = np.meshgrid(tt, a, indexing="ij")
    disp = np.zeros_like(T)
    # 大きなしわ：ひじの内側と袖の下にたまる斜めの谷。山は広く、谷は細い
    for _ in range(6):
        tk = rng.uniform(0.22, 0.86)
        ak = rng.choice([0.0, np.pi * 0.5, np.pi, np.pi * 1.5]) + rng.uniform(-0.6, 0.6)
        da = np.angle(np.exp(1j * (A - ak)))
        cover = np.exp(-(da / rng.uniform(0.9, 1.5)) ** 2)
        x = T - tk - rng.uniform(-0.07, 0.07) * np.sin(da)
        disp -= rng.uniform(0.05, 0.10) * np.exp(-(x / rng.uniform(0.018, 0.030)) ** 2) * cover
    # 袖口の手前と肩の付け根は細かく寄せる（縦の細いひだ）。深さは不規則に
    gather = np.clip((T - 0.80) / 0.17, 0, 1) ** 1.5 + 0.35 * np.clip((0.10 - T) / 0.10, 0, 1)
    pl = np.abs(np.sin(A * 6.5 + 0.8 * np.sin(A * 2 + rng.uniform(0, TAU))))
    disp += gather * (0.10 * pl - 0.05) * (0.7 + 0.3 * np.sin(A * 3 + rng.uniform(0, TAU)))
    disp += 0.018 * _noise(A * 0.05 + T * 0.9, 0.18, seed + 1)  # わずかなよれ
    rings, centers = [], []
    for i, t in enumerate(tt):
        c = j["腕"] + d * (t0 + (t1 - t0) * t) - v * sag[i]
        r = prof[i] * (1 + disp[i])
        rings.append(c + np.outer(np.cos(a) * r, u) + np.outer(np.sin(a) * r, v))
        centers.append(c)
    return loft(rings, centers=centers)


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
    m.material("sole", "靴底", (0.26, 0.17, 0.11), toon=0)
    m.material("stitch", "縫い目", (0.36, 0.25, 0.16), toon=0, edge=False)
    c.hair_material("hair", "髪", HAIR, flecks=[(0.98, 0.42, 0.50), (1.0, 0.90, 0.30), (0.30, 0.82, 0.86), (0.98, 0.62, 0.25)])
    c.hair_material("tip", "毛先", TIP)
    m.material("bow", "リボン", BOW, toon=0)
    m.material("bow_tip", "リボンの房", (0.86, 0.48, 0.52), toon=0)

    c.build_body(torso_bands=[(1.0, "suit")], leg_bands=[(1.0, "suit")], arm_bands=[(-1.0, "suit")],
                 neck_mat="suit", hand_mat="glove")
    c.build_face(closed_mouth_color=(0.40, 0.11, 0.12))

    torso = c.skin_torso()
    # ---- コート（公式の絵）：ゆったりしたショート丈。肩を覆い、前は開いて、裾へ広がる。しわとたるみを入れる ----
    a0, a1 = np.radians(-90 + 26), np.radians(270 - 26)
    rng = np.random.default_rng(5)

    def grow(y):  # 胴からの膨らみ：肩はぴったり、胸の下から裾へ広がる
        return 1.16 + 0.10 * float(np.clip((0.800 - y) / 0.03, 0, 1)) + 0.30 * smoothstep(0.735, 0.645, y)

    # 垂れるひだ：胸や肩甲骨で支えられた所から下へ、幅と深さを増しながら落ちる。前身ごろは浅め
    folds = []
    for _ in range(9):
        ak = rng.uniform(a0 + 0.35, a1 - 0.35)
        front = np.exp(-(np.angle(np.exp(1j * (ak + np.pi / 2))) / 0.9) ** 2)
        folds.append((ak, rng.uniform(0.73, 0.79), rng.uniform(0.025, 0.050) * (1 - 0.6 * front), rng.uniform(0.16, 0.30)))

    def fold_at(a, y):
        f = np.zeros_like(a)
        for ak, ys0, dep, w in folds:
            t = smoothstep(ys0, 0.645, y)
            da = np.angle(np.exp(1j * (a - ak)))
            f += dep * t ** 1.3 * np.exp(-(da / (w * (0.45 + 0.75 * t))) ** 2)
        return f

    def coat_ring(y, n=150):
        rx, rz, cz = parts.torso_r(y, grow(y))
        a = np.linspace(a0, a1, n)
        fold = 1 + fold_at(a, y) - 0.012 * smoothstep(0.79, 0.65, y)
        return np.stack([rx * fold * np.cos(a), np.full(n, y), cz + rz * fold * np.sin(a)], 1)

    def hem_lift(a):  # 裾はひだの山で少し下がり、谷で上がる
        return -0.10 * fold_at(a, 0.645) + 0.002 * _noise(a, 1.3, 11)
    ys = np.linspace(0.828, 0.645, 46)
    rings = []
    for y in ys:
        r = coat_ring(y)
        if y < 0.68:
            r[:, 1] += hem_lift(np.linspace(a0, a1, len(r))) * smoothstep(0.68, 0.645, y)
        rings.append(r)
    m.add("coat", loft(rings, closed=False, centers=[np.array([0, y, parts.torso_r(y)[2]]) for y in ys]), torso)
    # 首を包んであごの下まで立つファーの襟（前は少し開く）と、肩へ広がるファーのショール
    neck_c = np.array([0.0, 0.0, 0.004])
    # 立ち襟は背の高い 1 本の厚いファー（輪を積むと蛇腹に見える）。肩のショールは平たく広く寝かせる
    up = np.array([0, 1.0, 0])
    arc = ring(neck_c + [0, 0.845, 0], [1, 0, 0], [0, 0, 1], 0.034, 0.032, 90,
               np.radians(-90 + 24), np.radians(270 - 24), closed=False)
    m.add("fur", fur(arc, 0.015, rv=0.012, side=up, seed=1, n=20), m.skin_auto(["上半身2", "首"]))
    rx, rz, cz = parts.torso_r(0.818, grow(0.818) * 1.02)
    arc = ring([0, 0.822, cz], [1, 0, 0], [0, 0, 1], rx + 0.006, rz + 0.008, 120, a0 + 0.15, a1 - 0.15, closed=False)
    m.add("fur", fur(arc, 0.010, rv=0.017, side=up, seed=2, n=20), m.skin_auto(["上半身2", "首"]))
    # 裾と前の縁のファー
    hem = coat_ring(0.645, 160)
    hem[:, 1] += hem_lift(np.linspace(a0, a1, 160))
    m.add("fur", fur(hem, 0.011, seed=3), torso)
    for k in (0, -1):
        edge = np.array([coat_ring(y, 150)[k] for y in np.linspace(0.826, 0.648, 40)])
        m.add("fur", fur(edge, 0.010, seed=5 + k), torso)

    # ---- スカート：黄土色のひだが 2 段に重なって広がる。腰にファー ----
    # 下の段が水平の円盤に見えないよう、広がりを抑えて丈で重ねる
    sk = physics.skirt(c, 0.590, 0.478, 1.80)
    m.add("skirt", a_skirt(0.590, 0.522, 1.55, pleats=22, depth=0.08, seed=12), sk)
    m.add("skirt", a_skirt(0.556, 0.478, 1.80, pleats=28, depth=0.10, seed=13, bulge=0.004), sk)
    wrx, wrz, wcz = parts.torso_r(0.588, 1.06)  # 腰のファーは体に沿わせる（大きいと棚のように張り出す）
    m.add("fur", fur(ring([0, 0.588, wcz], [1, 0, 0], [0, 0, 1], wrx + 0.005, wrz + 0.005, 120), 0.009, closed=True, seed=6),
          m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        # 袖：大きくふくらみ、何段もしわが寄って、手首でファーの袖口に絞られる
        m.add("coat", puff_sleeve(s, -0.012, 0.252, seed=21 if s > 0 else 34), arm)
        ribs = np.interp(np.linspace(0, 1, 14), [0, 0.15, 0.85, 1], [0.0128, 0.0158, 0.0140, 0.0108])
        m.add("glove", parts.sleeve(s, 0.222, 0.284, ribs, n=64, rmod=lambda a: 1 + 0.045 * np.sin(a * 18)), arm)
        cuff = [parts.arm_point(s, 0.255) + v for v in
                ring([0, 0, 0], *_perp(parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)), 0.034, 0.034, 72)]
        m.add("fur", fur(cuff, 0.014, closed=True, seed=7 if s > 0 else 8), arm)
        # ブーツ：足首のすぐ上までの短い丈（公式の絵）。履き口に大きなファーの縁、丸く太いつま先
        leg = c.skin_leg(jp)
        m.add("coat", parts.leg_tube(s, 0.098, 0.030, lambda t: 1.95 + 0.15 * t, steps=16, n=56,
                                     rmod=lambda a: 1 + 0.025 * np.sin(a * 5)), leg)
        top = parts.leg_at(0.100, s)[1]
        m.add("fur", fur(ring(top, [1, 0, 0], [0, 0, 1], 0.046, 0.046, 90), 0.018, closed=True, seed=9 if s > 0 else 10), leg)
        # 足の部分：丸く太いつま先の木型。厚い靴底、つま先の U 字の縫い目（モカシン縫い）、前に下がるファーの玉
        ank = m.skin_fixed(jp + "足首")
        L = Last(s, wide=1.38, high=1.22, long=1.14, round_toe=1.7, lift=0.0085)
        m.add("coat", L.foot(n=56), ank)
        m.add("sole", L.sole(thick=0.0085, welt=0.0030, heel=0.0), ank)
        ts = np.linspace(0.46, 0.90, 26)
        ph = 1.10 * np.sqrt(np.clip(1 - ((ts - 0.46) / 0.44) ** 2, 0, 1))
        seam = np.vstack([L.line(ts, np.pi / 2 + ph, 0.0008), L.line(ts[::-1], np.pi / 2 - ph[::-1], 0.0008)])
        m.add("stitch", tube(seam, 0.0017, n=8, cap0=True, cap1=True), ank)
        # 靴底との境の縫い目
        tt = np.concatenate([np.linspace(0.02, 0.98, 40), np.linspace(0.98, 0.02, 40)])
        aa = np.concatenate([np.full(40, -0.42), np.full(40, np.pi + 0.42)])
        welt = L.line(tt, aa, 0.0006)
        m.add("stitch", tube(np.vstack([welt, welt[:1]]), 0.0013, n=8), ank)

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
    knot = np.array([0.0, 0.992, 0.040])
    for mesh in parts.bow(knot, 0.062, 0.038, tilt=0.20, facing=(0.0, 0.55, 1.0)):
        m.add("bow", mesh, m.skin_fixed("頭"))
    # 結び目から後ろへ垂れる 2 本の帯と、先のピンクの房（公式の絵）
    for sg, ln in ((1, 1.0), (-1, 0.86)):
        path = catmull([knot + [sg * 0.004, -0.004, 0.012], knot + [sg * 0.022, -0.030 * ln, 0.040],
                        knot + [sg * 0.040, -0.074 * ln, 0.056], knot + [sg * 0.050, -0.100 * ln, 0.058]], 12)
        side = np.tile([1.0, 0, 0.25 * sg], (len(path), 1))
        wd = np.interp(np.linspace(0, 1, len(path)), [0, 0.2, 1], [0.006, 0.013, 0.016])
        m.add("bow", tube(path, wd, 0.0022, n=12, side=side, cap0=True), m.skin_fixed("頭"))
        d = path[-1] - path[-2]
        d /= np.linalg.norm(d)
        m.add("bow_tip", tube([path[-1] - d * 0.001, path[-1] + d * 0.012], [0.016, 0.017], 0.0020, n=12,
                              side=side[:2], cap0=True, cap1=True, rmod=lambda a: 1 + 0.10 * np.sin(a * 2)), m.skin_fixed("頭"))
