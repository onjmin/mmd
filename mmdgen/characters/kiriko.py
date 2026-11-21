"""蓄音キリコ：若草色のサイドポニー＋片目隠しの斜め前髪、赤褐色の瞳、エスキモー風のファー付きコート。"""

import numpy as np

from ..body import ARM_ANGLE, Character, SIDES, arm_dir, finish, joints
from ..builder import Model
from ..textures import TextureSet
from ..geom import MeshData, catmull, ellipsoid, loft, ring, smoothstep, torus, tube, TAU
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


def fur(path, r, closed=False, seed=0, n=20, rv=None, side=None):
    """ファーの縁：毛足のふくらみがやさしく揺らぐ太い管。
    rv を渡すと断面を楕円にする（side 方向が r、直交方向が rv）。"""
    path = np.asarray(path, float)
    pts = np.vstack([path, path[:1]]) if closed else path
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    rng = np.random.default_rng(seed)
    if closed:  # 継ぎ目でずれないよう、一周でちょうど繰り返す凹凸にする
        k = max(3, int(round(t[-1] / (r * 3.0))))
        x = t[:-1] / t[-1] * TAU
        bump = 1 + 0.06 * (np.sin(x * k + rng.uniform(0, TAU)) + 0.6 * np.sin(x * (k + 3) + rng.uniform(0, TAU))
                           + 0.4 * np.sin(x * (2 * k - 1) + rng.uniform(0, TAU))) / 2.0
        taper = 1.0
    else:
        bump = 1 + 0.06 * _noise(t, r * 3.0, seed)
        taper = 0.45 + 0.55 * np.clip(np.minimum(t, t[-1] - t) / (r * 2.2), 0, 1)  # 端は自然に丸く細める
    rr = r * bump * taper
    p1, p2 = rng.uniform(0, TAU, 2)
    return tube(path, rr, None if rv is None else rr * (rv / r), n=n, side=side, cap0=not closed, cap1=not closed,
                closed=closed, rmod=lambda a: 1 + 0.03 * np.sin(a * 5 + p1) + 0.02 * np.sin(a * 8 + p2))


def _smooth(y, k=5):
    """1 次元の値をならす（端は伸ばす）。"""
    pad = np.pad(y, k, mode="edge")
    w = np.exp(-np.linspace(-2, 2, 2 * k + 1) ** 2)
    return np.convolve(pad, w / w.sum(), mode="same")[k:-k]


def a_skirt(y0, y1, r1_scale, pleats, depth, seed, n=240, steps=18, r0_scale=1.05, margin=0.012):
    """一重の A ラインプリーツスカート。中腹で棚のように膨らまない滑らかな円錐シルエット。"""
    rng = np.random.default_rng(seed)
    rx0, rz0, cz0 = parts.torso_r(y0, r0_scale)
    ys = np.linspace(y0, y1, steps)
    sx, sz, cz = [], [], []
    for y in ys:
        t = (y0 - y) / (y0 - y1)
        x0 = rx0 + (rx0 * r1_scale - rx0) * (t ** 0.92)
        z0 = rz0 + (rx0 * r1_scale * 0.88 - rz0) * (t ** 0.92)
        c = cz0 + 0.008 * t
        x1, z1 = parts.clear_legs(y, x0, z0, c, margin=margin)
        sx.append(x1); sz.append(z1); cz.append(c)

    def ease(v):
        raw = np.maximum.accumulate(np.array(v))
        sm = _smooth(raw, 4)
        return np.maximum.accumulate(raw[0] + (sm - sm[0]))
    sx, sz = ease(sx), ease(sz)
    a = np.arange(n) / n * TAU
    ph = a * pleats / 2 + 0.18 * np.sin(a * 3 + rng.uniform(0, TAU))
    amp = 1 + 0.25 * np.sin(a * 2 + rng.uniform(0, TAU)) * np.sin(a * 5 + rng.uniform(0, TAU))
    rings, centers = [], []
    for k, y in enumerate(ys):
        t = k / (steps - 1)
        r = 1 + depth * (t ** 0.82) * amp * np.abs(np.sin(ph))
        ctr = np.array([0, y, cz[k]])
        pts = ctr + np.stack([sx[k] * r * np.cos(a), np.zeros(n), sz[k] * r * np.sin(a)], 1)
        pts[:, 1] -= 0.0025 * t * np.cos(2 * a)
        rings.append(pts)
        centers.append(ctr)
    return loft(rings, cap0=True, centers=centers)


def puff_sleeve(s, t0, t1, seed, nt=72, na=72):
    """滑らかにふくらむランタン袖。ひじから下へボリュームを持たせ、手首のファー袖口へ柔らかく絞る。"""
    rng = np.random.default_rng(seed)
    j = joints(s)
    d = arm_dir(s)
    u = np.array([0, 0, 1.0])
    v = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0])
    tt = np.linspace(0, 1, nt)
    a = np.linspace(0, TAU, na, endpoint=False)
    prof = _smooth(np.interp(tt, [0, 0.12, 0.42, 0.68, 0.84, 0.94, 1.0],
                             [0.036, 0.043, 0.052, 0.058, 0.055, 0.043, 0.033]), 4)
    sag = _smooth(np.interp(tt, [0, 0.25, 0.72, 0.92, 1.0], [0.0, 0.002, 0.009, 0.006, 0.0]), 4)
    T, A = np.meshgrid(tt, a, indexing="ij")
    ph0, ph1 = rng.uniform(0, TAU, 2)
    # 袖の下側・ひじ内側にゆるやかな布のたるみを入れ、ギザギザなへこみを作らない
    fold = 0.028 * np.sin(np.pi * T) ** 1.3 * (np.sin(A * 2 + ph0) + 0.5 * np.cos(A * 3 + ph1))
    # 袖口の手前を柔らかく寄せる縦ひだ
    gather = np.clip((T - 0.78) / 0.19, 0, 1) ** 1.6 + 0.25 * np.clip((0.12 - T) / 0.12, 0, 1)
    pl = np.abs(np.sin(A * 6.0 + 0.4 * np.sin(A * 2 + ph0)))
    disp = fold + gather * (0.075 * pl - 0.035)
    rings, centers = [], []
    for i, t in enumerate(tt):
        c = j["腕"] + d * (t0 + (t1 - t0) * t) - v * sag[i]
        r = prof[i] * (1 + disp[i])
        rings.append(c + np.outer(np.cos(a) * r, u) + np.outer(np.sin(a) * r, v))
        centers.append(c)
    return loft(rings, cap0=True, cap1=True, centers=centers)


def build(out_dir):
    m = Model("蓄音キリコ", "Chikune Kiriko", "蓄音キリコ（非公式・二次創作モデル）", 158)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(0.86, 0.58, 0.36), eye_color=(0.66, 0.24, 0.12), bust=0.95,
                  mouth=dict(rx=0.0092, ry=0.0033, smile=0.55, open_bottom=1.12),
                  eye=dict(rx=0.016, ry=0.0125, top=0.88, iris=0.70, iris_h=1.05, iris_y=-0.06, iris_in=0.05))
    m.material("suit", "インナー", SUIT, toon=0)
    m.material("coat", "コート", COAT, double=True, toon=1)
    m.material("fur", "ファー", FUR, toon=1)
    m.material("glove", "手袋", GLOVE, toon=0)
    m.material("skirt", "スカート", COAT, double=True, toon=1)
    m.material("sole", "靴底", (0.26, 0.17, 0.11), toon=0)
    m.material("stitch", "縫い目", (0.36, 0.25, 0.16), toon=0, edge=False)
    m.material("welt_gold", "靴の線", (0.88, 0.72, 0.26), toon=0, edge=False)
    c.hair_material("hair", "髪", HAIR, flecks=[(1.0, 0.36, 0.56), (1.0, 0.92, 0.28), (0.26, 0.84, 0.92), (1.0, 0.60, 0.22)])
    c.hair_material("tip", "毛先", TIP)
    m.material("bow", "リボン", BOW, toon=0)
    m.material("bow_tip", "リボンの房", (0.86, 0.48, 0.52), toon=0)

    c.build_body(torso_bands=[(1.0, "suit")], leg_bands=[(1.0, "suit")], arm_bands=[(-1.0, "suit")],
                 neck_mat="suit", hand_mat="glove")
    c.build_face(closed_mouth_color=(0.40, 0.11, 0.12), brow_color=(0.24, 0.18, 0.12),
                 brow_dy=0.02, brow_tilt=0.08, brow_thick=1.08)

    torso = c.skin_torso()
    # ---- エスキモー風ショートジャケット（公式立ち絵）：
    # ・襟元のVの開き（タートルネックがのぞく）から胸元で前が合わさり、裾へ向かってハの字に開く
    # ・見頃や裾・前合わせに原画にないわたは付けず、すっきりしたスエード生地にする ----
    rng = np.random.default_rng(5)

    def front_gap(y):
        # 上は V ネック (20°)、胸全体 y=0.76..0.71 は前が合わさり (2°..4°)、裾 y=0.645 へ向かいハの字に開く (24°)
        return np.radians(np.interp(y, [0.645, 0.690, 0.720, 0.760, 0.790, 0.818], [24.0, 12.0, 3.5, 2.2, 11.0, 20.0]))

    folds = []
    for _ in range(7):
        ak = rng.uniform(-np.radians(40), np.radians(220))
        front = np.exp(-(np.angle(np.exp(1j * (ak + np.pi / 2))) / 0.9) ** 2)
        folds.append((ak, rng.uniform(0.73, 0.79), rng.uniform(0.016, 0.030) * (1 - 0.65 * front), rng.uniform(0.18, 0.32)))

    def fold_at(a, y):
        f = np.zeros_like(a)
        for ak, ys0, dep, w in folds:
            t = smoothstep(ys0, 0.645, y)
            da = np.angle(np.exp(1j * (a - ak)))
            f += dep * t ** 1.3 * np.exp(-(da / (w * (0.45 + 0.75 * t))) ** 2)
        return f

    def coat_ring(y, n=150):
        g = float(front_gap(y))
        a = np.linspace(-np.pi / 2 + g, 1.5 * np.pi - g, n)
        off_out = 0.0095 + 0.0115 * smoothstep(0.80, 0.645, y)
        off_in = 0.0040 + 0.0055 * smoothstep(0.80, 0.645, y)
        outer = c.torso_ring_point(y, a, offset=off_out)
        inner = c.torso_ring_point(y, a, offset=off_in)
        cz = parts.torso_r(y)[2]
        radial = outer - np.array([0.0, y, cz])
        radial[:, 1] = 0.0
        rl = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-8
        disp = (fold_at(a, y) - 0.006 * smoothstep(0.78, 0.65, y))[:, None] * 0.055
        outer = outer + (radial / rl) * disp
        return outer, inner, a

    def hem_lift(a):
        return -0.05 * fold_at(a, 0.645) + 0.0016 * _noise(a, 1.3, 11)

    # コートの上端は肩の上の線（0.82〜0.83）を越えて首の付け根まで覆う（肩を水平に張らせたので、0.818 で止めると肩のインナーがのぞく）
    ys = np.concatenate([np.linspace(0.834, 0.820, 8), np.linspace(0.818, 0.645, 46)])
    rings, in_rings, ctrs = [], [], []
    for y in ys:
        r, ri, a = coat_ring(y)
        if y < 0.68:
            hl = hem_lift(a) * smoothstep(0.68, 0.645, y)
            r[:, 1] += hl
            ri[:, 1] += hl
        # 肩の上はほぼ水平なので、横へずらすだけでは肩の面から浮かない。上へも持ち上げる
        lift = smoothstep(0.808, 0.824, y)
        r[:, 1] += 0.0075 * lift
        ri[:, 1] += 0.0030 * lift
        cz = parts.torso_r(y)[2]
        rings.append(r)
        in_rings.append(ri)
        ctrs.append(np.array([0.0, y, cz]))
    m.add("coat", loft(rings, closed=False, centers=ctrs), torso)
    lining = loft(in_rings, closed=False, centers=ctrs)
    m.add("coat", MeshData(lining[0], lining[1][:, ::-1], lining.uv), torso)
    hem_ctr = np.array([0.0, ys[-1] + 0.05, ctrs[-1][2]])
    m.add("coat", loft([rings[-1], in_rings[-1]], closed=False, centers=[hem_ctr, hem_ctr]), torso)

    # 襟は 1 つだけ：首の後ろから両鎖骨を通り、胸元の合わせ目で V 字に合わさる一体のふわふわシープファー襟
    v_path_pts = []
    up_side = np.array([0, 1.0, 0])
    ts_col = np.linspace(-1.0, 1.0, 72)
    for tc in ts_col:
        yc = 0.838 - 0.088 * (np.abs(tc) ** 1.55)
        g = float(front_gap(yc))
        a_col = np.pi / 2 - tc * (np.pi - g * 0.88)
        pt_c = c.torso_ring_point(yc, np.array([a_col]), offset=0.0115)[0]
        v_path_pts.append(pt_c)
    col_skin = m.skin_vertical("首", "上半身2", 0.835, 0.760)
    m.add("fur", fur(v_path_pts, 0.013, rv=0.016, side=up_side, seed=1, n=22), col_skin)

    # ---- スカート（一重のプリーツスカート＋腰にファー 1 段） ----
    sk = physics.skirt(c, 0.590, 0.476, 1.66)
    m.add("skirt", a_skirt(0.588, 0.476, 1.56, pleats=20, depth=0.075, seed=12, r0_scale=1.05, margin=0.012), sk)
    wrx, wrz, wcz = parts.torso_r(0.588, 1.06)
    m.add("fur", fur(ring([0, 0.588, wcz], [1, 0, 0], [0, 0, 1], wrx + 0.005, wrz + 0.005, 120), 0.010, closed=True, seed=6),
          m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        m.add("coat", puff_sleeve(s, -0.012, 0.252, seed=21 if s > 0 else 34), arm)
        ribs = np.interp(np.linspace(0, 1, 14), [0, 0.15, 0.85, 1], [0.0128, 0.0158, 0.0140, 0.0108])
        m.add("glove", parts.sleeve(s, 0.222, 0.284, ribs, n=64, rmod=lambda a: 1 + 0.045 * np.sin(a * 18)), arm)
        cuff = [parts.arm_point(s, 0.255) + v for v in
                ring([0, 0, 0], *_perp(parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)), 0.034, 0.034, 72)]
        m.add("fur", fur(cuff, 0.014, closed=True, seed=7 if s > 0 else 8), arm)
        # ブーツ：左右のファーが中央でぶつからない太さに整え、かかとも甲へ切れ目なく接続する
        leg = c.skin_leg(jp)
        m.add("coat", parts.leg_tube(s, 0.100, 0.016, lambda t: 1.72 + 0.28 * (t ** 1.2), steps=20, n=56,
                                     rmod=lambda a: 1 + 0.014 * np.sin(a * 5), cap0=True, cap1=False), leg)
        top = parts.leg_at(0.100, s)[1]
        m.add("fur", fur(ring(top, [1, 0, 0], [0, 0, 1], 0.036, 0.036, 90), 0.012, closed=True, seed=9 if s > 0 else 10), leg)
        ank = m.skin_fixed(jp + "足首")
        L = Last(s, wide=1.36, high=1.18, long=1.14, round_toe=1.65, lift=0.0085)
        m.add("coat", L.foot(n=56), ank)
        m.add("sole", L.sole(thick=0.0085, welt=0.0030, heel=0.0), ank)
        ts = np.linspace(0.46, 0.90, 26)
        ph = 1.05 * np.sqrt(np.clip(1 - ((ts - 0.46) / 0.44) ** 2, 0, 1))
        seam = np.vstack([L.line(ts, np.pi / 2 + ph, 0.0005), L.line(ts[::-1], np.pi / 2 - ph[::-1], 0.0005)])
        m.add("stitch", tube(seam, 0.0010, n=8, cap0=True, cap1=True), ank)
        tt = np.concatenate([[0.0], np.linspace(0.02, 0.98, 40), [1.0], np.linspace(0.98, 0.02, 40)])
        aa = np.concatenate([[-0.40], np.full(40, -0.40), [np.pi + 0.40], np.full(40, np.pi + 0.40)])
        welt = L.line(tt, aa, 0.0008)
        m.add("welt_gold", tube(welt, 0.0011, n=8, closed=True), ank)

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
    # 斜めぱっつんの前髪：左前髪がおでこから浮きすぎないよう lift_end を控えめにする
    top = c.eye["pitch"] + 0.54
    for th in np.linspace(-0.95, 0.95, 7):
        cover = np.clip((0.30 - th) / 0.85, 0, 1)
        e_end = top - 0.72 * cover
        c.bang("hair", th, e_end, 0.0325, e_start=1.34, pointy=False, lift_end=1.028 + 0.024 * cover, sway=-0.12 * cover)
    # あごの長さのボブ：横は口の下、後ろはうなじで切れ、毛先は柔らかくまとまる
    for th in np.linspace(1.35, np.pi, 5):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            y_end = 0.850 - 0.014 * (th - 1.35) / (np.pi - 1.35)
            c.drape("hair", sg * th, y_end, 0.056, flare=0.0, pointy=True, splay=0.22, wave=0.0025)
    # 右の耳の後ろで結んだサイドポニー
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
    # 頭の上の後ろ寄りに大きなリボン
    knot = np.array([0.0, 0.992, 0.040])
    for mesh in parts.bow(knot, 0.062, 0.038, tilt=0.20, facing=(0.0, 0.55, 1.0)):
        m.add("bow", mesh, m.skin_fixed("頭"))
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
