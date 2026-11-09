"""束音ロゼ：黒髪の姫カット＋内側の緑、黒いチャイナ風トップスと大きな袖、炎の裾の赤い腰布。"""

import numpy as np

from ..body import ARM_ANGLE, Character, Head, SIDES, finish, joints, _flat_loft
from ..builder import Model
from ..textures import TextureSet
from ..geom import MeshData, catmull, ellipsoid, loft, ring, rot_axes, torus, tube, TAU
from .. import parts, physics

BLACK = (0.15, 0.14, 0.17)
RED = (0.78, 0.14, 0.17)
PINK = (1.0, 0.50, 0.70)
PALE_PINK = (1.0, 0.78, 0.90)
HAIR = (0.19, 0.19, 0.24)  # 立ち絵の黒髪は灰色寄りでつやがある
TEAL = (0.16, 0.66, 0.56)
GOLD = (0.78, 0.62, 0.32)

# 胸元の横長キーホール開口の中心・半径（胸の上斜面〜谷間が見える位置。巨乳の頂点 y=0.702 はこの下の黒い服の中に入る）
KH_YC, KH_RX, KH_RY = 0.746, 0.036, 0.0115


def build(out_dir):
    m = Model("束音ロゼ", "Tabane Roze", "束音ロゼ（非公式・二次創作モデル）", 165)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(1.0, 0.86, 0.76), eye_color=(0.12, 0.62, 0.46), bust=1.24,
                  eye=dict(rx=0.0165, ry=0.0128, top=0.88, iris=0.70, iris_h=1.05, iris_y=-0.06, iris_in=0.05),
                  mouth=dict(rx=0.0085, ry=0.0009, smile=1.0))
    m.material("black", "服", BLACK, toon=0)
    m.material("red", "赤", RED, toon=0)
    m.material("red_trim", "赤縁", RED, toon=0, edge=False)
    m.material("red_d", "腰布", RED, double=True, toon=0)
    m.material("pink_d", "炎", PINK, double=True, toon=0)
    m.material("pale_d", "炎の先", PALE_PINK, double=True, toon=0)
    m.material("flame", "裾の炎", PINK, double=True, edge=False,
               tex=c.tex.flame("flame.png", (0.90, 0.30, 0.45), PINK, PALE_PINK))
    m.material("white", "白", (0.96, 0.95, 0.93), toon=0)
    m.material("meander", "腰布の雷文", RED, double=True, tex=c.tex.meander("meander.png", RED, (0.97, 0.95, 0.92)))
    noedge = dict(edge=False, shadow=False, shade=1.0, toon_tex=c.tex.toon("toon_flat.png", (1, 1, 1)))
    m.material("oj", "肩のOJ", RED, tex=c.tex.mark("oj.png", "OJ", (0.80, 0.12, 0.15)), **noedge)
    m.material("buckle", "バックル", (1, 1, 1), tex=c.tex.buckle("buckle.png"), **noedge)
    m.material("cleavage", "谷間", (1.0, 0.86, 0.76), tex=c.tex.nose("nose.png", (1.0, 0.86, 0.76)), **noedge)
    m.material("yinyang", "陰陽玉", (1, 1, 1), tex=c.tex.yinyang("yinyang.png", (0.40, 0.80, 0.62), (0.96, 0.95, 0.93)), toon=0)
    m.material("sleeve", "袖", BLACK, double=True, toon=0)
    c.hair_material("hair", "髪", HAIR)
    c.hair_material("teal", "髪（内側）", TEAL)
    m.material("gold", "金具", GOLD, toon=0, edge=False)
    m.material("gray", "ヘッドセット", (0.32, 0.32, 0.36), toon=0)
    m.material("jade", "髪飾り（緑）", (0.40, 0.80, 0.62), toon=0)

    # 胴体メッシュ自体で胸元の横長開口・袖ぐり・黒いチャイナ服を切り分ける（2重メッシュの突き抜けや脇の隙間がゼロになる）
    c.build_body(torso_bands=[(1.0, "black"), (0.655, "skin"), (0.566, "black")],
                 leg_bands=[(1.0, "black"), (0.538, "skin")], arm_bands=[(-1.0, "skin")],
                 leg_cut=(0.538, 0.030),
                 torso_mat_fn=roze_torso_mat, torso_snap_fn=lambda r_pts, y, angs: roze_torso_snap(c, r_pts, y, angs))
    c.build_face()

    torso = c.skin_torso()

    # 胸元の開口から見える谷間の縦影ライン（立体的な谷間の底に淡い肌影を添える）
    cl_x = np.linspace(-0.0024, 0.0024, 5)
    cl_y = np.linspace(KH_YC + KH_RY * 0.85, KH_YC - KH_RY * 0.85, 7)
    cl_rings = [c.torso_point_xy(cl_x, np.full_like(cl_x, yy), sg=-1, offset=0.0004) for yy in cl_y]
    m.add("cleavage", _flat_loft(cl_rings, np.array([0.0, 0.0, -1.0]), closed=False), m.skin_fixed("上半身2"))

    # 胸元の横長開口の赤いパイピング縁取り（胸と谷間の3D曲面に隙間なく密着させる）
    kh_a = np.linspace(0, TAU, 49)
    kh_pts = c.torso_point_xy(KH_RX * np.cos(kh_a), KH_YC + KH_RY * np.sin(kh_a), sg=-1, offset=0.0008)
    m.add("red_trim", tube(kh_pts, 0.0022, n=10), m.skin_fixed("上半身2"))

    # 袖ぐりの赤いパイピング縁取り（前〜脇の下〜後ろを1本の滑らかなU字ループでつなぐ）と、赤いアンダーバスト帯
    for s in (1, -1):
        ah_y_f = np.linspace(0.838, 0.735, 16)
        ah_y_b = np.linspace(0.735, 0.838, 16)[1:]
        pts_f = c.torso_point_xy(s * armhole_x(ah_y_f), ah_y_f, sg=-1, offset=0.0008)
        pts_b = c.torso_point_xy(s * armhole_x(ah_y_b), ah_y_b, sg=1, offset=0.0008)
        m.add("red_trim", tube(np.vstack([pts_f, pts_b]), 0.0023, n=8), torso)
    ub_angs = np.arange(64) / 64 * TAU
    ub_rings = [c.torso_ring_point(y, ub_angs, offset=0.0014) for y in (0.661, 0.656, 0.652)]
    m.add("red", loft(ub_rings, cap0=True, cap1=True, centers=[[0, 0.661, 0], [0, 0.656, 0], [0, 0.652, 0]]), torso)

    # 首の下半分を覆う立て襟と、その縁の赤
    m.add("black", tube([[0, 0.784, 0.008], [0, 0.815, 0.006], [0, 0.846, 0.004]], [0.0255, 0.0225, 0.0218], n=20),
          m.skin_auto(["上半身2", "首"]))
    m.add("red_trim", torus([0, 0.846, 0.004], [0, 1, 0], 0.0218, 0.0025), m.skin_fixed("首"))
    # 襟の金ボタン、中央の赤い縦ライン、胸の上の赤い飾り結び 2 本
    m.add("gold", ellipsoid([0, 0.828, -0.0235], [0.0035] * 3, nu=10, nv=6), m.skin_fixed("首"))
    v_top_y = np.linspace(0.786, KH_YC + KH_RY, 6)
    m.add("red_trim", tube(c.torso_point_xy(np.zeros_like(v_top_y), v_top_y, sg=-1, offset=0.0012), 0.0015, n=6), m.skin_fixed("上半身2"))
    for y in (0.775, 0.764):
        fx = np.linspace(-0.015, 0.015, 9)
        fy = y - 0.0008 * (1.0 - (fx / 0.015) ** 2)
        m.add("red_trim", tube(c.torso_point_xy(fx, fy, sg=-1, offset=0.0014), 0.0019, n=8), m.skin_fixed("上半身2"))

    # ベルト（体に沿わせて隙間をなくす）と、雲模様の彫りが入った四角い金バックル
    belt_rings = [c.torso_ring_point(y, ub_angs, offset=0.0028) for y in np.linspace(0.578, 0.560, 4)]
    m.add("black", loft(belt_rings, cap0=True, cap1=True, centers=[[0, y, 0.005] for y in np.linspace(0.578, 0.560, 4)]), torso)
    bx = np.linspace(-0.018, 0.018, 9)
    by = np.linspace(0.580, 0.558, 7)
    b_front = [c.torso_point_xy(bx, np.full_like(bx, yy), sg=-1, offset=0.0052) for yy in by]
    m.add("buckle", _flat_loft(b_front, np.array([0.0, 0.0, -1.0]), closed=False), m.skin_fixed("下半身"))
    b_rim_pts = np.vstack([
        c.torso_point_xy(bx, np.full_like(bx, 0.580), sg=-1, offset=0.0046),
        c.torso_point_xy(np.full_like(by, 0.018), by, sg=-1, offset=0.0046),
        c.torso_point_xy(bx[::-1], np.full_like(bx, 0.558), sg=-1, offset=0.0046),
        c.torso_point_xy(np.full_like(by, -0.018), by[::-1], sg=-1, offset=0.0046),
    ])
    m.add("gold", tube(b_rim_pts, 0.0014, n=6), m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        # 大きな袖（肩は出し、手首で絞って5本指の手が綺麗に出るようにする）
        sleeve, cuff = balloon_sleeve(s)
        m.add("sleeve", sleeve, arm)
        m.add("red", cuff, arm)
        m.add("red", parts.arm_torus(s, 0.085, 0.0185, 0.0035), arm)
        # 腰布：腰の横から垂れて脚の外〜後ろを覆い（前は開く）、裾が炎になる。
        # 長い布なので脚には付けず、腰から下げた物理の布にする（走ると後ろへなびき、上げた脚に押されてめくれる）
        mid = 0.50 if s > 0 else np.pi - 0.50  # 0 が外側、π/2 が後ろ
        a0, a1 = mid - 1.60, mid + 1.60
        radius = lambda y: 0.066 + 0.100 * ((0.57 - y) / 0.57) ** 1.3  # 裾へ向けて広がる

        def surface(a, y, s=s):
            r, c0 = parts.leg_at(y, s)
            return c0 + radius(y) * np.array([np.cos(a), 0.0, np.sin(a)])

        def param(p, s=s, a0=a0):
            cs = np.array([parts.leg_at(y, s)[1] for y in p[:, 1]])
            a = np.arctan2(p[:, 2] - cs[:, 2], p[:, 0] - cs[:, 0])
            return a0 + np.mod(a - a0, 2 * np.pi), p[:, 1]

        leg = physics.cloth(c, f"腰布{jp}", surface, np.linspace(a0, a1, 7), [0.570, 0.460, 0.350, 0.240, 0.130, 0.020],
                            "下半身", closed=False, lim=0.5, root_lim=1.4, rscale=0.40)(param)
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


def armhole_x(y):
    """肩出しチャイナ服の袖ぐり（y=0.735 の脇の下 rx から y=0.840 の首元へ滑らかに絞る半幅）。"""
    y_arr = np.asarray(y, float)
    rx, _, _ = parts.torso_r(y_arr)
    xc = np.interp(y_arr, [0.735, 0.760, 0.795, 0.822, 0.840], [0.080, 0.065, 0.048, 0.035, 0.028])
    return np.minimum(xc, rx)


def roze_torso_snap(c, r_pts, y, angs):
    """胴体メッシュの頂点を胸元の開口楕円と袖ぐりの境界線上にスナップし、境界のギザギザを無くす。"""
    pts = r_pts.copy()
    sa = np.sin(angs)
    if abs(y - KH_YC) < KH_RY:
        x_kh = KH_RX * np.sqrt(max(0.0, 1.0 - ((y - KH_YC) / KH_RY) ** 2))
        for s in (-1.0, 1.0):
            front_idx = np.nonzero(sa < -0.25)[0]
            best = front_idx[np.argmin(np.abs(pts[front_idx, 0] - s * x_kh))]
            if abs(pts[best, 0] - s * x_kh) < 0.0045:
                pts[best] = c.torso_point_xy(s * x_kh, y, sg=-1)
    if y > 0.742:
        x_ah = float(armhole_x(y))
        for s in (-1.0, 1.0):
            for sg, mask in ((-1, sa < -0.22), (1, sa > 0.22)):
                side_idx = np.nonzero(mask)[0]
                best = side_idx[np.argmin(np.abs(pts[side_idx, 0] - s * x_ah))]
                if abs(pts[best, 0] - s * x_ah) < 0.0045:
                    pts[best] = c.torso_point_xy(s * x_ah, y, sg=sg)
    return pts


def roze_torso_mat(c_pt, base_mat):
    """胴体メッシュの三角形ごとに、胸元のキーホール開口と肩出し部分を『肌』に切り替える。"""
    x, y, z = c_pt
    if base_mat == "black" and y > 0.655:
        if z < 0 and ((x / KH_RX) ** 2 + ((y - KH_YC) / KH_RY) ** 2) < 1.0:
            return "skin"
        if y > 0.735 and abs(x) > float(armhole_x(y)):
            return "skin"
    return base_mat


def balloon_sleeve(s, t0=0.09, t1=0.258, n=22):
    """二の腕から始まる風船袖。下へ垂れて膨らみ、手首の赤い袖口で絞って5本指の手を見せる。返り値は (袖, 袖口)。"""
    d = parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)
    d /= np.linalg.norm(d)
    up = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0.0])
    back = np.array([0, 0, 1.0])
    radii = [0.022, 0.050, 0.078, 0.093, 0.100, 0.096, 0.080, 0.056]
    ts = np.linspace(t0, t1, len(radii))
    sag = lambda r, u: max(0.0, r - 0.018) * (0.78 - 0.55 * (u ** 1.8))  # 袖口へ向かって腕の中心へ戻す
    rings, centers = [], []
    for i, (t, r) in enumerate(zip(ts, radii)):
        u = i / (len(radii) - 1)
        c = parts.arm_point(s, t) - up * sag(r, u)
        rings.append(ring(c, back, up, r * 0.92, r, n))
        centers.append(c)
    sleeve = loft(rings, centers=centers)
    cr, cc = [], []
    end = parts.arm_point(s, t1) - up * sag(radii[-1], 1.0)
    for dt, r in [(0.0, 0.056), (0.022, 0.058), (0.044, 0.055)]:
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
    rng = np.random.default_rng(7)
    # 原画のように髪は大きなかたまりで描き、細い束は並べない。
    # 姫カットの前髪：太い束 7 本を重ねてまっすぐな面にし、眉の高さで切りそろえる（目の高さから決める）
    cut = c.eye["pitch"] + 0.41
    for th in (-0.70, -0.46, -0.23, 0.0, 0.23, 0.46, 0.70):
        c.bang("hair", th, cut - 0.02 * abs(th), 0.0175, thick=0.0060, e_start=1.25, pointy=False, lift_end=1.032)
    # 姫毛：目尻より外、頬のすぐ横に左右 2 本ずつ。胸の高さで切れる
    for sg in (1, -1):
        for th, e_end, w in [(1.22, -0.62, 0.0130), (1.36, -0.50, 0.0150)]:
            c.bang("hair", sg * th, e_end, w, thick=0.0060, e_start=1.25, pointy=True, lift_end=1.030)
    # 結い髪から下がる左右の房（公式の立ち絵）：玉の下から顔の横を少し外へ流れ、肩〜胸の高さで細く分かれて終わる黒髪
    hc, hr = c.hair_surface()
    for sg in (1, -1):
        base = hc + Head.dir(sg * 1.05, 0.80) * hr
        for k, (y_end, w, mat, back, out) in enumerate([(0.800, 0.022, "hair", 0.004, 0.046), (0.770, 0.018, "hair", 0.016, 0.050)]):
            j = rng.uniform(-0.004, 0.004, 3)
            out = out + j[0]
            if mat == "teal":  # 青緑は黒い房の裏、あごの高さから出す
                pts = [base + [sg * out * 0.55, -0.075, 0.010 + back * 0.6],
                       base + [sg * out * 0.85, -0.090 + j[1], 0.012 + back],
                       base + [sg * out, -0.115, 0.014 + back]]
            else:
                pts = [base + [0, 0, back * 0.3],
                       base + [sg * 0.024, -0.022, 0.006 + back * 0.5],
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
    # 後ろ髪：外側の黒は肩甲骨まで。その内側は青緑一色で、腰まで届く（黒の下から青緑がのぞく）
    for th in np.linspace(2.0, np.pi, 4):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            c.drape("hair", sg * th, 0.72 + rng.uniform(-0.02, 0.02), 0.058,
                    flare=0.04, pointy=True, bones=3)
    # 内側の青緑は黒の裏から始め、横寄りほど低い所（あごの下）から出す。頭の上や顔の横に青緑を出さず、
    # 肩から下で体の両脇と背中に見える（公式の立ち絵では腰の高さまで左右に見える）
    for th in np.linspace(1.9, np.pi, 6):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            e0 = -0.15 - 0.35 * (np.pi - th) / (np.pi - 1.9)
            c.drape("teal", sg * th, 0.585 + rng.uniform(-0.015, 0.015), 0.050, thick=0.006, e0=e0,
                    flare=0.035, pointy=True, bones=3, scale=(1.03, 1.02, 1.04), clearance=(0.096, 0.084, 0.068))
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
        orb_mesh = ellipsoid(orb, [0.0175, 0.0175, 0.0155], nu=24, nv=16)
        orb_uv = np.column_stack([
            (orb_mesh[0][:, 0] - orb[0]) * sg / 0.035 + 0.5,
            0.5 - (orb_mesh[0][:, 1] - orb[1]) / 0.035,
        ])
        m.add("yinyang", orb_mesh, m.skin_fixed("頭"), uv=np.clip(orb_uv, 0.02, 0.98))
        # ヘッドセット（左耳）とマイク（あごの輪郭線と重なって二重あごに見えないよう、頬の外側から口元の少し横へすっきり伸ばす）
    ear = c.head.c + np.array([c.head.r[0] + 0.006, -0.010, 0.004])  # 頭の横幅に沿わせる
    m.add("gray", ellipsoid(ear, [0.007, 0.016, 0.014]), m.skin_fixed("頭"))
    mic = catmull([ear + np.array([0.002, -0.008, -0.004]), c.head.c + np.array([0.056, -0.036, -0.042]),
                   c.head.c + np.array([0.024, -0.048, -0.072])], 14)
    m.add("gray", tube(mic, 0.0018, n=6), m.skin_fixed("頭"))
    m.add("gray", ellipsoid(mic[-1], [0.0045] * 3, nu=10, nv=6), m.skin_fixed("頭"))
