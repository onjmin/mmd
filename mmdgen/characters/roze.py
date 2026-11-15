"""束音ロゼ：黒髪の姫カット＋内側の青緑、黒いチャイナ風トップスと大きな離れ袖、炎の裾の赤い腰布。"""

import numpy as np

from ..body import ARM_ANGLE, Character, Head, SIDES, finish, joints, _flat_loft
from ..builder import Model
from ..textures import TextureSet
from ..geom import MeshData, catmull, ellipsoid, loft, ring, rot_axes, smoothstep, torus, tube, TAU
from .. import parts, physics

BLACK = (0.15, 0.14, 0.17)
RED = (0.78, 0.14, 0.17)
PINK = (1.0, 0.52, 0.72)
PALE_PINK = (1.0, 0.80, 0.92)
HAIR = (0.19, 0.19, 0.24)
TEAL = (0.16, 0.66, 0.56)
GOLD = (0.78, 0.62, 0.32)

# 胸元の横長キーホール開口の中心・半径（引き上げた胸頂点 y=0.732 の上斜面〜谷間に自然に合わせる）
KH_YC, KH_RX, KH_RY = 0.755, 0.036, 0.0095


def build(out_dir):
    m = Model("束音ロゼ", "Tabane Roze", "束音ロゼ（非公式・二次創作モデル）", 165)
    c = Character(m, tex=TextureSet(out_dir), hair_color=HAIR, skin_color=(1.0, 0.86, 0.76), eye_color=(0.12, 0.62, 0.46), bust=1.24,
                  eye=dict(rx=0.0165, ry=0.0128, top=0.88, iris=0.70, iris_h=1.05, iris_y=-0.06, iris_in=0.05),
                  mouth=dict(rx=0.0078, ry=0.0009, smile=1.0))
    m.material("black", "服", BLACK, toon=0)
    m.material("red", "赤", RED, toon=0)
    m.material("red_trim", "赤縁", RED, toon=0, edge=False)
    m.material("red_d", "腰布", RED, double=True, toon=0)
    m.material("black_trim", "黒縁", BLACK, double=True, toon=0, edge=False)
    m.material("pink_d", "炎", PINK, double=True, toon=0, edge=False)
    m.material("pale_d", "炎の先", PALE_PINK, double=True, toon=0, edge=False)
    m.material("flame", "裾の炎", PINK, double=True, edge=False,
               tex=c.tex.flame("flame.png", RED, PINK, PALE_PINK))
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
    for k, sp in (("black", 0.20), ("sleeve", 0.20), ("red", 0.26), ("red_d", 0.22), ("gold", 0.60)):
        m._mat[k].specular, m._mat[k].power = (sp, sp, sp), 22.0
    m.material("jade", "髪飾り（緑）", (0.40, 0.80, 0.62), toon=0)

    c.build_body(torso_bands=[(1.0, "black"), (0.655, "skin"), (0.566, "black")],
                 leg_bands=[(1.0, "black"), (0.540, "skin"), (0.185, "pink_d")], arm_bands=[(-1.0, "skin")],
                 leg_cut=(0.540, 0.024),
                 torso_mat_fn=roze_torso_mat, torso_snap_fn=lambda r_pts, y, angs: roze_torso_snap(c, r_pts, y, angs))
    c.build_face(brow_color=(0.22, 0.15, 0.15), brow_dy=-0.02, brow_tilt=0.06, brow_thick=1.10)

    torso = c.skin_torso()

    # 胸元の開口から見える谷間の縦影ライン
    cl_x = np.linspace(-0.0024, 0.0024, 5)
    cl_y = np.linspace(KH_YC + KH_RY * 0.85, KH_YC - KH_RY * 0.85, 7)
    cl_rings = [c.torso_point_xy(cl_x, np.full_like(cl_x, yy), sg=-1, offset=0.0004) for yy in cl_y]
    m.add("cleavage", _flat_loft(cl_rings, np.array([0.0, 0.0, -1.0]), closed=False), m.skin_fixed("上半身2"))

    # 胸元の横長開口の赤いパイピング縁取り
    kh_a = np.linspace(0, TAU, 48, endpoint=False)
    kh_pts = c.torso_point_xy(KH_RX * np.cos(kh_a), KH_YC + KH_RY * np.sin(kh_a), sg=-1, offset=0.0008)
    m.add("red_trim", tube(kh_pts, 0.0020, n=10, closed=True), m.skin_fixed("上半身2"))

    # 袖ぐりの赤いパイピング縁取り・アンダーバスト帯・立ち絵の赤い縦ステッチライン＆脇リボン
    for s in (1, -1):
        ah_y_f = np.linspace(0.838, 0.735, 16, endpoint=False)
        ah_y_b = np.linspace(0.735, 0.838, 16, endpoint=False)
        pts_f = c.torso_point_xy(s * armhole_x(ah_y_f), ah_y_f, sg=-1, offset=0.0008)
        pts_b = c.torso_point_xy(s * armhole_x(ah_y_b), ah_y_b, sg=1, offset=0.0008)
        m.add("red_trim", tube(np.vstack([pts_f, pts_b]), 0.0021, n=8, closed=True), torso)
        # 胴体前面左右の赤い縦ライン（胸下〜裾）とウエスト脇の小さな赤いリボン
        sy = np.linspace(0.742, 0.656, 12)
        sx = s * (0.036 + 0.008 * smoothstep(0.742, 0.656, sy))
        m.add("red_trim", tube(c.torso_point_xy(sx, sy, sg=-1, offset=0.0010), 0.0014, n=6), m.skin_fixed("上半身2"))
        b_pt = c.torso_point_xy(np.array([s * 0.044]), np.array([0.661]), sg=-1, offset=0.0026)[0]
        for mesh in parts.bow(b_pt, 0.0095, 0.0050, tilt=0.15, facing=(s * 0.25, 0.0, -1.0)):
            m.add("red_trim", mesh, torso)

    ub_angs = np.arange(64) / 64 * TAU
    ub_rings = [c.torso_ring_point(y, ub_angs, offset=0.0013) for y in (0.661, 0.656, 0.652)]
    m.add("red", loft(ub_rings, cap0=True, cap1=True, centers=[[0, 0.661, 0], [0, 0.656, 0], [0, 0.652, 0]]), torso)

    # 首の立て襟（首の肌メッシュより前・袖ぐりの内側に収まる首元のリングで作り、肌貫通と肩はみ出しを防ぐ）
    col_ys = (0.818, 0.828, 0.838, 0.846)
    col_rings, col_ctrs = [], []
    for yy in col_ys:
        rx_c = float(np.interp(yy, [0.818, 0.846], [0.030, 0.0235]))
        rz_c = float(np.interp(yy, [0.818, 0.846], [0.027, 0.0255]))
        cz_c = float(np.interp(yy, [0.818, 0.846], [0.003, 0.002]))
        ctr_c = np.array([0.0, yy, cz_c])
        col_rings.append(ring(ctr_c, [1, 0, 0], [0, 0, 1], rx_c, rz_c, 48))
        col_ctrs.append(ctr_c)
    col_skin = m.skin_auto(["上半身2", "首"])
    m.add("black", loft(col_rings, cap0=True, cap1=True, centers=col_ctrs), col_skin)
    m.add("red_trim", tube(col_rings[-1], 0.0018, n=8, closed=True), m.skin_fixed("首"))
    m.add("gold", ellipsoid([0.0, 0.832, 0.0025 - 0.0262 - 0.0012], [0.0032] * 3, nu=10, nv=6), m.skin_fixed("首"))
    v_top_y = np.linspace(0.786, KH_YC + KH_RY, 6)
    m.add("red_trim", tube(c.torso_point_xy(np.zeros_like(v_top_y), v_top_y, sg=-1, offset=0.0012), 0.0014, n=6), m.skin_fixed("上半身2"))
    for y in (0.777, 0.766):
        fx = np.linspace(-0.015, 0.015, 9)
        fy = y - 0.0008 * (1.0 - (fx / 0.015) ** 2)
        m.add("red_trim", tube(c.torso_point_xy(fx, fy, sg=-1, offset=0.0014), 0.0017, n=8), m.skin_fixed("上半身2"))
        for ex in (-0.0165, 0.0165):
            ea = np.linspace(0, TAU, 16, endpoint=False)
            loop = c.torso_point_xy(ex + 0.0034 * np.cos(ea), y + 0.0034 * np.sin(ea), sg=-1, offset=0.0015)
            m.add("red_trim", tube(loop, 0.0013, n=8, closed=True), m.skin_fixed("上半身2"))
        knot = c.torso_point_xy(np.array([0.0]), np.array([y - 0.0008]), sg=-1, offset=0.0028)[0]
        m.add("red_trim", ellipsoid(knot, [0.0034, 0.0030, 0.0028], nu=16, nv=10), m.skin_fixed("上半身2"))

    # ベルト＋雲模様の金バックル＋立ち絵の赤い縦ベルトループ 4 本
    belt_rings = [c.torso_ring_point(y, ub_angs, offset=0.0026) for y in np.linspace(0.579, 0.559, 4)]
    m.add("black", loft(belt_rings, cap0=True, cap1=True, centers=[[0, y, 0.005] for y in np.linspace(0.579, 0.559, 4)]), torso)
    for lx in (-0.072, -0.040, 0.040, 0.072):
        ly = np.linspace(0.581, 0.557, 4)
        m.add("red_trim", tube(c.torso_point_xy(np.full_like(ly, lx), ly, sg=-1, offset=0.0034), 0.0022, n=6, cap0=True, cap1=True),
              m.skin_fixed("下半身"))
    bx = np.linspace(-0.018, 0.018, 9)
    by = np.linspace(0.580, 0.558, 7)
    b_front = [c.torso_point_xy(bx, np.full_like(bx, yy), sg=-1, offset=0.0052) for yy in by]
    m.add("buckle", _flat_loft(b_front, np.array([0.0, 0.0, -1.0]), closed=False), m.skin_fixed("下半身"))
    b_rim_pts = np.vstack([
        c.torso_point_xy(bx[:-1], np.full(len(bx) - 1, 0.580), sg=-1, offset=0.0046),
        c.torso_point_xy(np.full(len(by) - 1, 0.018), by[:-1], sg=-1, offset=0.0046),
        c.torso_point_xy(bx[:0:-1], np.full(len(bx) - 1, 0.558), sg=-1, offset=0.0046),
        c.torso_point_xy(np.full(len(by) - 1, -0.018), by[:0:-1], sg=-1, offset=0.0046),
    ])
    m.add("gold", tube(b_rim_pts, 0.0014, n=6, closed=True), m.skin_fixed("下半身"))

    for jp, en, s in SIDES:
        arm = c.skin_arm(jp)
        # 公式立ち絵準拠の広袖：二の腕の赤いバンド（金輪なし）→ 外面に 2 本の赤ライン＆リボン付きドレープ袖 → 折り返した空洞の赤い袖口
        sleeve, cuff, stripes, arm_band, cuff_seam, bow_ctr, bow_dir = bell_sleeve(s)
        m.add("sleeve", sleeve, arm)
        m.add("red", cuff, arm)
        m.add("red", arm_band, arm)
        m.add("black_trim", tube(cuff_seam, 0.0012, n=8, closed=True), arm)
        for st in stripes:
            m.add("red_trim", tube(st, 0.0015, n=6, cap0=True, cap1=True), arm)
        for mesh in parts.bow(bow_ctr, 0.011, 0.0055, tilt=0.10, facing=bow_dir):
            m.add("red_trim", mesh, arm)

        # 腰布：太もも前はスリットで開き、ひざ下で前が閉じて広い裾（雷文〜ピンクの鬼火へ溶ける袴状パンツ）になる
        def waist_cloth_pt(u, y, s=s):
            u_arr, y_arr = np.broadcast_arrays(np.asarray(u, float), np.asarray(y, float))
            ang_t = -0.72 + u_arr * (np.pi / 2 + 0.14 + 0.72)
            off_t = 0.0014 + 0.014 * smoothstep(0.561, 0.542, y_arr)
            pt_t = c.torso_ring_point(y_arr, ang_t, offset=off_t)
            _, c0 = parts.leg_at(np.clip(y_arr, 0.04, 0.565), 1.0)
            close_w = smoothstep(0.295, 0.205, y_arr)
            ang0 = -1.04 - (np.pi - 1.04) * close_w
            ang1 = 1.72 + (np.pi - 1.72) * close_w
            ang_l = ang0 + u_arr * (ang1 - ang0)
            t_dn = np.maximum(0.570 - y_arr, 0.0) / 0.570
            rad = 0.068 + 0.020 * np.sin(t_dn * np.pi * 0.78) + 0.016 * t_dn + 0.0025 * np.sin(u_arr * 7 * np.pi + s)
            u_sm_x = smoothstep(0.58, 1.0, u_arr)
            u_sm_over = smoothstep(0.75, 1.0, u_arr)
            u_sm_z = smoothstep(0.45, 1.0, u_arr)
            cx_open = c0[0] * (1.0 - u_sm_x) - 0.002 * u_sm_over
            cx_closed = c0[0] + 0.015
            cx = cx_open * (1.0 - close_w) + cx_closed * close_w
            cz = c0[2] + 0.008 * u_sm_z * (1.0 - close_w)
            ca, sa = np.cos(ang_l), np.sin(ang_l)
            rx_eff = rad * (1.0 - close_w * 0.56 * np.maximum(-ca, 0.0))
            rz_eff = rad * (0.94 + 0.06 * close_w)
            pt_l = np.stack([
                cx + rx_eff * ca,
                y_arr,
                cz + rz_eff * sa,
            ], -1)
            w_b = smoothstep(0.561, 0.538, y_arr)
            if w_b.ndim > 0:
                w_b = w_b[..., None]
            pt = pt_t * (1.0 - w_b) + pt_l * w_b
            pt[..., 0] *= s
            pt[..., 2] += 0.0008 * s * smoothstep(0.80, 1.0, u_arr) * (1.0 - close_w)
            return pt

        def surface(u, y, s=s):
            return waist_cloth_pt(u, y, s)

        def param(p, s=s):
            ug = np.linspace(0.0, 1.0, 33)
            yc = np.clip(p[:, 1], 0.004, 0.570)
            q = waist_cloth_pt(ug[None, :], yc[:, None], s)
            idx = np.argmin(((q - p[:, None, :]) ** 2).sum(-1), axis=1)
            return ug[idx], p[:, 1]

        # 太ももの付け根〜中上部の当たり判定を実際の太さに合わせ、足を上げたときに腰布を貫通しないようにする
        for idx_r, (yy_r, rr_r) in enumerate([(0.505, 0.048), (0.435, 0.045)]):
            physics._rigid(m, f"太もも補助{jp}{idx_r}", jp + "足", physics.BODY, physics.SPHERE,
                           (rr_r, 0, 0), parts.leg_at(yy_r, s)[1])
        leg = physics.cloth(c, f"腰布{jp}", surface, np.linspace(0.0, 1.0, 7),
                            [0.570, 0.505, 0.425, 0.335, 0.235, 0.130, 0.020],
                            "下半身", closed=False, lim=0.65, root_lim=1.5, rscale=0.52)(param)
        us = np.linspace(0.0, 1.0, 112)
        red_ys = np.concatenate([[0.570, 0.566], np.linspace(0.561, 0.205, 28)])
        # 涙滴スリットの縁に沿う細い黒パイピング
        edge_ys = np.linspace(0.558, 0.205, 32)
        m.add("black_trim", tube(waist_cloth_pt(np.full_like(edge_ys, 0.01), edge_ys, s), 0.0024, n=8, cap0=True, cap1=True), leg)

        bands = [(red_ys, "red_d"), (np.linspace(0.205, 0.155, 6), "meander"), (np.linspace(0.155, 0.000, 24), "flame")]
        for ys_band, mat in bands:
            rings = []
            for k, yy in enumerate(ys_band):
                pts = waist_cloth_pt(us, yy, s)
                if mat == "flame" and k == len(ys_band) - 1:
                    pts[:, 1] += 0.028 * np.abs(np.sin(us * np.pi * 6.0)) ** 1.4
                rings.append(pts)
            mesh = loft(rings, closed=False, centers=[np.array([0.0, yy, 0.006]) for yy in ys_band])
            m.add(mat, mesh, leg, uv=mesh.uv * [{"meander": 8.0, "flame": 3.0}.get(mat, 1.0), 1.0])

        # 足元の裾から外側へふわっと舞い上がる淡い鬼火の炎の舌（角ばった水平帯を出さず足先を自然に隠す）
        ank = m.skin_fixed(jp + "足首")
        _, leg_ctr = parts.leg_at(0.08, s)
        fl_ctr = leg_ctr + np.array([s * 0.014, 0.0, 0.0])
        for kw, (ang0, d_ang, r0, r1, y0, y1) in enumerate([
            (0.20, 0.85, 0.068, 0.095, 0.002, 0.098),
            (1.70, 0.70, 0.064, 0.088, 0.004, 0.088),
            (3.40, 0.65, 0.044, 0.060, 0.003, 0.082),
            (5.05, 0.75, 0.066, 0.092, 0.002, 0.094),
        ]):
            aa = ang0 + d_ang * s * np.linspace(0, 1, 14)
            tt_f = np.linspace(0, 1, 14)
            yy = np.linspace(y0, y1, 14) + 0.006 * np.sin(tt_f * np.pi)
            rr = np.linspace(r0, r1, 14) + 0.008 * np.sin(tt_f * np.pi) ** 1.4
            ca, sa = np.cos(aa), np.sin(aa)
            rx_t = np.where(s * ca < 0, np.minimum(rr, 0.038), rr)
            fpath = np.stack([fl_ctr[0] + rx_t * ca, yy, fl_ctr[2] + rr * sa], 1)
            fw = 0.011 * np.sin(np.linspace(0.10, 1.0, 14) * np.pi) ** 0.75 * (1.0 - 0.88 * tt_f ** 2)
            m.add("pale_d" if kw % 2 else "pink_d",
                  tube(fpath, np.maximum(fw, 0.0006), np.maximum(fw * 0.45, 0.0004), n=10, cap0=True, cap1=True), ank)

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


def bell_sleeve(s, t0=0.088, t1=0.238, n=72):
    """立ち絵準拠の大きく垂れる広袖＋折り返した開いた赤袖口＋外面 2 本の赤ライン＆リボン。
    返り値は (黒袖, 赤袖口, 赤ライン配列, 二の腕赤バンド, 袖口の黒縫い線, リボン中心, リボン向き)。"""
    d = parts.arm_point(s, 1.0) - parts.arm_point(s, 0.0)
    d /= np.linalg.norm(d)
    up = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0.0])
    back = np.array([0, 0, 1.0])
    # 二の腕のフィットした赤いバンド（金輪なし）
    band_radii = [0.0205, 0.0218, 0.0212]
    arm_band = parts.sleeve(s, t0 - 0.012, t0 + 0.005, band_radii, n=48)

    us = np.linspace(0, 1, 44)
    a = np.arange(n) / n * TAU
    # 上側は腕に沿ってすっきりと、下側（-up）へ生地がたっぷりと垂れ下がる自然なドレープ形状
    rx_u = np.interp(us, [0.0, 0.14, 0.45, 0.75, 0.92, 1.0], [0.022, 0.036, 0.052, 0.060, 0.058, 0.052])
    ry_u = np.interp(us, [0.0, 0.14, 0.45, 0.75, 0.92, 1.0], [0.022, 0.042, 0.064, 0.072, 0.068, 0.058])
    sag_u = 0.026 * np.sin(np.pi * us * 0.90) ** 1.15
    rng = np.random.default_rng(3 if s > 0 else 8)
    ph = rng.uniform(0, TAU, 3)

    rings, centers = [], []
    stripe_front, stripe_top = [], []
    bow_ctr = None
    for idx, u in enumerate(us):
        c = parts.arm_point(s, t0 + (t1 - t0) * u) - up * sag_u[idx]
        gather = np.clip((0.14 - u) / 0.14, 0, 1) * 0.035 * np.abs(np.sin(a * 8 + ph[0]))
        drape = 0.022 * np.sin(np.pi * u) ** 1.2 * (np.sin(a * 3 + ph[1]) + 0.5 * np.cos(a * 5 + ph[2]))
        k = 1.0 + gather + drape
        ring_pts = c + np.outer(np.cos(a) * rx_u[idx] * k, back) + np.outer(np.sin(a) * ry_u[idx] * k, up)
        rings.append(ring_pts)
        centers.append(c)
        # 正面〜上面の見えやすい外面に並行する 2 本の赤ライン（立ち絵準拠）
        for ang_st, dst in ((np.pi * 0.62, stripe_top), (np.pi * 0.84, stripe_front)):
            j = int(round((ang_st / TAU) * n)) % n
            dst.append(ring_pts[j] + (ring_pts[j] - c) * 0.016)
        if idx == 15:
            p_m = 0.5 * (stripe_top[-1] + stripe_front[-1])
            bow_ctr = p_m + (p_m - c) * 0.07

    bow_dir = np.array([0.22 * s, 0.50, -0.84])
    sleeve = loft(rings, cap0=True, cap1=True, centers=centers)

    # 折り返した広い赤い袖口（空洞にして中から 5 本指の手がのぞく＋真ん中に細い黒縫い線）
    end_c = parts.arm_point(s, t1) - up * sag_u[-1]
    cuff_specs = [
        (0.000, 0.052, 0.058),
        (0.003, 0.056, 0.062),
        (0.018, 0.058, 0.064),
        (0.032, 0.056, 0.062),
        (0.034, 0.050, 0.055),
        (0.020, 0.044, 0.048),
        (-0.025, 0.014, 0.015),  # 袖口の内側奥へ閉じて腕にかぶせる
    ]
    cr, cc = [], []
    for dt, rx_c, ry_c in cuff_specs:
        ci = end_c + d * dt
        cr.append(ring(ci, back, up, rx_c, ry_c, n))
        cc.append(ci)
    cuff = loft(cr, cap0=True, cap1=False, centers=cc)
    cuff_seam = cr[2]  # 真ん中の黒い縁取り線
    return sleeve, cuff, [np.array(stripe_front), np.array(stripe_top)], arm_band, cuff_seam, bow_ctr, bow_dir


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


def floating_wisp(pts, w_max, mat, skin, m):
    """頭（結い髪）の周囲に離れて浮かぶ両端尖りの平たい繊細な遊び毛（リボン状に反らせて触角感をなくす）。"""
    path = catmull(np.asarray(pts, float), 18)
    t = np.linspace(0, 1, len(path))
    env = np.sin(t * np.pi) ** 0.72
    m.add(mat, tube(path, np.maximum(w_max * env, 0.0005), np.maximum(w_max * 0.34 * env, 0.0003),
                    n=10, side=np.array([0, 0, 1.0]), cap0=True, cap1=True), skin)


def hair(c):
    m = c.m
    c.hair_cap("hair")
    rng = np.random.default_rng(7)
    # 姫カットの前髪：眉毛がはっきり見える高さでまっすぐ切りそろえ、こめかみまで隙間なく覆う
    cut = c.eye["pitch"] + 0.44
    for th in np.linspace(-0.88, 0.88, 9):
        c.bang("hair", th, cut - 0.015 * abs(th), 0.0175, thick=0.0055, e_start=1.34, pointy=False, lift_end=1.028)
    # 姫毛：頬の輪郭に沿って優しく内へ入る細い横髪
    for sg in (1, -1):
        for th, e_end, w in [(1.18, -0.58, 0.0135), (1.32, -0.46, 0.0150)]:
            c.bang("hair", sg * th, e_end, w, thick=0.0055, e_start=1.34, pointy=True, lift_end=1.024, sway=-sg * 0.06)
    # 結い髪から下がる左右の房（立ち絵準拠）：首〜鎖骨の高さでふんわり内へカールして止める
    hc, hr = c.hair_surface()
    bun_th, bun_ph = 1.14, 0.56
    for sg in (1, -1):
        base = hc + Head.dir(sg * bun_th, bun_ph) * hr
        for k, (y_end, w, back, out) in enumerate([(0.832, 0.020, 0.004, 0.032), (0.818, 0.017, 0.014, 0.038)]):
            j = rng.uniform(-0.003, 0.003, 3)
            out = out + j[0]
            pts = [
                base + [0, 0, back * 0.3],
                base + [sg * 0.020, -0.020, 0.006 + back * 0.5],
                base + [sg * (out + 0.004), -0.062 + j[1], 0.010 + back],
                [sg * (abs(base[0]) + out - 0.010), y_end + 0.025, base[2] + 0.010 + back],
                [sg * (abs(base[0]) + out - 0.022 + j[2]), y_end, base[2] + 0.004 + back],
            ]
            path = catmull(np.array(pts, float), 28)
            tg = np.gradient(path, axis=0)
            sides = np.cross(tg, [0, 0, 1.0])
            n = len(path)
            tt = np.linspace(0, 1, n)
            widths = w * (0.65 + 0.35 * np.minimum(1, tt * 4)) * (1 - 0.90 * np.clip((tt - 0.70) / 0.30, 0, 1))
            c.strand(path, widths, 0.0048, sides, "hair", skin=c.hair_chain(path, base[1] - 0.03, 3, w * 0.6))
    # 後ろ髪（外側の黒は肩〜肩甲骨までを隙間なく覆う）
    for th in np.linspace(1.90, np.pi, 5):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            c.drape("hair", sg * th, 0.735 + rng.uniform(-0.015, 0.015), 0.064,
                    flare=0.035, pointy=True, bones=3)
    # 内側の青緑は黒の裏から腰までなめらかにつながる
    for th in np.linspace(1.85, np.pi, 6):
        for sg in ((1, -1) if th < np.pi - 1e-3 else (1,)):
            e0 = -0.15 - 0.35 * (np.pi - th) / (np.pi - 1.85)
            c.drape("teal", sg * th, 0.588 + rng.uniform(-0.012, 0.012), 0.056, thick=0.006, e0=e0,
                    flare=0.032, pointy=True, bones=3, scale=(1.03, 1.02, 1.04), clearance=(0.096, 0.084, 0.068))
    # 頭の両側（こめかみ上外寄り）のお団子＆陰陽玉＋「宙に斜め外上へふわっと浮く」繊細な遊び毛
    hd = m.skin_fixed("頭")
    for sg in (1, -1):
        d = Head.dir(sg * bun_th, bun_ph)
        base = hc + d * hr
        m.add("hair", ellipsoid(base + d * 0.002, [0.013, 0.013, 0.012], nu=18, nv=12), hd)
        orb = base + d * 0.009
        orb_mesh = ellipsoid(orb, [0.0142, 0.0142, 0.0126], nu=24, nv=16)
        orb_uv = np.column_stack([
            (orb_mesh[0][:, 0] - orb[0]) * sg / 0.0284 + 0.5,
            0.5 - (orb_mesh[0][:, 1] - orb[1]) / 0.0284,
        ])
        m.add("yinyang", orb_mesh, hd, uv=np.clip(orb_uv, 0.02, 0.98))
        m.add("gold", torus(orb, [0, 0, 1], 0.0144, 0.0011, n=36, m=8), hd)
        m.add("red_trim", torus(base + d * 0.003, d, 0.0126, 0.0015, n=28, m=8), hd)
        # 原画のように結い髪の外側へ空気を挟んでふんわり舞い上がるカーブ毛束
        floating_wisp([
            orb + [sg * 0.015, 0.014, 0.001],
            orb + [sg * 0.029, 0.025, 0.003],
            orb + [sg * 0.044, 0.028, 0.005],
        ], 0.0038, "hair", hd, m)
        floating_wisp([
            orb + [sg * 0.018, 0.003, 0.003],
            orb + [sg * 0.036, 0.010, 0.005],
            orb + [sg * 0.048, 0.006, 0.007],
        ], 0.0030, "hair", hd, m)
        floating_wisp([
            orb + [sg * 0.024, -0.026, 0.006],
            orb + [sg * 0.036, -0.054, 0.008],
            orb + [sg * 0.032, -0.082, 0.005],
        ], 0.0040, "hair", hd, m)

    # ヘッドセット（左耳）とマイク
    ear = c.head.c + np.array([c.head.r[0] + 0.006, -0.010, 0.004])
    m.add("gray", ellipsoid(ear, [0.007, 0.016, 0.014], nu=32, nv=20), hd)
    m.add("black", ellipsoid(ear + [0.0045, 0, 0], [0.0045, 0.0125, 0.0108], nu=28, nv=16), hd)
    m.add("gold", torus(ear + [0.0050, 0, 0], [1, 0, 0], 0.0118, 0.0010, n=36, m=8), hd)
    m.add("jade", ellipsoid(ear + [0.0088, 0, 0], [0.0014, 0.0034, 0.0034], nu=16, nv=10), hd)
    mic = catmull([ear + np.array([0.002, -0.008, -0.004]), c.head.c + np.array([0.056, -0.036, -0.042]),
                   c.head.c + np.array([0.024, -0.048, -0.072])], 14)
    m.add("gray", tube(mic, 0.0018, n=10), hd)
    m.add("gray", ellipsoid(mic[0], [0.0038, 0.0038, 0.0038], nu=16, nv=10), hd)
    m.add("gold", torus(mic[-3], mic[-1] - mic[-4], 0.0024, 0.0008, n=20, m=6), hd)
    md = mic[-1] - mic[-3]
    md /= np.linalg.norm(md)
    side = np.cross(md, [0, 1.0, 0])
    side /= np.linalg.norm(side)
    m.add("black", ellipsoid(mic[-1] + md * 0.003, [0.0072, 0.0050, 0.0050], rot_axes(md, np.cross(side, md), side), nu=24, nv=14), hd)
