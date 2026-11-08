"""共通の素体：ボーン、体、顔（目・口のデカール）、髪の部品。"""

import numpy as np

from . import pmx
from .geom import (UP, MeshData, catmull, ellipsoid, loft, norm, ring, rot_axes, smoothstep, tube, TAU)

ARM_ANGLE = np.radians(37)
SIDES = (("左", "L", 1.0), ("右", "R", -1.0))

SHOULDER_Y = 0.785
NECK_Y = 0.805

TORSO = [  # y, rx, rz, cz（cz は +Z = 背中側へのずれ）
    (0.808, 0.024, 0.022, 0.002),
    (0.800, 0.046, 0.032, 0.004),
    (0.792, 0.062, 0.038, 0.004),
    (0.780, 0.077, 0.043, 0.004),
    (0.762, 0.080, 0.046, 0.002),
    (0.735, 0.074, 0.050, 0.000),
    (0.700, 0.068, 0.050, 0.000),
    (0.660, 0.059, 0.044, 0.002),
    (0.620, 0.050, 0.039, 0.004),
    (0.590, 0.053, 0.040, 0.004),
    (0.555, 0.069, 0.048, 0.006),
    (0.520, 0.075, 0.050, 0.008),
    (0.495, 0.068, 0.046, 0.008),
    (0.475, 0.042, 0.032, 0.006),
]

LEG = [  # y, 半径, x, z
    (0.565, 0.046, 0.040, 0.004),
    (0.525, 0.061, 0.049, 0.004),
    (0.450, 0.055, 0.049, 0.000),
    (0.370, 0.044, 0.047, -0.002),
    (0.300, 0.033, 0.046, -0.004),
    (0.270, 0.032, 0.046, -0.003),
    (0.220, 0.035, 0.046, 0.004),
    (0.170, 0.032, 0.045, 0.006),
    (0.120, 0.025, 0.044, 0.006),
    (0.070, 0.020, 0.044, 0.006),
    (0.040, 0.019, 0.044, 0.006),
]

ARM = [  # 腕からの距離, 半径
    (-0.025, 0.024),
    (0.000, 0.027),
    (0.070, 0.024),
    (0.150, 0.019),
    (0.200, 0.019),
    (0.280, 0.014),
]


def arm_dir(s):
    return np.array([s * np.cos(ARM_ANGLE), -np.sin(ARM_ANGLE), 0.0])


def joints(s):
    """片側の関節位置（正規化単位）。"""
    d = arm_dir(s)
    j = {}
    j["肩"] = np.array([s * 0.025, 0.786, 0.004])
    j["腕"] = np.array([s * 0.085, 0.780, 0.006])
    j["ひじ"] = j["腕"] + d * 0.150
    j["手首"] = j["ひじ"] + d * 0.130
    j["手先"] = j["手首"] + d * 0.075
    j["足"] = np.array([s * 0.048, 0.530, 0.004])
    j["ひざ"] = np.array([s * 0.046, 0.280, -0.004])
    j["足首"] = np.array([s * 0.044, 0.052, 0.006])
    j["つま先"] = np.array([s * 0.044, 0.006, -0.062])
    return j


# ---------------------------------------------------------------- 頭と顔

class Head:
    def __init__(self, c=(0.0, 0.917, -0.004), r=(0.072, 0.0805, 0.0766)):
        self.c = np.array(c, float)
        self.r = np.array(r, float)

    def shape(self, loc):
        loc = np.array(loc, float)
        y = loc[..., 1]
        t = np.clip(-y / self.r[1], 0, 1)
        loc[..., 0] *= 1 - 0.50 * t ** 1.25
        front = loc[..., 2] < 0
        loc[..., 2] = np.where(front, loc[..., 2] * (1 - 0.10 * t), loc[..., 2] * (1 - 0.25 * t))
        loc[..., 1] *= 1 + 0.06 * t
        return loc

    def point(self, d):
        return self.c + self.shape(np.asarray(d) * self.r)

    @staticmethod
    def dir(yaw, pitch):
        return np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])

    def basis(self, d0):
        e1 = norm(np.cross(d0, UP))
        e2 = np.cross(e1, d0)
        return e1, e2

    def project(self, d0, ab, off):
        """接平面上の (a, b) を頭の表面に貼り付ける。off は表面からの浮かせ量（半径比）。"""
        e1, e2 = self.basis(d0)
        ab = np.asarray(ab, float)
        q = self.point(d0) + ab[..., :1] * e1 + ab[..., 1:2] * e2
        d = norm((q - self.c) / self.r)
        p = self.point(d)
        return self.c + (p - self.c) * (1 + off)

    def mesh(self):
        return ellipsoid(self.c, self.r, nu=32, nv=20, shape=self.shape)

    def decal(self, d0, rings_ab, off, closed=True):
        rings = [self.project(d0, r, off) for r in rings_ab]
        return _flat_loft(rings, d0, closed)


def _flat_loft(rings, outward, closed=True):
    mesh = loft(rings, closed=closed)
    verts, faces = mesh
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if np.sum(fn @ outward) < 0:
        faces = faces[:, ::-1]
    return MeshData(verts, faces, mesh.uv)


def ellipse_ab(cx, cy, rx, ry, top=1.0, bottom=1.0, curve=0.0, K=3, n=28):
    t = np.arange(n) / n * TAU
    out = []
    for k in range(K + 1):
        f = max(k / K, 1e-3)
        a = cx + rx * f * np.cos(t)
        sb = np.sin(t)
        b = cy + ry * f * sb * np.where(sb > 0, top, bottom) + curve * ((a - cx) / max(rx, 1e-6)) ** 2 * ry
        out.append(np.stack([a, b], 1))
    return out


def strip_ab(pts, widths):
    pts = np.asarray(pts, float)
    tng = np.gradient(pts, axis=0)
    tng /= np.linalg.norm(tng, axis=1, keepdims=True)
    nrm = np.stack([-tng[:, 1], tng[:, 0]], 1)
    w = np.asarray(widths)[:, None]
    return [pts + nrm * w, pts - nrm * w]


def remap_ab(rings, fn):
    return [fn(r.copy()) for r in rings]


# ---------------------------------------------------------------- モデル本体

class Character:
    """素体を組み、各キャラのモジュールが服や髪を足していく。"""

    def __init__(self, model, skin_color, eye_color, bust=1.0, eye=None, mouth=None, tex=None, hair_color=(0.2, 0.2, 0.2)):
        self.m = model
        self.tex = tex
        self.hair_color = hair_color
        self.head = Head()
        self.skin_color = skin_color
        self.eye_color = np.array(eye_color, float)
        self.bust = bust
        self.eye = dict(yaw=0.35, pitch=-0.13, rx=0.0205, ry=0.0185, top=0.85, tilt=0.0, iris=0.74, iris_in=0.0)
        self.eye.update(eye or {})
        self.mouth = dict(rx=0.0075, ry=0.0016, smile=-0.8, open_bottom=1.0, pitch=-0.60, nose_pitch=-0.36)
        self.mouth.update(mouth or {})
        m = model
        m.default_toon_tex = tex.toon("toon_cloth.png", (0.80, 0.78, 0.88))
        m.material("skin", "肌", skin_color, toon_tex=tex.toon("toon_skin.png", (1.0, 0.82, 0.80)))
        self._rig()
        self._chains = 0
        physics().body_rigids(self)

    # ---- ボーン ----
    def _rig(self):
        m = self.m
        rmv = pmx.BONE_ROTATE | pmx.BONE_MOVE | pmx.BONE_VISIBLE | pmx.BONE_OPERABLE
        m.add_bone("全ての親", "mother", (0, 0, 0), None, rmv, tail=(0, 0.06, 0))
        m.add_bone("センター", "center", (0, 0.40, 0), "全ての親", rmv, tail=(0, -0.05, 0))
        m.add_bone("グルーブ", "groove", (0, 0.42, 0), "センター", rmv, tail=(0, -0.04, 0))
        m.add_bone("上半身", "upper body", (0, 0.590, 0.004), "グルーブ", tail="上半身2", seg_to=(0, 0.700, 0.002))
        m.add_bone("上半身2", "upper body2", (0, 0.700, 0.002), "上半身", tail="首", seg_to=(0, NECK_Y, 0.004))
        m.add_bone("首", "neck", (0, NECK_Y, 0.004), "上半身2", tail="頭", seg_to=(0, 0.845, 0.0))
        m.add_bone("頭", "head", (0, 0.860, 0.0), "首", tail=(0, 0.08, 0), seg_to=(0, 0.97, 0.0))
        m.add_bone("両目", "eyes", (0, 1.0, -0.06), "頭", tail=(0, 0, -0.03))
        d0 = Head.dir(self.eye["yaw"], self.eye["pitch"])
        for jp, en, s in SIDES:
            dd = d0 * np.array([s, 1, 1])
            pos = self.head.point(dd) - dd * 0.03
            m.add_bone(jp + "目", "eye_" + en, pos, "頭", tail=(0, 0, -0.03), inherit=("両目", 1.0))
        m.add_bone("下半身", "lower body", (0, 0.590, 0.004), "グルーブ", tail=(0, -0.08, 0), seg_to=(0, 0.50, 0.006))
        for jp, en, s in SIDES:
            j = joints(s)
            m.add_bone(jp + "肩", "shoulder_" + en, j["肩"], "上半身2", tail=jp + "腕", seg_to=j["腕"])
            m.add_bone(jp + "腕", "arm_" + en, j["腕"], jp + "肩", tail=jp + "ひじ", seg_to=j["ひじ"])
            m.add_bone(jp + "ひじ", "elbow_" + en, j["ひじ"], jp + "腕", tail=jp + "手首", seg_to=j["手首"])
            m.add_bone(jp + "手首", "wrist_" + en, j["手首"], jp + "ひじ", tail=j["手先"] - j["手首"], seg_to=j["手先"])
        for jp, en, s in SIDES:
            j = joints(s)
            m.add_bone(jp + "足", "leg_" + en, j["足"], "下半身", tail=jp + "ひざ", seg_to=j["ひざ"])
            m.add_bone(jp + "ひざ", "knee_" + en, j["ひざ"], jp + "足", tail=jp + "足首", seg_to=j["足首"])
            m.add_bone(jp + "足首", "ankle_" + en, j["足首"], jp + "ひざ", tail=jp + "つま先", seg_to=j["つま先"])
            m.add_bone(jp + "つま先", "toe_" + en, j["つま先"], jp + "足首", pmx.BONE_ROTATE, tail=(0, 0, -0.02))
        ikf = rmv | pmx.BONE_IK
        for jp, en, s in SIDES:
            j = joints(s)
            knee = ((-np.pi, 0.0, 0.0), (-0.0087, 0.0, 0.0))
            m.add_bone(jp + "足ＩＫ", "leg IK_" + en, j["足首"], "全ての親", ikf, tail=(0, 0, 0.08),
                       ik=(jp + "足首", 40, 2.0, [(jp + "ひざ", knee), (jp + "足", None)]))
            m.add_bone(jp + "つま先ＩＫ", "toe IK_" + en, j["つま先"], jp + "足ＩＫ", ikf, tail=(0, -0.06, 0),
                       ik=(jp + "つま先", 3, 4.0, [(jp + "足首", None)]))

    # ---- ウェイトの近道 ----
    def skin_torso(self):
        return self.m.skin_auto(["下半身", "上半身", "上半身2", "左足", "右足"],
                                {"左足": 1.7, "右足": 1.7, "下半身": 1.1})

    def skin_leg(self, jp):
        return self.m.skin_auto(["下半身", jp + "足", jp + "ひざ", jp + "足首"], {"下半身": 1.5})

    def skin_arm(self, jp):
        return self.m.skin_auto(["上半身2", jp + "肩", jp + "腕", jp + "ひじ", jp + "手首"],
                                {"上半身2": 1.4, jp + "肩": 1.3})

    def skin_hair(self):
        return self.m.skin_vertical("頭", "上半身2", 0.80, 0.70)

    def skin_skirt(self, y_top, y_bot):
        m = self.m

        def f(p):
            t = np.clip((y_top - p[:, 1]) / (y_top - y_bot), 0, 1) * 0.55
            out = []
            for pi, ti in zip(p, t):
                side = "左足" if pi[0] >= 0 else "右足"
                k = ti * min(1.0, abs(pi[0]) / 0.03)
                out.append(("下半身", side, 1.0 - k))
            return out
        return f

    # ---- 体 ----
    def build_body(self, torso_bands, leg_bands, arm_bands, neck_mat="skin", hand_mat="skin", leg_cut=None):
        """bands: [(境目, 材質)]。torso/leg は y（上から）、arm は腕の付け根からの距離。
        leg_cut=(y, 高さ) で、脚の y の境目を外側ほど持ち上げる（ショーツの脚の切れ込み）。"""
        m = self.m
        h = self.head
        m.add("skin", h.mesh(), m.skin_fixed("頭"), normals=lambda v: norm((v - h.c) / h.r))  # 顔の影を球のように滑らかに
        neck = np.array([[0, 0.785, 0.008], [0, 0.835, 0.004], [0, 0.895, 0.002]])
        m.add(neck_mat, tube(neck, [0.023, 0.0195, 0.019], n=16), m.skin_auto(["上半身2", "首", "頭"], {"頭": 1.3}))

        # 胴体
        ys = _samples([t[0] for t in TORSO], [b[0] for b in torso_bands], 34)
        rings, centers = [], []
        for y in ys:
            rx, rz, cz = (np.interp(-y, [-t[0] for t in TORSO], [t[k] for t in TORSO]) for k in (1, 2, 3))
            c = np.array([0, y, cz])
            rings.append(ring(c, [1, 0, 0], [0, 0, 1], rx, rz, 32))
            centers.append(c)
        self._banded(rings, centers, ys, torso_bands, self.skin_torso(), cap1=True, descending=True)

        # 胸
        top_mat = _band_at(torso_bands, 0.70, descending=True)
        for jp, en, s in SIDES:
            c = np.array([s * 0.032, 0.705, -0.030])
            r = np.array([0.031, 0.029, 0.026]) * self.bust
            R = rot_axes([np.cos(0.25), 0, s * np.sin(0.25)], [0, 1, 0], [-s * np.sin(0.25), 0, np.cos(0.25)])
            m.add(top_mat, ellipsoid(c, r, R, nu=20, nv=12), m.skin_fixed("上半身2"))

        # 脚
        for jp, en, s in SIDES:
            ys = _samples([t[0] for t in LEG], [b[0] for b in leg_bands], 40)
            rings, centers = [], []
            for y in ys:
                r, x, z = (np.interp(-y, [-t[0] for t in LEG], [t[k] for t in LEG]) for k in (1, 2, 3))
                c = np.array([s * x, y, z])
                pts = ring(c, [1, 0, 0], [0, 0, 1], r, r * 1.02, 22)
                if leg_cut:
                    # 境目のまわり ±0.04 だけを、外側ほど上へずらす（輪どうしは交差しない）
                    bump = max(0.0, 1 - abs(y - leg_cut[0]) / 0.04)
                    side = np.clip((pts[:, 0] - c[0]) * s / r, 0, 1) ** 0.7
                    pts[:, 1] += leg_cut[1] * bump * side
                rings.append(pts)
                centers.append(c)
            self._banded(rings, centers, ys, leg_bands, self.skin_leg(jp), cap1=True, descending=True)

        # 腕と手
        for jp, en, s in SIDES:
            j = joints(s)
            d = arm_dir(s)
            u = np.array([0, 0, 1.0])
            v = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0])
            ts = _samples([a[0] for a in ARM], [b[0] for b in arm_bands], 24, descending=False)
            rings, centers = [], []
            for t in ts:
                r = np.interp(t, [a[0] for a in ARM], [a[1] for a in ARM])
                c = j["腕"] + d * t
                rings.append(ring(c, u, v, r, r, 20))
                centers.append(c)
            self._banded(rings, centers, ts, arm_bands, self.skin_arm(jp), cap0=True, cap1=True, descending=False)
            sh_mat = _band_at(arm_bands, 0.0, descending=False)
            m.add(sh_mat, ellipsoid(j["腕"] + np.array([-s * 0.006, 0.004, 0]), [0.030, 0.028, 0.032], nu=18, nv=10),
                  m.skin_auto(["上半身2", jp + "肩", jp + "腕"], {"上半身2": 1.2}))
            self.add_hand(jp, s, hand_mat)

    def add_hand(self, jp, s, mat):
        m = self.m
        j = joints(s)
        d = arm_dir(s)
        thick = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0])
        R = rot_axes(d, thick, np.cross(d, thick))
        m.add(mat, ellipsoid(j["手首"] + d * 0.036, [0.040, 0.011, 0.020], R, nu=18, nv=10), m.skin_fixed(jp + "手首"))
        m.add(mat, ellipsoid(j["手首"] + d * 0.022 + np.array([0, 0, -0.018]), [0.020, 0.008, 0.008], R, nu=12, nv=8),
              m.skin_fixed(jp + "手首"))

    def _banded(self, rings, centers, keys, bands, skin, cap0=False, cap1=False, descending=True):
        """リング列を境目で切って材質ごとに張る（境目のリングは共有）。"""
        runs = []
        cur, start = None, 0
        for i, k in enumerate(keys):
            mat = _band_at(bands, k - (1e-6 if descending else -1e-6), descending)
            if i == 0:
                cur = mat
            elif mat != cur:
                runs.append((cur, start, i))
                cur, start = mat, i
        runs.append((cur, start, len(keys) - 1))
        for idx, (mat, a, b) in enumerate(runs):
            if b <= a:
                continue
            mesh = loft(rings[a:b + 1], cap0=cap0 and idx == 0, cap1=cap1 and idx == len(runs) - 1,
                        centers=centers[a:b + 1])
            self.m.add(mat, mesh, skin)

    # ---- 顔 ----
    def build_face(self, closed_mouth_color=(0.45, 0.16, 0.17), blush=True, brow_color=None):
        """目は白目・瞳・まつ毛の 3 枚の板に、描いたテクスチャを貼る。"""
        m = self.m
        h = self.head
        E = self.eye
        T = self.tex
        # 目や頬は陰影で灰色にならないよう、真っ白なトゥーンと環境色 1 で描いた色そのままにする
        noedge = dict(edge=False, shadow=False, shade=1.0, toon_tex=T.toon("toon_flat.png", (1, 1, 1)))
        m.material("eye_white", "白目", (1, 1, 1), tex=T.eye_white(), **noedge)
        m.material("iris", "瞳", self.eye_color, tex=T.iris("iris.png", self.eye_color), **noedge)
        m.material("lash", "まつ毛", (0.1, 0.06, 0.07), tex=T.lash(), **noedge)
        bc = brow_color if brow_color is not None else np.asarray(self.hair_color) * 0.6
        m.material("brow", "眉", bc, tex=T.brow("brow.png", bc), **noedge)
        m.material("mouth", "口", closed_mouth_color, **{**noedge, "shade": 0.4})
        m.material("nose", "鼻", np.asarray(self.skin_color) * np.array([0.90, 0.78, 0.76]), **{**noedge, "shade": 0.4})
        if blush:
            m.material("blush", "頬", (1.0, 0.5, 0.5), tex=T.blush(), **noedge)
        for name, en, panel in [("まばたき", "blink", 2), ("笑い", "smile", 2), ("ウィンク", "wink", 2),
                                ("ウィンク右", "wink R", 2), ("あ", "a", 3), ("い", "i", 3), ("う", "u", 3),
                                ("え", "e", 3), ("お", "o", 3)]:
            m.define_morph(name, en, panel)

        rx, ry = E["rx"], E["ry"]
        lid = -0.30 * ry

        def closer(arch):
            """まぶたを閉じた形。arch > 0 で ^（笑い）、< 0 で ‿（まばたき）。"""
            def f(r):
                a = r[:, 0]
                w = np.abs(a).max()
                r[:, 1] = lid + (r[:, 1] - lid) * 0.04 + arch * ry * (1 - (a / w) ** 2)
                return r
            return f

        def plane(a_half, b_lo, b_hi, s, cx=0.0, cy=0.0, cols=9, rows=7):
            """接平面上の長方形。u は目尻方向が 1、v は上が 0。"""
            us = np.linspace(-1, 1, cols)
            out = []
            for b in np.linspace(b_hi, b_lo, rows):
                a = cx + us * a_half * s
                out.append(np.stack([a, np.full(cols, cy + b)], 1))
            return out

        for jp, en, s in SIDES:
            d0 = Head.dir(s * E["yaw"], E["pitch"])
            tilt = E["tilt"]

            def tilted(rows):
                return [np.stack([r[:, 0], r[:, 1] + tilt * r[:, 0] * s], 1) for r in rows]

            layers = [
                ("eye_white", plane(rx * 1.00, -0.99 * ry, 0.90 * ry * E["top"], s), 0.010, "頭"),
                ("iris", plane(rx * E["iris"], -0.90 * ry, 1.00 * ry * E["top"], s, cx=-s * E["iris_in"] * rx), 0.016, jp + "目"),
                ("lash", plane(rx * 1.08, -0.823 * ry, 0.995 * ry * E["top"], s), 0.022, "頭"),
            ]
            for mat, rows, off, bone in layers:
                rows = tilted(rows)
                base = m.add(mat, h.decal(d0, rows, off, closed=False), m.skin_fixed(bone))
                m.add_morph_part("まばたき", base, h.decal(d0, remap_ab(rows, closer(-0.22)), off, closed=False)[0])
                m.add_morph_part("笑い", base, h.decal(d0, remap_ab(rows, closer(0.30)), off, closed=False)[0])
                m.add_morph_part("ウィンク" if s > 0 else "ウィンク右", base,
                                 h.decal(d0, remap_ab(rows, closer(-0.22)), off, closed=False)[0])
            bd = Head.dir(s * (E["yaw"] + 0.02), E["pitch"] + 0.40)
            m.add("brow", h.decal(bd, plane(0.026, -0.0065, 0.0065, s), 0.020, closed=False), m.skin_fixed("頭"))
            if blush:
                cd = Head.dir(s * 0.56, -0.36)
                m.add("blush", h.decal(cd, plane(0.020, -0.010, 0.010, s), 0.006, closed=False), m.skin_fixed("頭"))

        # 鼻は小さな影だけ
        m.add("nose", h.decal(Head.dir(0.03, self.mouth["nose_pitch"]), ellipse_ab(0, 0, 0.0013, 0.0019), 0.010), m.skin_fixed("頭"))

        # 口
        M = self.mouth
        md = Head.dir(0.0, self.mouth["pitch"])

        def mouth(rx, ry, smile, bottom=1.0):
            return ellipse_ab(0, 0, rx, ry, top=0.35, bottom=bottom, curve=smile * 0.5)

        base_ab = mouth(M["rx"], M["ry"], M["smile"], M["open_bottom"])
        base = m.add("mouth", h.decal(md, base_ab, 0.012), m.skin_fixed("頭"))
        for name, (sx, sy) in {"あ": (1.05, 0.0105), "い": (1.35, 0.0045), "う": (0.62, 0.0065),
                               "え": (1.15, 0.0070), "お": (0.95, 0.0095)}.items():
            m.add_morph_part(name, base, h.decal(md, mouth(M["rx"] * sx, max(sy, M["ry"]), M["smile"] * 0.4, 1.0),
                                                 0.012)[0])

    # ---- 髪 ----
    def hair_material(self, key, name, color, tip=None):
        """筋の入った髪テクスチャと、上寄りに光の輪が出るスフィアを付けた材質。"""
        T = self.tex
        return self.m.material(key, name, color, tex=T.hair(f"hair_{key}.png", color, tip),
                               sphere=T.hair_sphere(), sphere_mode=2)

    def hair_surface(self, scale=(1.06, 1.02, 1.07), lift=(0, 0.000, 0.006)):
        return self.head.c + np.array(lift), self.head.r * np.array(scale)

    def hair_cap(self, mat, keep=None, scale=(1.06, 1.02, 1.07)):
        c, r = self.hair_surface(scale)
        if keep is None:
            def keep(d):
                face = (d[:, 2] < -0.15) & (d[:, 1] < 0.50)
                return ~face & (d[:, 1] > -0.40)
        self.m.add(mat, ellipsoid(c, r, nu=36, nv=22, keep=keep), self.m.skin_fixed("頭"))

    def strand(self, path, widths, thick, sides, mats, blunt=False, skin=None):
        """髪の束。mats は各点の材質（途中で色を変えられる）。"""
        skin = skin or self.skin_hair()
        path = np.asarray(path, float)
        n = len(path)
        widths = np.broadcast_to(np.asarray(widths, float), (n,))
        thick = np.broadcast_to(np.asarray(thick, float), (n,))
        tg = norm(np.gradient(path, axis=0))
        sides = norm(np.asarray(sides, float) - np.einsum("ij,ij->i", np.asarray(sides, float), tg)[:, None] * tg)
        nn = norm(np.cross(tg, sides))
        # 断面は外側がふくらんだ三日月形にして、束に厚みを出す（nn は頭の内側を向く）
        a = np.arange(12) / 12 * TAU
        ca, sa = np.cos(a), np.sin(a)
        rings = [path[i] + np.outer(ca * widths[i], sides[i])
                 + np.outer(sa * thick[i] - 1.3 * thick[i] * (1 - ca ** 2), nn[i]) for i in range(n)]
        if isinstance(mats, str):
            mats = [mats] * n
        start = 0
        for i in range(1, n + 1):
            if i == n or mats[i] != mats[start]:
                end = min(i, n - 1)
                if end > start:
                    mesh = loft(rings[start:end + 1], cap0=(start == 0), cap1=(end == n - 1 and blunt),
                                centers=list(path[start:end + 1]))
                    self.m.add(mats[start], mesh, skin)
                start = i

    def bang(self, mat, theta, e_end, width, thick=0.006, e_start=1.25, pointy=True, scale=(1.06, 1.02, 1.07),
             lift_end=1.07, mats=None, sway=0.0):
        c, r = self.hair_surface(scale)
        phis = np.linspace(e_start, e_end, 12)
        t = (e_start - phis) / (e_start - e_end)
        k = 1.0 + (lift_end - 1.0) * t ** 2
        th = theta + sway * t ** 2
        dirs = np.stack([np.sin(th) * np.cos(phis), np.sin(phis), -np.cos(th) * np.cos(phis)], 1)
        path = c + dirs * r * k[:, None]
        sides = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1)
        w = width * (1 - 0.85 * t ** 3) if pointy else np.full_like(t, width)
        self.strand(path, w, thick, sides, mats or mat, blunt=not pointy)

    def drape(self, mat, theta, y_end, width, thick=0.007, e0=1.05, flare=0.10, pointy=True,
              scale=(1.06, 1.02, 1.07), mats_fn=None, clearance=(0.100, 0.090, 0.074), wave=0.0, bones=0, splay=0.0):
        """頭から垂れ下がる髪の束。体（肩・胸）をよけて外側に広がる。bones を渡すと物理で揺れる。
        splay はあごの高さから下へ行くほど外へ開く量（姫毛を顔から離して広げる）。"""
        c, r = self.hair_surface(scale)
        h = np.array([np.sin(theta), 0.0, -np.cos(theta)])
        pts = []
        for phi in np.linspace(e0, 0.0, 8, endpoint=False):
            d = np.array([np.sin(theta) * np.cos(phi), np.sin(phi), -np.cos(theta) * np.cos(phi)])
            pts.append(c + d * r)
        req = 1.0 / np.sqrt((np.sin(theta) / r[0]) ** 2 + (np.cos(theta) / r[2]) ** 2)
        b = clearance[1] if np.cos(theta) > 0 else clearance[2]
        rcl = 1.0 / np.sqrt((np.sin(theta) / clearance[0]) ** 2 + (np.cos(theta) / b) ** 2)
        ys = np.linspace(c[1], y_end, 16)
        for i, y in enumerate(ys):
            t = 1 - smoothstep(SHOULDER_Y + 0.01, c[1], y)
            rad = req + (max(req, rcl * 1.06) - req) * t + flare * max(0.0, SHOULDER_Y - y) + splay * max(0.0, c[1] - 0.06 - y)
            wv = wave * np.sin((c[1] - y) * 40 + theta * 3) * min(1.0, (c[1] - y) * 8)
            pts.append(np.array([c[0], y, c[2]]) + h * (rad + wv))
        path = catmull(pts, 30)
        n = len(path)
        tt = np.linspace(0, 1, n)
        w = width * (1 - 0.9 * smoothstep(0.75, 1.0, tt)) if pointy else np.full(n, width)
        w = w * (0.75 + 0.25 * smoothstep(0.0, 0.25, tt))
        sides = np.tile([np.cos(theta), 0.0, np.sin(theta)], (n, 1))
        mats = [mats_fn(p) for p in path] if mats_fn else mat
        self.strand(path, w, thick, sides, mats, blunt=not pointy,
                    skin=self.hair_chain(path, c[1] - 0.03, bones, width * 0.6) if bones else None)

    def hair_chain(self, path, y_from, bones, radius, parent="頭"):
        """path の y_from より下に揺れる骨を通し、そのウェイト関数を返す。"""
        ph = physics()
        self._chains += 1
        pts = ph.resample(path, y_from, bones + 1)
        names, _ = ph.chain(self, f"髪{self._chains}_", pts, parent, radius)
        return ph.skin_chain(parent, names, pts)


def physics():
    from . import physics as ph  # physics は body を読み込むので、使うときに読む
    return ph


def _samples(profile_keys, band_keys, count, descending=True):
    lo, hi = min(profile_keys), max(profile_keys)
    base = np.linspace(hi, lo, count) if descending else np.linspace(lo, hi, count)
    extra = [k for k in band_keys if lo < k < hi]
    allk = np.unique(np.concatenate([base, extra, profile_keys]))
    return allk[::-1] if descending else allk


def _band_at(bands, key, descending=True):
    """bands は [(境目, 材質)] を体の上（腕なら付け根）から順に。key が属する材質を返す。"""
    cur = bands[0][1]
    for k, mat in bands:
        if (descending and key <= k) or (not descending and key >= k):
            cur = mat
    return cur


def finish(model):
    """表示枠を整える。"""
    m = model
    B = m.bone
    m.frames.append(("Root", "Root", 1, [(0, B["全ての親"])]))
    m.frames.append(("表情", "Exp", 1, [(1, i) for i in range(len(m.morphs))]))
    groups = [
        ("センター", ["センター", "グルーブ"]),
        ("体(上)", ["上半身", "上半身2", "首", "頭", "両目", "左目", "右目"]),
        ("腕", [p + n for p in "左右" for n in ("肩", "腕", "ひじ", "手首")]),
        ("体(下)", ["下半身"]),
        ("足", [p + n for p in "左右" for n in ("足", "ひざ", "足首", "足ＩＫ", "つま先ＩＫ")]),
    ]
    extra = [b.name for b in m.bones if b.name not in {n for _, ns in groups for n in ns}
             and b.name not in ("全ての親",) and "つま先" not in b.name]
    if extra:
        groups.append(("その他", extra))
    for name, names in groups:
        m.frames.append((name, name, 0, [(0, B[n]) for n in names if n in B]))
