"""共通の素体：ボーン（5本指ボーン含む）、体、5本指の手、顔（立体的な目・口内/歯/舌・鼻筋）、髪の部品。"""

import numpy as np

from . import pmx
from .builder import _merge_seams
from .geom import (UP, MeshData, catmull, ellipsoid, loft, norm, ring, rot_axes, smoothstep, tube, vertex_normals, TAU)

ARM_ANGLE = np.radians(37)
SIDES = (("左", "L", 1.0), ("右", "R", -1.0))

SHOULDER_Y = 0.800
NECK_Y = 0.805

TORSO = [  # y, rx, rz, cz（cz は +Z = 背中側へのずれ）
    # 肩の上はアニメの 3D モデル（ref の標準体形）に合わせ、首のすぐ下で平らに張り、肩先で丸く落ちる。
    # 首の付け根の肉（僧帽筋）は首の後ろ寄りに付く
    (0.840, 0.021, 0.019, 0.008),
    (0.835, 0.036, 0.020, 0.010),
    (0.830, 0.064, 0.022, 0.011),
    (0.825, 0.077, 0.024, 0.010),
    (0.818, 0.088, 0.027, 0.009),
    (0.810, 0.093, 0.031, 0.007),
    (0.800, 0.090, 0.036, 0.004),
    (0.788, 0.086, 0.041, 0.003),
    (0.774, 0.081, 0.045, 0.002),
    (0.758, 0.076, 0.047, 0.001),
    (0.735, 0.071, 0.049, 0.000),
    (0.700, 0.064, 0.047, 0.000),
    (0.670, 0.056, 0.042, 0.001),
    (0.640, 0.049, 0.038, 0.003),
    (0.618, 0.047, 0.037, 0.004),
    (0.595, 0.051, 0.039, 0.004),
    (0.570, 0.062, 0.045, 0.005),
    (0.545, 0.072, 0.049, 0.007),
    (0.520, 0.076, 0.050, 0.008),
    (0.495, 0.068, 0.046, 0.008),
    (0.475, 0.042, 0.032, 0.006),
]

# くびれ〜腰回りの輪郭に折れ目が出ないよう、Catmull-Rom スプラインで滑らかに補間した断面表を作る
_TORSO_DENSE = catmull(np.array(TORSO, float), 16)
_TORSO_ORDER = np.argsort(_TORSO_DENSE[:, 0])
_TORSO_Y = _TORSO_DENSE[_TORSO_ORDER, 0]
_TORSO_RX = _TORSO_DENSE[_TORSO_ORDER, 1]
_TORSO_RZ = _TORSO_DENSE[_TORSO_ORDER, 2]
_TORSO_CZ = _TORSO_DENSE[_TORSO_ORDER, 3]


def torso_profile(y, grow=1.0):
    """高さ y における滑らかな胴体断面 (rx, rz, cz)。"""
    rx = np.interp(y, _TORSO_Y, _TORSO_RX) * grow
    rz = np.interp(y, _TORSO_Y, _TORSO_RZ) * grow
    cz = np.interp(y, _TORSO_Y, _TORSO_CZ)
    return rx, rz, cz

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
    # アニメの 3D モデルの腕は細い筒（ref の標準体形で半径 0.014〜0.015）
    (-0.025, 0.013),
    (0.000, 0.0165),
    (0.070, 0.0155),
    (0.150, 0.0135),
    (0.200, 0.0135),
    (0.265, 0.0106),
    (0.278, 0.0092),
]

# 手首座標系 (d: 腕の向き, up: 手の甲側, bk: +Z 小指側) での各指の関節位置 (u, v, z) と太さ (rv, rz)
FINGER_SPECS = [
    ("親指", "thumb", "０１２", [
        (0.0055, -0.0040, -0.0095),
        (0.0178, -0.0074, -0.0235),
        (0.0276, -0.0090, -0.0322),
        (0.0368, -0.0100, -0.0392),
    ], [(0.0062, 0.0065), (0.0046, 0.0048), (0.0036, 0.0038), (0.0026, 0.0028)]),
    ("人指", "fore", "１２３", [
        (0.0308, -0.0006, -0.0098),
        (0.0478, -0.0009, -0.0118),
        (0.0592, -0.0015, -0.0133),
        (0.0680, -0.0020, -0.0145),
    ], [(0.0041, 0.0036), (0.0032, 0.0030), (0.0026, 0.0025), (0.0019, 0.0020)]),
    ("中指", "middle", "１２３", [
        (0.0324, -0.0002, -0.0012),
        (0.0512, -0.0004, -0.0014),
        (0.0638, -0.0009, -0.0015),
        (0.0732, -0.0014, -0.0016),
    ], [(0.0043, 0.0037), (0.0034, 0.0031), (0.0027, 0.0026), (0.0020, 0.0021)]),
    ("薬指", "third", "１２３", [
        (0.0298, -0.0008,  0.0072),
        (0.0476, -0.0012,  0.0090),
        (0.0595, -0.0017,  0.0103),
        (0.0684, -0.0022,  0.0114),
    ], [(0.0041, 0.0035), (0.0031, 0.0029), (0.0025, 0.0024), (0.0019, 0.0019)]),
    ("小指", "little", "１２３", [
        (0.0260, -0.0018,  0.0145),
        (0.0410, -0.0023,  0.0176),
        (0.0500, -0.0028,  0.0198),
        (0.0578, -0.0033,  0.0215),
    ], [(0.0036, 0.0031), (0.0027, 0.0026), (0.0022, 0.0022), (0.0017, 0.0017)]),
]


def arm_dir(s):
    return np.array([s * np.cos(ARM_ANGLE), -np.sin(ARM_ANGLE), 0.0])


def hand_axes(s):
    """手首の局所座標系：(d=指先方向, up=手の甲方向, bk=+Z=小指方向)。"""
    d = arm_dir(s)
    up = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0.0])
    bk = np.array([0.0, 0.0, 1.0])
    return d, up, bk


def joints(s):
    """片側の関節位置（正規化単位）。"""
    d, up, bk = hand_axes(s)
    j = {}
    j["肩"] = np.array([s * 0.025, 0.796, 0.006])
    j["腕"] = np.array([s * 0.085, 0.790, 0.008])
    j["ひじ"] = j["腕"] + d * 0.150
    j["手首"] = j["ひじ"] + d * 0.130
    j["手先"] = j["手首"] + d * 0.075
    w = j["手首"]
    for fn, _, nums, pts, _ in FINGER_SPECS:
        for k, ch in enumerate(nums):
            u, v, z = pts[k]
            j[fn + ch] = w + d * u + up * v + bk * z
        u, v, z = pts[-1]
        j[fn + "先"] = w + d * u + up * v + bk * z
    j["足"] = np.array([s * 0.048, 0.530, 0.004])
    j["ひざ"] = np.array([s * 0.046, 0.280, -0.004])
    j["足首"] = np.array([s * 0.044, 0.052, 0.006])
    j["つま先"] = np.array([s * 0.044, 0.006, -0.062])
    return j


# ---------------------------------------------------------------- 頭と顔


def ear(c, n, up, back, a=0.0090, b=0.0155, K=11, J=44):
    """アニメ風の耳。輪郭は上が丸く耳たぶが細い C の形。前（顔側）の縁で頭に付く。
    外の面に凹凸を付ける：外の縁が巻いた耳の縁（耳輪）、その内側の山（対輪）、真ん中のくぼみ（耳甲介）、前の小さな出っ張り（耳珠）。
    c は耳の中心、n は耳の面の向き（外）、up は上、back は後ろ。a・b は前後・上下の半径。"""
    th = np.linspace(0, 2 * np.pi, J, endpoint=False)  # 0 が後ろ、π/2 が上、π が前、3π/2 が耳たぶ
    deg = np.degrees(th)
    cs, sn = np.cos(th), np.sin(th)
    ox = np.where(cs < 0, 0.75, 1.0) * cs * (1 - 0.30 * np.clip(-sn, 0, 1)) - 0.15 * np.clip(-sn, 0, 1)
    oy = sn
    # 耳輪は後ろ・上・前上を回り、前（顔に付く側）と耳たぶの前では消える
    rim_on = np.clip(np.minimum((190 - deg) / 40, 1), 0, 1) + np.clip((deg - 250) / 40, 0, 1)
    rim_on = np.clip(rim_on, 0, 1)
    anti_on = np.clip(1 - np.abs(((deg + 180) % 360 - 180) - 45) / 75, 0, 1)

    def relief(t, j):
        x, y = t * ox[j], t * oy[j]
        hgt = 0.0012 + 0.0034 * np.exp(-((t - 0.84) / 0.11) ** 2) * rim_on[j]
        hgt += 0.0016 * np.exp(-((t - 0.56) / 0.10) ** 2) * anti_on[j]
        hgt -= 0.0030 * np.exp(-((x + 0.12) / 0.45) ** 2 - ((y + 0.05) / 0.45) ** 2)
        hgt += 0.0020 * np.exp(-((x + 0.78) / 0.18) ** 2 - ((y + 0.22) / 0.22) ** 2)
        hgt -= 0.0010 * max(0.0, t - 0.93) / 0.07  # 縁は丸めて裏へ回す
        return x * a, y * b, hgt

    front, behind = [], []
    for k in range(K + 1):
        t = k / K
        for j in range(J):
            x, y, hg = relief(t, j)
            base = c + back * x + up * y
            front.append(base + n * hg)
            behind.append(base - n * (0.0020 + 0.0004 * t))
    V = np.array(front + behind)
    off = len(front)
    idx = lambda k, j: k * J + (j % J)
    faces = []
    for k in range(K):
        for j in range(J):
            a0, a1, b0, b1 = idx(k, j), idx(k, j + 1), idx(k + 1, j), idx(k + 1, j + 1)
            faces += [[a0, b0, a1], [a1, b0, b1], [a0 + off, a1 + off, b0 + off], [a1 + off, b1 + off, b0 + off]]
    for j in range(J):  # 外周の側面
        a0, a1 = idx(K, j), idx(K, j + 1)
        faces += [[a0, a1, a0 + off], [a1, a1 + off, a0 + off]]
    faces = np.array(faces)
    # 向きをそろえる：表は n 側、裏は -n 側、側面は外周の外向き
    tri = V[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(1) - c
    side = np.arange(len(faces)) >= 4 * K * J
    want = np.where(side, np.einsum("ij,ij->i", fn, cen), np.where(np.arange(len(faces)) % 4 < 2, fn @ n, -(fn @ n)))
    flip = want < 0
    faces[flip] = faces[flip][:, ::-1]
    # 耳輪の内側（くぼみ）は影の色にして、立体が読めるようにする。境目は格子の輪に沿わせてなめらかにする
    ring_k = np.arange(len(faces)) // (4 * J)
    inner = (np.arange(len(faces)) < 4 * K * J) & (np.arange(len(faces)) % 4 < 2) & (ring_k < round(K * 0.64))
    return MeshData(V, faces[~inner]), MeshData(V, faces[inner])


# 頭の形：あご先 0〜頭頂 1 の高さごとの水平断面。既製アニメ・MMDモデル（初音ミク・N4D等）の頭部を正規化した値。
# 横幅は目の高さで頭の高さの 0.75 倍に抑え（広いと顔が横に長く見える）、口の高さまでは幅を残してから
# 丸いあご先へ回り込む（2階差分が負の滑らかな凸曲線なので、頬は張らず、尖った V 字のあごにもならない）。
HEAD_H = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
          0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
HEAD_W = [0.050, 0.140, 0.215, 0.275, 0.315, 0.343, 0.360, 0.367, 0.373, 0.376, 0.377,
          0.377, 0.375, 0.372, 0.367, 0.356, 0.335, 0.310, 0.276, 0.218, 0.135]
HEAD_F = [0.318, 0.346, 0.372, 0.388, 0.395, 0.390, 0.386, 0.395, 0.405, 0.415, 0.436,
          0.447, 0.447, 0.440, 0.437, 0.417, 0.414, 0.390, 0.329, 0.260, 0.145]
HEAD_B = [0.09, 0.16, 0.236, 0.293, 0.347, 0.377, 0.399, 0.426, 0.442, 0.45, 0.455,
          0.455, 0.451, 0.448, 0.437, 0.425, 0.402, 0.368, 0.322, 0.254, 0.14]
HEAD_NOSE = [0.003, 0.005, 0.010, 0.020, 0.038, 0.044, 0.030, 0.018, 0.012, 0.008, 0.004,
             0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# あご〜頬〜こめかみの輪郭に折れ目が出ないよう、Catmull-Rom スプラインで滑らかに補間した断面表を作る
_HEAD_DENSE = catmull(np.column_stack([HEAD_H, HEAD_W, HEAD_F, HEAD_B, HEAD_NOSE]), 12)
_HEAD_ORDER = np.argsort(_HEAD_DENSE[:, 0])
_HEAD_H_S = _HEAD_DENSE[_HEAD_ORDER, 0]
_HEAD_W_S = _HEAD_DENSE[_HEAD_ORDER, 1]
_HEAD_F_S = _HEAD_DENSE[_HEAD_ORDER, 2]
_HEAD_B_S = _HEAD_DENSE[_HEAD_ORDER, 3]
_HEAD_NOSE_S = _HEAD_DENSE[_HEAD_ORDER, 4]


def _mix_c(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


class Head:
    """頭（肌の面）。高さごとの断面を、前後で半径の違う楕円に鼻の出っ張りを足した形でつなぐ。
    r は髪の面を置くための大きさの目安（楕円体の半径）として残している。"""

    def __init__(self, chin_y=0.860, height=0.138, z=-0.004, flat=2.50):
        self.H, self.y0, self.flat = height, chin_y, flat
        f, b = max(HEAD_F), max(HEAD_B)
        self.zc = z - (b - f) / 2 * height  # 前後のいちばん出た所の真ん中を z に合わせる
        self.c = np.array([0.0, chin_y + 0.52 * height, z])
        self.r = np.array([max(HEAD_W), 0.50, (f + b) / 2]) * height

    def radius(self, h, phi):
        """高さ h（あご 0〜頭頂 1）、正面から測った角 phi での断面の半径。"""
        h = np.clip(h, 0, 1)
        w, f, b, nose = (np.interp(h, _HEAD_H_S, t) * self.H for t in (_HEAD_W_S, _HEAD_F_S, _HEAD_B_S, _HEAD_NOSE_S))
        sp, cp = np.sin(phi), np.cos(phi)
        d = np.where(cp > 0, f, b)
        # あご〜フェイスライン下部（h < 0.35）は内側へ凹まないふっくらした超楕円（n >= 2.25）にし、美少女らしい頬の丸みを出す
        n_front = 2.25 + (self.flat - 2.25) * smoothstep(0.08, 0.34, h)
        n = np.where(cp > 0, n_front, 2.0)
        rho = (np.abs(sp / w) ** n + np.abs(cp / d) ** n) ** (-1 / n)
        return rho + nose * np.exp(-((rho * sp) / (0.044 * self.H)) ** 2) * np.clip(cp, 0, 1) ** 2.0

    def surface(self, h, phi):
        h_c = np.clip(h, 0, 1)
        rho = self.radius(h_c, phi)
        # 正面から見えるあご〜頬（|phi| <= 55°）の高さは一切変えず、耳の下〜うなじ（cp < 0.45）だけをなだらかに持ち上げて二重あごを防ぐ
        jaw_back = np.clip((0.45 - np.cos(phi)) / 1.45, 0.0, 1.0) ** 1.4
        jaw_lift = 0.065 * (1.0 - smoothstep(0.0, 0.32, h_c)) * jaw_back
        y = self.y0 + (h_c + jaw_lift) * self.H
        return np.stack([rho * np.sin(phi), y, self.zc - rho * np.cos(phi)], -1)

    def pitch_at(self, h, yaw=0.0):
        """あごから h（頭の高さ比）の高さで、正面から yaw の向きにある面の点を指す仰角。"""
        v = self.surface(h, yaw) - self.c
        return float(np.arctan2(v[1], np.hypot(v[0], v[2])))

    def point(self, d):
        """中心 c から向き d へ伸ばした線が頭の面と交わる点。"""
        d = norm(np.asarray(d, float))
        lo, hi = np.zeros(d.shape[:-1]), np.full(d.shape[:-1], 0.2)
        for _ in range(40):
            t = (lo + hi) / 2
            p = self.c + d * t[..., None]
            h = (p[..., 1] - self.y0) / self.H
            phi = np.arctan2(p[..., 0], self.zc - p[..., 2])
            inside = (np.hypot(p[..., 0], p[..., 2] - self.zc) < self.radius(h, phi)) & (h <= 1) & (h >= -0.05)
            lo, hi = np.where(inside, t, lo), np.where(inside, hi, t)
        return self.c + d * ((lo + hi) / 2)[..., None]

    @staticmethod
    def dir(yaw, pitch):
        return np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])

    def basis(self, d0):
        e1 = norm(np.cross(d0, UP))
        e2 = np.cross(e1, d0)
        return e1, e2

    def project(self, d0, ab, off, plane_dir=None):
        """接平面上の (a, b) を頭の表面に水平投影して貼り付ける。
        球中心からの放射投影だと顔の下部（口や目尻）で外側ほど Y が上へ引っ張られて歪むため、
        接平面上の高さ q_y をそのまま頭の表面の高さ h に対応させて歪みを防ぐ。"""
        p0 = self.point(d0)
        pdir = d0 if plane_dir is None else plane_dir
        e1 = norm(np.cross(np.array([pdir[0], 0.0, pdir[2]]), UP))
        e2 = UP
        ab = np.asarray(ab, float)
        q = p0 + ab[..., :1] * e1 + ab[..., 1:2] * e2
        h_q = np.clip((q[..., 1] - self.y0) / self.H, 0.0, 1.0)
        phi_q = np.arctan2(q[..., 0], self.zc - q[..., 2])
        p = self.surface(h_q, phi_q)
        out = norm(np.stack([p[..., 0], (p[..., 1] - self.c[1]) * 0.25, p[..., 2] - self.zc], -1))
        return p + out * (off * 0.085)

    def mesh(self, nh=48, nphi=64):
        hs = np.linspace(0, 1, nh)
        phis = np.arange(nphi) / nphi * TAU
        rings = [self.surface(h, phis) for h in hs]
        return loft(rings, closed=True, cap0=True, cap1=True)

    def decal(self, d0, rings_ab, off, closed=True, plane_dir=None):
        rings = [self.project(d0, r, off, plane_dir=plane_dir) for r in rings_ab]
        return _flat_loft(rings, d0 if plane_dir is None else plane_dir, closed)


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


def mouth_grid_ab(rx, ry, smile=0.0, top=0.45, bottom=1.0, cols=15, rows=9):
    """UV が (左→右, 上唇→下唇) に素直に対応する口内の格子。歯・舌・口内テクスチャが歪まず貼れる。"""
    us = np.linspace(-0.98, 0.98, cols)
    vs = np.linspace(1.0, -1.0, rows)  # +1 が上唇、-1 が下唇
    out = []
    for v in vs:
        a = rx * us
        half = np.sqrt(np.maximum(0.0, 1.0 - us ** 2))
        b_up = ry * top * half
        b_dn = -ry * bottom * half
        arch = smile * 0.35 * (us ** 2) * ry
        t = (1.0 - v) * 0.5  # 0 (上) .. 1 (下)
        b = b_up * (1.0 - t) + b_dn * t + arch
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
        # 目・鼻・口の高さは、あご先からの高さ（頭の肌の高さ比）で決める（MMD美少女モデルの頭部の黄金比：目は下半分に大きく、瞳は目頭際に寄せて内側の白目をほぼゼロにする）
        self.eye = dict(yaw=0.690, h=0.420, rx=0.0228, ry=0.0084, top=0.88, tilt=0.0, iris=0.56, iris_in=0.28)
        self.eye.update(eye or {})
        self.eye["pitch"] = self.head.pitch_at(self.eye["h"], self.eye["yaw"])
        self.mouth = dict(rx=0.0080, ry=0.0010, smile=1.0, open_bottom=1.0, h=0.137, nose_h=0.232, fang=False)
        self.mouth.update(mouth or {})
        self.mouth["pitch"] = self.head.pitch_at(self.mouth["h"])
        self.mouth["nose_pitch"] = self.head.pitch_at(self.mouth["nose_h"], 0.02)
        m = model
        m.default_toon_tex = tex.toon("toon_cloth.png", (0.82, 0.80, 0.89))
        m.material("skin", "肌", skin_color, toon_tex=tex.toon("toon_skin.png", (1.0, 0.85, 0.80)))
        m.material("ear_in", "耳の内側", np.asarray(skin_color) * np.array([0.88, 0.72, 0.70]),
                   toon_tex=tex.toon("toon_skin.png", (1.0, 0.85, 0.80)), edge=False)
        self._rig()
        self._chains = 0
        self._head_info = None
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
            m.add_bone(jp + "手首", "wrist_" + en, j["手首"], jp + "ひじ", tail=j["手先"] - j["手首"], seg_to=j["中指１"])
            # 5本指ボーン（親指０〜２、人指・中指・薬指・小指１〜３）
            for fn, fen, nums, _, _ in FINGER_SPECS:
                par = jp + "手首"
                for idx, ch in enumerate(nums):
                    bname = jp + fn + ch
                    ename = f"{fen}{idx + (0 if fn == '親指' else 1)}_{en}"
                    nxt = j[fn + nums[idx + 1]] if idx + 1 < len(nums) else j[fn + "先"]
                    tail_arg = (jp + fn + nums[idx + 1]) if idx + 1 < len(nums) else tuple(nxt - j[fn + ch])
                    m.add_bone(bname, ename, j[fn + ch], par, tail=tail_arg, seg_to=nxt)
                    par = bname
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
        return self.smooth_joints(self.m.skin_auto(["下半身", jp + "足", jp + "ひざ", jp + "足首"], {"下半身": 1.5}),
                                  [(jp + "足", jp + "ひざ", 0.040), (jp + "ひざ", jp + "足首", 0.025)])

    def skin_arm(self, jp):
        return self.smooth_joints(self.m.skin_auto(["上半身2", jp + "肩", jp + "腕", jp + "ひじ", jp + "手首"],
                                                   {"上半身2": 1.4, jp + "肩": 1.3}),
                                  [(jp + "腕", jp + "ひじ", 0.030), (jp + "ひじ", jp + "手首", 0.020)])

    def smooth_joints(self, f, pairs):
        """ひじ・ひざのまわりは、親の骨の向きに沿って幅 width のあいだで親→子へ線形に移す。
        距離だけで振り分けると関節で重みが急に切り替わり、曲げたとき外側がとがって内側がつぶれる。"""
        m = self.m

        def g(p):
            out = f(p)
            for par, ch, width in pairs:
                a, e = m.seg[par]
                d = (p - e) @ ((e - a) / np.linalg.norm(e - a))
                for i in np.nonzero(np.abs(d) < width)[0]:
                    if out[i][0] in (par, ch):
                        out[i] = (ch, par, float(np.clip(0.5 + d[i] / (2 * width), 0, 1)))
            return out
        return g

    def skin_hair(self):
        return self.m.skin_vertical("頭", "上半身2", 0.80, 0.70)

    def skin_skirt(self, y_top, y_bot):
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
    def torso_ring_point(self, y, ang, offset=0.0):
        """高さ y、角度 ang（0=+X左, -π/2=-Z正面, +π/2=+Z背中）における胸・谷間・へそ込みの胴体表面座標。"""
        y_arr = np.asarray(y, float)
        ang_arr = np.asarray(ang, float)
        rx, rz, cz = torso_profile(y_arr)
        ca, sa = np.cos(ang_arr), np.sin(ang_arr)
        x0 = rx * ca
        z0 = cz + rz * sa
        dx_total = np.zeros_like(x0)
        dz_total = np.zeros_like(z0)
        front = np.clip(-sa, 0.0, 1.0)
        if self.bust > 0:
            # 胸全体のボリュームが前へ張り出し、中央（x=0）に自然な谷間のくぼみができる既製アニメモデル準拠の胸形状
            y_apex = 0.706
            dy = np.where(y_arr >= y_apex, (y_arr - y_apex) / 0.074, (y_arr - y_apex) / 0.046)
            t_up = np.clip(dy, 0.0, 1.0)
            t_dn = np.clip(-dy, 0.0, 1.0)
            uy = np.clip(1.0 - dy ** 2, 0.0, 1.0)
            prof_y = (uy ** (0.65 + 0.55 * t_up - 0.18 * t_dn)) * smoothstep(0.0, 0.15, uy)

            ux = np.clip(1.0 - (x0 / 0.068) ** 2, 0.0, 1.0)
            cleavage_dip = 0.22 * np.exp(-((x0 / 0.011) ** 2))
            twin_peak = (ux ** 0.55) * smoothstep(0.0, 0.12, ux) * (
                1.0 + 0.14 * np.exp(-(((np.abs(x0) - 0.032) / 0.020) ** 2)) - cleavage_dip
            )
            b_scale = self.bust ** 1.35
            f_mask = front ** 1.2
            dz_total -= 0.0275 * b_scale * prof_y * twin_peak * f_mask
            dx_total += np.sign(x0) * 0.0040 * b_scale * prof_y * np.clip((np.abs(x0) - 0.025) / 0.040, 0.0, 1.0) * f_mask
        # へその立体的な縦くぼみ（y ≈ 0.606, x = 0, 正面）
        nav_r2 = (x0 / 0.0038) ** 2 + ((y_arr - 0.606) / 0.0068) ** 2
        dz_total += 0.0024 * np.exp(-nav_r2) * (front ** 2)
        x = x0 + dx_total
        z = z0 + dz_total
        if np.any(offset != 0.0):
            vx, vz = x, z - cz
            vm = np.hypot(vx, vz) + 1e-9
            x = x + offset * (vx / vm)
            z = z + offset * (vz / vm)
        return np.stack([x, np.broadcast_to(y_arr, x.shape), z], -1)

    def torso_point_xy(self, x, y, sg=-1, offset=0.0):
        """胴体の (x, y) における前（sg=-1）または後ろ（sg=+1）の 3D 表面座標。"""
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        rx, _, _ = torso_profile(y_arr)
        ca = np.clip(x_arr / np.maximum(rx, 1e-6), -1.0, 1.0)
        ang = sg * np.arccos(ca)
        return self.torso_ring_point(y_arr, ang, offset=offset)

    def build_body(self, torso_bands, leg_bands, arm_bands, neck_mat="skin", hand_mat="skin", leg_cut=None,
                   torso_mat_fn=None, torso_snap_fn=None):
        """bands: [(境目, 材質)]。torso/leg は y（上から）、arm は腕の付け根からの距離。
        leg_cut=(y, 高さ) で、脚の y の境目を外側ほど持ち上げる（ショーツの脚の切れ込み）。
        torso_mat_fn(c_pts, base_mat) を渡すと、胴体メッシュの三角形ごとに材質を切り替えられる（胸元の開口や袖ぐり）。"""
        m = self.m
        h = self.head
        hmesh = h.mesh()
        self._head_mesh = hmesh

        def head_normals(v):
            # アニメ顔法線（スフィア法線転写）：輪郭線の押し出しが均一になる滑らかな楕円体法線をベースにしつつ、
            # 顔前面（額・頬・鼻下・あご）の法線を正面〜やや上向きに揃えて、鼻横や口元の汚い自己陰影を消し、
            # 斜めから見たときだけ頬の外縁とあご下に柔らかいアニメ調の影落ちが出るようにする
            base_n = norm((v - h.c) / h.r)
            front_w = np.clip((h.zc - v[:, 2] + 0.006) / 0.036, 0.0, 1.0) * np.clip(1.0 - np.abs(v[:, 0]) / 0.052, 0.0, 1.0)
            target_n = norm(np.column_stack([base_n[:, 0] * 0.42, np.clip(base_n[:, 1] * 0.35 + 0.18, 0.08, 0.45), np.full(len(v), -0.90)]))
            return norm(base_n * (1.0 - front_w[:, None]) + target_n * front_w[:, None])

        self._head_info = m.add("skin", hmesh, m.skin_fixed("頭"), normals=head_normals)
        # 耳：アニメの 3D モデル（ref の N4D）に合わせ、目の中ほどから鼻の下までの高さ（h=0.35）、
        # 頭の横の少し後ろに控えめに添える
        for s in (1, -1):
            phi = s * (np.pi / 2 + 0.16)
            p = h.surface(0.35, phi)
            out = np.array([np.sin(phi), 0.0, -np.cos(phi)])          # 頭の外向き
            back = np.array([-np.cos(phi) * s, 0.0, -np.sin(phi) * s])  # 頭の表面に沿って後ろ向き
            back = norm(back - out * (back @ out))
            up = norm(np.array([0.0, 1.0, 0.18]))                     # 上をやや後ろへ倒す
            n = norm(out * np.cos(0.42) - back * np.sin(0.42))          # 面は外のやや前を向く
            n, up = norm(n * np.cos(0.18) - up * np.sin(0.18)), norm(up * np.cos(0.18) + n * np.sin(0.18))
            bk = norm(np.cross(up, n)) * (1 if np.cross(up, n) @ back > 0 else -1)
            outer, inner = ear(p + back * 0.0065 - out * 0.0005, n, up, bk, a=0.0078, b=0.0138)
            m.add("skin", outer, m.skin_fixed("頭"))
            m.add("ear_in", inner, m.skin_fixed("頭"))
        neck = np.array([[0, 0.785, 0.008], [0, 0.835, 0.005], [0, 0.895, 0.006]])
        m.add(neck_mat, tube(neck, [0.0205, 0.0170, 0.0160], n=16), m.skin_auto(["上半身2", "首", "頭"], {"頭": 1.3}))

        # 胴体（胸・谷間・くびれ・へそを一体成形した滑らかな高密度メッシュ：楕円体の継ぎ目溝や2重メッシュの突き抜けが一切出ない）
        chest_ys = np.linspace(0.660, 0.780, 36)
        navel_ys = np.linspace(0.594, 0.618, 9)
        extra_keys = list(chest_ys) + list(navel_ys) + [b[0] for b in torso_bands]
        ys = _samples([t[0] for t in TORSO], extra_keys, 60)
        n_ang = 84
        angs = np.arange(n_ang) / n_ang * TAU
        rings, centers = [], []
        for y in ys:
            _, _, cz = torso_profile(y)
            r_pts = self.torso_ring_point(y, angs)
            if torso_snap_fn is not None:
                r_pts = torso_snap_fn(r_pts, y, angs)
            rings.append(r_pts)
            centers.append(np.array([0.0, y, cz]))
        self._banded(rings, centers, ys, torso_bands, self.skin_torso(), cap1=True, descending=True, mat_fn=torso_mat_fn)

        # おなかが肌として出ている場合は、へそのくぼみに繊細な陰影デカールを重ねる
        if _band_at(torso_bands, 0.606, descending=True) == "skin":
            noedge = dict(edge=False, shadow=False, shade=1.0, toon_tex=self.tex.toon("toon_flat.png", (1, 1, 1)))
            m.material("navel", "へそ", self.skin_color, tex=self.tex.navel("navel.png", self.skin_color), **noedge)
            nx = np.linspace(-0.0055, 0.0055, 7)
            ny = np.linspace(0.615, 0.597, 9)
            n_rings = [self.torso_point_xy(nx, np.full_like(nx, yy), sg=-1, offset=0.0005) for yy in ny]
            m.add("navel", _flat_loft(n_rings, np.array([0.0, 0.0, -1.0]), closed=False), m.skin_fixed("上半身"))

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
            # 肩の丸み（三角筋）：腕の向きに長い楕円体にして、玉ではなく腕へ流れる形にする
            m.add(sh_mat, ellipsoid(j["腕"] + d * 0.004 + np.array([-s * 0.004, 0.001, 0]), [0.024, 0.016, 0.018],
                                    rot_axes(d, v, np.cross(d, v)), nu=18, nv=10),
                  m.skin_auto(["上半身2", jp + "肩", jp + "腕"], {"上半身2": 1.2}))
            self.add_hand(jp, s, hand_mat)

    def add_hand(self, jp, s, mat):
        """立体的な5本指の手：手首から滑らかにつながる掌（母指球のふくらみ付き）と、関節ごとにウェイトの入った5本の指。"""
        m = self.m
        j = joints(s)
        d, up, bk = hand_axes(s)
        w = j["手首"]

        # 1) 掌（手首 u=-0.006 から指の付け根 u=0.032 まで）
        palm_slices = [
            # (u, v_center, z_center, rv, rz_thumb, rz_pinky, thumb_bulge)
            (-0.006,  0.0000,  0.0000, 0.0106, 0.0108, 0.0108, 0.0000),
            ( 0.000, -0.0004,  0.0002, 0.0098, 0.0118, 0.0122, 0.0012),
            ( 0.007, -0.0012,  0.0006, 0.0084, 0.0142, 0.0138, 0.0032),
            ( 0.015, -0.0014,  0.0015, 0.0074, 0.0148, 0.0152, 0.0028),
            ( 0.023, -0.0012,  0.0022, 0.0065, 0.0142, 0.0164, 0.0010),
            ( 0.029, -0.0011,  0.0024, 0.0056, 0.0136, 0.0168, 0.0000),
            ( 0.032, -0.0011,  0.0024, 0.0042, 0.0126, 0.0156, 0.0000),
        ]
        n_ring = 24
        ang = np.arange(n_ring) / n_ring * TAU
        ca, sa = np.cos(ang), np.sin(ang)  # ca: bk (+Z 小指 / -Z 親指), sa: up (+甲 / -掌)
        p_rings, p_centers = [], []
        for u, vc, zc, rv, rzt, rzp, tbulge in palm_slices:
            rz = np.where(ca < 0, rzt, rzp)
            # 掌側（sa < 0）かつ親指側（ca < 0）に母指球の柔らかなふくらみを足し、小指側は少し手前に引く
            bulge_v = -tbulge * np.clip(-ca, 0, 1) ** 1.4 * np.clip(-sa, 0, 1)
            arch_u = -0.0035 * (u / 0.032) * np.clip(ca, 0, 1) ** 1.6
            c = w + d * u + up * vc + bk * zc
            pts = (c + np.outer(arch_u, d)
                   + np.outer(sa * rv + bulge_v, up)
                   + np.outer(ca * rz, bk))
            p_rings.append(pts)
            p_centers.append(c)
        palm_bones = [jp + "ひじ", jp + "手首", jp + "親指０", jp + "人指１", jp + "中指１", jp + "薬指１", jp + "小指１"]
        palm_skin = self.smooth_joints(
            m.skin_auto(palm_bones, {jp + "ひじ": 1.35, jp + "親指０": 1.25}),
            [(jp + "ひじ", jp + "手首", 0.018)],
        )
        m.add(mat, loft(p_rings, cap0=True, cap1=True, centers=p_centers), palm_skin)

        # 2) 5本の指（各指の根元を掌の中に少し埋め込み、指先まで滑らかな断面でつなぐ）
        nf = 14
        fa = np.arange(nf) / nf * TAU
        fca, fsa = np.cos(fa), np.sin(fa)
        for fn, _, nums, pts_spec, rads_spec in FINGER_SPECS:
            pts = [w + d * u + up * v + bk * z for u, v, z in pts_spec]
            f_dir = norm(pts[1] - pts[0])
            root_pt = pts[0] - f_dir * (0.0045 if fn == "親指" else 0.0055)
            chain_pts = [root_pt] + pts
            chain_rads = [rads_spec[0]] + list(rads_spec)
            # 指のパスを細かく補間し、先端に丸いドームを付ける
            sub_pts, sub_rv, sub_rz = [], [], []
            segs = len(chain_pts) - 1
            for k in range(segs):
                steps = 4 if k < segs - 1 else 5
                for s_i in range(steps):
                    t = s_i / steps
                    sub_pts.append(chain_pts[k] * (1 - t) + chain_pts[k + 1] * t)
                    rv0, rz0 = chain_rads[k]
                    rv1, rz1 = chain_rads[k + 1]
                    # 関節のわずかな節のメリハリ
                    knuckle = 1.0 + (0.04 if s_i == 0 and k in (1, 2) else 0.0)
                    sub_rv.append((rv0 * (1 - t) + rv1 * t) * knuckle)
                    sub_rz.append((rz0 * (1 - t) + rz1 * t) * knuckle)
            tip_dir = norm(chain_pts[-1] - chain_pts[-2])
            rv_tip, rz_tip = chain_rads[-1]
            for tip_t, scale_t in [(0.0, 1.0), (0.45, 0.84), (0.80, 0.56), (1.0, 0.22)]:
                sub_pts.append(chain_pts[-1] + tip_dir * (rv_tip * 1.15 * tip_t))
                sub_rv.append(rv_tip * scale_t)
                sub_rz.append(rz_tip * scale_t)

            f_rings = []
            for p_i, rv_i, rz_i in zip(sub_pts, sub_rv, sub_rz):
                # 指の腹側（fsa < 0）を少しふっくらさせ、甲側をすっきりさせる
                pad = 1.0 + 0.08 * np.clip(-fsa, 0, 1)
                f_rings.append(p_i + np.outer(fsa * rv_i * pad, up) + np.outer(fca * rz_i, bk))

            fbones = [jp + "手首"] + [jp + fn + ch for ch in nums]
            fpairs = [(fbones[k], fbones[k + 1], 0.005) for k in range(len(fbones) - 1)]
            fskin = self.smooth_joints(m.skin_auto(fbones, {jp + "手首": 1.35}), fpairs)
            m.add(mat, loft(f_rings, cap0=True, cap1=True, centers=sub_pts), fskin)

    def _banded(self, rings, centers, keys, bands, skin, cap0=False, cap1=False, descending=True, mat_fn=None):
        """リング列を境目で切って材質ごとに張る（境目のリングと法線は全体メッシュで共有し、材質境界の輪郭線割れを防ぐ）。"""
        full_verts, full_faces = loft(rings, cap0=cap0, cap1=cap1, centers=centers)
        full_nrm = _merge_seams(full_verts, vertex_normals(full_verts, full_faces))
        nrm_map = {tuple(k): n for k, n in zip(np.round(full_verts * 1e5).astype(np.int64), full_nrm)}
        nrm_fn = lambda v: np.array([nrm_map[tuple(k)] for k in np.round(v * 1e5).astype(np.int64)])
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
            if mat_fn is None:
                self.m.add(mat, mesh, skin, normals=nrm_fn)
            else:
                verts, faces = mesh
                cen = verts[faces].mean(axis=1)
                f_mats = [mat_fn(c_pt, mat) for c_pt in cen]
                for sub_mat in dict.fromkeys(f_mats):
                    sel = np.array([fm == sub_mat for fm in f_mats])
                    self.m.add(sub_mat, MeshData(verts, faces[sel], mesh.uv), skin, normals=nrm_fn)

    # ---- 顔 ----
    def build_face(self, closed_mouth_color=(0.45, 0.16, 0.17), blush=True, brow_color=None):
        """目は白目・瞳・まつ毛の 3 層。口はリップラインと口内（上の歯・舌・口内奥）の 2 層で構成する。"""
        m = self.m
        h = self.head
        E = self.eye
        M = self.mouth
        T = self.tex
        is_closed_mouth = M["ry"] < 0.0022
        # 目や頬は陰影で灰色にならないよう、真っ白なトゥーンと環境色 1 で描いた色そのままにする
        noedge = dict(edge=False, shadow=False, shade=1.0, toon_tex=T.toon("toon_flat.png", (1, 1, 1)))
        # 3 層の板の範囲（A は rx 単位で目頭 - → 目尻 +、B は ry 単位で上 +）。テクスチャは同じ輪郭をこの範囲に描く
        k = E["rx"] / E["ry"]
        ext = {"eye_white": (-1.0, 1.0, -1.0, 1.0),
               # 瞳は縦長で白目より背が高く、上はまつ毛、下は下の縁で切られる
               "iris": (-E["iris_in"] - E["iris"], -E["iris_in"] + E["iris"], -1.10, 1.90),
               "lash": (-1.14, 1.45, -1.0, 1.85)}
        m.material("eye_white", "白目", (1, 1, 1), tex=T.eye_white(ext["eye_white"], k), **noedge)
        m.material("iris", "瞳", self.eye_color, tex=T.iris("iris.png", self.eye_color, ext["iris"], k), **noedge)
        m.material("lash", "まつ毛", (0.1, 0.06, 0.07), tex=T.lash(ext["lash"], k), **noedge)
        # 眉は髪色そのままでなく、赤みのある焦げ茶に寄せる（既製モデルの眉の色）
        bc = brow_color if brow_color is not None else _mix_c(np.asarray(self.hair_color) * 0.60, (0.40, 0.20, 0.20), 0.5)
        m.material("brow", "眉", bc, tex=T.brow("brow.png", bc), **noedge)
        if is_closed_mouth:
            m.material("mouth", "口", closed_mouth_color, tex=T.mouth_line("mouth_line.png", closed_mouth_color), **noedge)
        else:
            m.material("mouth", "口", (0.28, 0.10, 0.11), **noedge)
        m.material("mouth_in", "口内", (1, 1, 1), tex=T.mouth_interior("mouth_in.png", fang=M.get("fang", False)), **noedge)
        m.material("nose", "鼻", self.skin_color, tex=T.nose("nose.png", self.skin_color), **noedge)
        if blush:
            m.material("blush", "頬", (1.0, 0.5, 0.5), tex=T.blush(), **noedge)
        for name, en, panel in [("まばたき", "blink", 2), ("笑い", "smile", 2), ("ウィンク", "wink", 2),
                                ("ウィンク右", "wink R", 2), ("あ", "a", 3), ("い", "i", 3), ("う", "u", 3),
                                ("え", "e", 3), ("お", "o", 3),
                                ("にこり", "cheerful", 1), ("怒り", "angry", 1), ("困る", "sad", 1)]:
            m.define_morph(name, en, panel)

        rx, ry = E["rx"], E["ry"]
        lid = -0.25 * ry

        def closer(arch, keep=0.0, pivot=None):
            """まぶたを閉じた形。arch > 0 で ^（笑い）、< 0 で ‿（まばたき）。
            pivot の高さが閉じた線に来て、上下は keep 倍に縮む。まつ毛は帯（上の縁）を閉じた線に重ね、
            少し厚みを残して黒い線として見せる。弧は目の幅 rx で決め、どの層も同じ線に重なるようにする。"""
            p = lid if pivot is None else pivot

            def f(r):
                a = r[:, 0]
                # pivot より下（白目・瞳・下まぶた）は閉じた線の裏へたたむ
                r[:, 1] = lid + np.maximum(r[:, 1] - p, -0.15 * ry) * keep + arch * ry * np.clip(1 - (a / rx) ** 2, 0, 1)
                return r
            return f

        def plane(a_half, b_lo, b_hi, s, cx=0.0, cy=0.0, cols=11, rows=9):
            """接平面上の長方形。u は目尻方向が 1、v は上が 0。"""
            us = np.linspace(-1, 1, cols)
            out = []
            for b in np.linspace(b_hi, b_lo, rows):
                a = cx + us * a_half * s
                out.append(np.stack([a, np.full(cols, cy + b)], 1))
            return out

        for jp, en, s in SIDES:
            d0 = Head.dir(s * E["yaw"], E["pitch"])
            # 目の接平面は正面寄りに向け、こめかみの余白を残しつつ丸いドーム型の目が正面を向くようにする
            pdir = Head.dir(s * E["yaw"] * 0.30, E["pitch"] * 0.75)
            tilt = E["tilt"]

            def tilted(rows):
                return [np.stack([r[:, 0], r[:, 1] + tilt * r[:, 0] * s], 1) for r in rows]

            def layer(mat, cols, rows):
                A0, A1, B0, B1 = ext[mat]
                return plane(rx * (A1 - A0) / 2, B0 * ry, B1 * ry, s, cx=s * rx * (A0 + A1) / 2, cols=cols, rows=rows)

            layers = [
                ("eye_white", layer("eye_white", 13, 11), 0.0025, "頭"),
                ("iris", layer("iris", 11, 11), 0.0048, jp + "目"),
                ("lash", layer("lash", 17, 15), 0.0072, "頭"),
            ]
            for mat, rows, off, bone in layers:
                rows = tilted(rows)
                shut = dict(keep=0.2, pivot=1.0 * ry) if mat == "lash" else {}
                base = m.add(mat, h.decal(d0, rows, off, closed=False, plane_dir=pdir), m.skin_fixed(bone))
                blink = h.decal(d0, remap_ab(rows, closer(-0.20, **shut)), off, closed=False, plane_dir=pdir)[0]
                m.add_morph_part("まばたき", base, blink)
                m.add_morph_part("笑い", base, h.decal(d0, remap_ab(rows, closer(0.28, **shut)), off, closed=False, plane_dir=pdir)[0])
                m.add_morph_part("ウィンク" if s > 0 else "ウィンク右", base, blink)
            # 眉は目のすぐ上（目の高さの半分ほど上）、目頭側から目の真ん中あたりまで
            bd = Head.dir(s * E["yaw"], E["pitch"] + 0.21)
            bpdir = Head.dir(s * E["yaw"] * 0.38, E["pitch"] + 0.18)
            brow_rows = plane(rx * 0.62, -0.0040, 0.0040, s, cx=-s * 0.30 * rx)
            b_base = m.add("brow", h.decal(bd, brow_rows, 0.0085, closed=False, plane_dir=bpdir), m.skin_fixed("頭"))
            # 眉の表情モーフ（にこり・怒り・困る・まばたき・笑い）
            m.add_morph_part("笑い", b_base, h.decal(bd, remap_ab(brow_rows, lambda r: r + [0, -0.0025]), 0.0085, closed=False, plane_dir=bpdir)[0])
            m.add_morph_part("にこり", b_base, h.decal(bd, remap_ab(brow_rows, lambda r: np.stack([r[:, 0], r[:, 1] - 0.002 - 0.12 * (r[:, 0] / 0.023) ** 2 * 0.005], 1)), 0.0085, closed=False, plane_dir=bpdir)[0])
            m.add_morph_part("怒り", b_base, h.decal(bd, remap_ab(brow_rows, lambda r, s=s: np.stack([r[:, 0], r[:, 1] + 0.22 * r[:, 0] * s - 0.002], 1)), 0.0085, closed=False, plane_dir=bpdir)[0])
            m.add_morph_part("困る", b_base, h.decal(bd, remap_ab(brow_rows, lambda r, s=s: np.stack([r[:, 0], r[:, 1] - 0.22 * r[:, 0] * s + 0.002], 1)), 0.0085, closed=False, plane_dir=bpdir)[0])
            if blush:
                # 目の下の縁から口の高さまで、頬に広くふんわり置く
                bh = E["h"] - 0.12
                cd = Head.dir(s * 0.52, self.head.pitch_at(bh, 0.52))
                cpdir = Head.dir(s * 0.30, self.head.pitch_at(bh, 0.30))
                m.add("blush", h.decal(cd, plane(0.0210, -0.0095, 0.0095, s), 0.0020, closed=False, plane_dir=cpdir), m.skin_fixed("頭"))

        # 鼻：鼻先の淡いピンクの点
        nd = Head.dir(0.006, M["nose_pitch"])
        m.add("nose", h.decal(nd, plane(0.0028, -0.0034, 0.0034, 1.0, cols=5, rows=5), 0.0032, closed=False), m.skin_fixed("頭"))

        # 口：リップライン ("mouth") ＋ 歯・舌・口内奥 ("mouth_in")
        md = Head.dir(0.0, M["pitch"])
        if is_closed_mouth:
            # 普段は閉じた微笑みリップライン。口内メッシュは極小にして裏に隠し、あいうえおモーフで開く
            lip_base_ab = plane(M["rx"], -0.0048, 0.0048, 1.0, cols=19, rows=7)
            in_base_ab = mouth_grid_ab(M["rx"] * 0.65, 0.0002, M["smile"] * 0.2, top=0.45, bottom=0.55)
            base_lip = m.add("mouth", h.decal(md, lip_base_ab, 0.0075, closed=False), m.skin_fixed("頭"))
            base_in = m.add("mouth_in", h.decal(md, in_base_ab, 0.0010, closed=False), m.skin_fixed("頭"))
        else:
            # 普段から開いた元気な口（キリコ・フェリス）：外周のリップ縁 ("mouth") と内側の歯・舌 ("mouth_in")
            lip_base_ab = ellipse_ab(0, 0, M["rx"], M["ry"] * 1.06, top=0.45, bottom=M["open_bottom"], curve=M["smile"] * 0.35)
            in_base_ab = mouth_grid_ab(M["rx"] * 0.93, M["ry"], M["smile"], top=0.45, bottom=M["open_bottom"])
            base_lip = m.add("mouth", h.decal(md, lip_base_ab, 0.0028), m.skin_fixed("頭"))
            base_in = m.add("mouth_in", h.decal(md, in_base_ab, 0.0048, closed=False), m.skin_fixed("頭"))

        vowels = {
            "あ": (1.05, 0.0095, -0.3, 0.0040),
            "い": (1.28, 0.0042, -0.5, 0.0012),
            "う": (0.60, 0.0055,  0.1, 0.0020),
            "え": (1.14, 0.0068, -0.4, 0.0028),
            "お": (0.88, 0.0088,  0.0, 0.0038),
        }
        for name, (sx, sy, sm, jaw_drop) in vowels.items():
            rx_v = M["rx"] * sx
            ry_v = max(sy, M["ry"] * 0.85)
            if is_closed_mouth:
                # 閉じたリップラインを上唇のアーチへ移動させ、その下に口内（歯・舌）を大きく開く
                lip_v_ab = remap_ab(lip_base_ab, lambda r, sx=sx, ry_v=ry_v: np.stack([
                    r[:, 0] * sx,
                    r[:, 1] * 0.5 + ry_v * 0.42 * np.sqrt(np.maximum(0.0, 1.0 - (r[:, 0] / max(M["rx"], 1e-6)) ** 2)),
                ], 1))
                in_v_ab = mouth_grid_ab(rx_v * 0.94, ry_v, sm, top=0.45, bottom=1.0)
                m.add_morph_part(name, base_lip, h.decal(md, lip_v_ab, 0.0055, closed=False)[0])
                m.add_morph_part(name, base_in, h.decal(md, in_v_ab, 0.0042, closed=False)[0])
            else:
                lip_v_ab = ellipse_ab(0, 0, rx_v, ry_v * 1.06, top=0.45, bottom=1.0, curve=sm * 0.35)
                in_v_ab = mouth_grid_ab(rx_v * 0.93, ry_v, sm, top=0.45, bottom=1.0)
                m.add_morph_part(name, base_lip, h.decal(md, lip_v_ab, 0.0028)[0])
                m.add_morph_part(name, base_in, h.decal(md, in_v_ab, 0.0048, closed=False)[0])

            # 口を大きく開けるモーフでは、あご先の肌も自然に少し下がるようにする
            if self._head_info is not None and jaw_drop > 0:
                hv = self._head_mesh[0].copy()
                h_rel = (hv[:, 1] - h.y0) / h.H
                front_mask = np.clip((-hv[:, 2] - 0.015) / 0.035, 0.0, 1.0)
                chin_mask = np.clip((0.20 - h_rel) / 0.20, 0.0, 1.0) ** 1.3 * np.exp(-(hv[:, 0] / 0.032) ** 2)
                hv[:, 1] -= jaw_drop * front_mask * chin_mask
                m.add_morph_part(name, self._head_info, hv)

    # ---- 髪 ----
    def hair_material(self, key, name, color, tip=None):
        """筋の入った髪テクスチャと、上寄りに光の輪が出るスフィアを付けた材質。"""
        T = self.tex
        return self.m.material(key, name, color, tex=T.hair(f"hair_{key}.png", color, tip),
                               sphere=T.hair_sphere(), sphere_mode=2)

    def hair_surface(self, scale=(1.06, 1.04, 1.07), lift=(0, 0.002, 0.006)):
        return self.head.c + np.array(lift), self.head.r * np.array(scale)

    def hair_cap(self, mat, keep=None, scale=(1.06, 1.04, 1.07)):
        c, r = self.hair_surface(scale)
        if keep is None:
            def keep(d):
                face = (d[:, 2] < -0.22) & (np.abs(d[:, 0]) < 0.78) & (d[:, 1] < 0.48)
                return ~face & (d[:, 1] > -0.40)
        self.m.add(mat, ellipsoid(c, r, nu=36, nv=22, keep=keep), self.m.skin_fixed("頭"))

    def strand(self, path, widths, thick, sides, mats, blunt=False, skin=None):
        """髪の束。頭頂〜額はしっかり厚みを保って肌を覆い、毛先近くで薄く絞る。"""
        skin = skin or self.skin_hair()
        path = np.asarray(path, float)
        n = len(path)
        t_along = np.linspace(0.0, 1.0, n)
        widths = np.broadcast_to(np.asarray(widths, float), (n,))
        taper = 1.0 - 0.55 * (np.clip((t_along - 0.70) / 0.30, 0.0, 1.0) ** 1.6)
        thick_arr = np.broadcast_to(np.asarray(thick, float), (n,)) * taper
        tg = norm(np.gradient(path, axis=0))
        sides = norm(np.asarray(sides, float) - np.einsum("ij,ij->i", np.asarray(sides, float), tg)[:, None] * tg)
        nn = norm(np.cross(tg, sides))
        # 断面は外側がふくらんだ三日月型にする（nn は頭の内側を向く）
        a = np.arange(12) / 12 * TAU
        ca, sa = np.cos(a), np.sin(a)
        rings = [path[i] + np.outer(ca * widths[i], sides[i])
                 + np.outer(sa * thick_arr[i] - 1.3 * thick_arr[i] * (1 - ca ** 2), nn[i]) for i in range(n)]
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

    def bang(self, mat, theta, e_end, width, thick=0.0055, e_start=1.25, pointy=True, scale=(1.06, 1.04, 1.07),
             lift_end=1.065, mats=None, sway=0.0):
        c, r = self.hair_surface(scale)
        phis = np.linspace(e_start, e_end, 14)
        t = (e_start - phis) / (e_start - e_end)
        k = 1.0 + (lift_end - 1.0) * (t ** 2)
        th = theta + sway * (t ** 2)
        dirs = np.stack([np.sin(th) * np.cos(phis), np.sin(phis), -np.cos(th) * np.cos(phis)], 1)
        path = c + dirs * r * k[:, None]
        sides = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], 1)
        if pointy:
            w = width * (1 - 0.85 * (t ** 2.6))
        else:
            # 姫カット・ぱっつんでも毛先 40% をすっと絞って束ごとの繊細なすき間（眉や額がのぞく軽やかな抜け感）を作る
            w = width * (1 - 0.62 * np.clip((t - 0.58) / 0.42, 0.0, 1.0) ** 1.4)
        self.strand(path, w, thick, sides, mats or mat, blunt=False)

    def drape(self, mat, theta, y_end, width, thick=0.007, e0=1.05, flare=0.10, pointy=True,
              scale=(1.06, 1.04, 1.07), mats_fn=None, clearance=(0.100, 0.090, 0.074), wave=0.0, bones=0, splay=0.0):
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
    finger_bones = [
        f"{p}{fn}{ch}"
        for p in "左右"
        for fn, _, nums, _, _ in FINGER_SPECS
        for ch in nums
    ]
    groups = [
        ("センター", ["センター", "グルーブ"]),
        ("体(上)", ["上半身", "上半身2", "首", "頭", "両目", "左目", "右目"]),
        ("腕", [p + n for p in "左右" for n in ("肩", "腕", "ひじ", "手首")]),
        ("指", finger_bones),
        ("体(下)", ["下半身"]),
        ("足", [p + n for p in "左右" for n in ("足", "ひざ", "足首", "足ＩＫ", "つま先ＩＫ")]),
    ]
    extra = [b.name for b in m.bones if b.name not in {n for _, ns in groups for n in ns}
             and b.name not in ("全ての親",) and "つま先" not in b.name]
    if extra:
        groups.append(("その他", extra))
    for name, names in groups:
        m.frames.append((name, name, 0, [(0, B[n]) for n in names if n in B]))
