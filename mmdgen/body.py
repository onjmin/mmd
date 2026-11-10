"""共通の素体：ボーン（5本指ボーン含む）、体、5本指の手、顔（立体的な目・口内/歯/舌・鼻筋）、髪の部品。"""

import numpy as np

from . import head_shape as _HS
from .textures import eye_outline
from . import pmx
from . import face as _face
from . import bake as _bake
from .builder import _merge_seams
from .geom import (UP, MeshData, catmull, ellipsoid, loft, norm, ring, rot_axes, smoothstep, tube, vertex_normals, TAU)

ARM_ANGLE = np.radians(37)
SIDES = (("左", "L", 1.0), ("右", "R", -1.0))

SHOULDER_Y = 0.800
NECK_Y = 0.805

TORSO = [  # y, rx, rz, cz（cz は +Z = 背中側へのずれ）
    # 肩の上は既製の MMD モデル（音街ウナ・うか）に合わせたなで肩：首の付け根から肩先へ 30° ほどでなだらかに下がり、
    # 肩先で丸く腕へ落ちる（首から腕の付け根までの落差が頭の高さの 0.2 ほど）。平らに横へ張るといかり肩になる。
    # 首の付け根の肉（僧帽筋）は首の後ろ寄りに付く
    (0.840, 0.014, 0.013, 0.004),  # 首の中に隠れる
    (0.836, 0.030, 0.019, 0.010),
    (0.831, 0.047, 0.021, 0.011),
    (0.825, 0.061, 0.023, 0.010),
    (0.818, 0.072, 0.026, 0.009),
    (0.810, 0.081, 0.030, 0.007),
    (0.801, 0.087, 0.035, 0.005),
    (0.790, 0.088, 0.040, 0.003),
    (0.777, 0.084, 0.044, 0.002),
    (0.760, 0.077, 0.047, 0.001),
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
    # 既製の MMD モデル（音街ウナ）の脚は太ももで脚の長さの 0.065〜0.08、ひざ・ふくらはぎで 0.055〜0.06。
    # それより少しだけ肉付きを残して細くする
    (0.565, 0.044, 0.040, 0.004),
    (0.525, 0.053, 0.047, 0.004),
    (0.450, 0.047, 0.047, 0.000),
    (0.370, 0.0385, 0.046, -0.002),
    (0.300, 0.0305, 0.045, -0.004),
    (0.270, 0.0295, 0.045, -0.003),
    (0.220, 0.0318, 0.045, 0.004),
    (0.170, 0.0285, 0.044, 0.006),
    (0.120, 0.0225, 0.044, 0.006),
    (0.070, 0.0185, 0.044, 0.006),
    (0.040, 0.0180, 0.044, 0.006),
]

ARM = [  # 腕からの距離, 半径
    # アニメの 3D モデルの腕は細い筒（ref の標準体形で半径 0.014〜0.015）
    (-0.025, 0.013),
    (0.000, 0.0190),  # 既製の MMD モデルの上腕は頭の高さの 0.14 ほど（細すぎると棒に見える）
    (0.070, 0.0180),
    (0.150, 0.0150),
    (0.200, 0.0142),
    (0.265, 0.0106),
    (0.278, 0.0092),
]

# 腕の太さも折れ目が出ないよう滑らかに補間する
_ARM_DENSE = catmull(np.array(ARM, float), 60)
_ARM_T, _ARM_R = _ARM_DENSE[np.argsort(_ARM_DENSE[:, 0]), 0], _ARM_DENSE[np.argsort(_ARM_DENSE[:, 0]), 1]

# 手首座標系 (d: 腕の向き, up: 手の甲側, bk: +Z 小指側) での各指の関節位置 (u, v, z) と太さ (rv, rz)
FINGER_SPECS = [
    ("親指", "thumb", "０１２", [
        (0.0055, -0.0040, -0.0095),
        (0.0178, -0.0074, -0.0235),
        (0.0276, -0.0090, -0.0322),
        (0.0368, -0.0100, -0.0392),
    ], [(0.0062, 0.0065), (0.0046, 0.0048), (0.0036, 0.0038), (0.0026, 0.0028)]),
    # 4 本の指は力を抜いた手のように、先へ行くほど掌の側（v の負）へ少しずつ曲げる。まっすぐだと板のような手に見える
    ("人指", "fore", "１２３", [
        (0.0308, -0.0006, -0.0098),
        (0.0477, -0.0024, -0.0118),
        (0.0588, -0.0052, -0.0133),
        (0.0670, -0.0085, -0.0145),
    ], [(0.0041, 0.0037), (0.0033, 0.0031), (0.0027, 0.0026), (0.0022, 0.0022)]),
    ("中指", "middle", "１２３", [
        (0.0324, -0.0002, -0.0012),
        (0.0511, -0.0022, -0.0014),
        (0.0633, -0.0052, -0.0015),
        (0.0721, -0.0088, -0.0016),
    ], [(0.0043, 0.0038), (0.0035, 0.0032), (0.0028, 0.0027), (0.0023, 0.0023)]),
    ("薬指", "third", "１２３", [
        (0.0298, -0.0008,  0.0072),
        (0.0475, -0.0028,  0.0090),
        (0.0590, -0.0057,  0.0103),
        (0.0673, -0.0091,  0.0114),
    ], [(0.0041, 0.0036), (0.0032, 0.0030), (0.0026, 0.0025), (0.0021, 0.0021)]),
    ("小指", "little", "１２３", [
        (0.0260, -0.0018,  0.0145),
        (0.0409, -0.0036,  0.0176),
        (0.0496, -0.0060,  0.0198),
        (0.0569, -0.0088,  0.0215),
    ], [(0.0036, 0.0032), (0.0028, 0.0027), (0.0023, 0.0023), (0.0019, 0.0019)]),
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
    j["肩"] = np.array([s * 0.025, 0.792, 0.006])
    j["腕"] = np.array([s * 0.082, 0.779, 0.008])  # 首（0.805）より頭の高さの 0.19 ほど低い（なで肩）
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


# 頭の形：既製の美少女モデル（CG アバター「うか」、CC-BY 4.0）の頭を一周輪切りにして測った表（head_shape.py、
# python -m mmdgen.head_scan で作る）。目のくぼみ・鼻・口元・あご・あごの下・後頭部・首の付け根まで、頭の肌はこの表だけで作る。
_HS_V = np.array(_HS.V)
_HS_PHI = np.radians(_HS.PHI)
_HS_R = np.array(_HS.RHO, float)
_HS_OK = np.isfinite(_HS_R)
_HS_R0 = np.where(_HS_OK, _HS_R, 0.0)


def _flatten_face(R0, ok, relief=0.45):
    """顔の正面の凹凸（目のくぼみ・鼻・口元）を、大きくならした面へ relief の割合まで寄せる。
    3D の萌え顔はリアルな凹凸を減らした平らな顔の方がかわいく見える（鼻を尖らせない）。あご先と輪郭は元のまま。"""
    def gauss(T, axis, sigma, mode):
        k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
        k /= k.sum()
        pad = [(0, 0), (0, 0)]
        pad[axis] = (3 * sigma, 3 * sigma)
        T = np.pad(T, pad, mode=mode)
        return np.apply_along_axis(lambda c: np.convolve(c, k, "valid"), axis, T)
    w = ok.astype(float)
    num, den = R0 * w, w
    for axis, sigma, mode in ((0, 6, "edge"), (1, 6, "reflect")):  # 高さ ±0.12、向き ±15 度ほど
        num, den = gauss(num, axis, sigma, mode), gauss(den, axis, sigma, mode)
    smooth = num / np.maximum(den, 1e-9)
    V, P = np.meshgrid(_HS_V, np.degrees(_HS_PHI), indexing="ij")
    mask = (1 - smoothstep(45, 75, P)) * smoothstep(-0.95, -0.75, V) * (1 - smoothstep(0.35, 0.6, V))
    return np.where(ok, R0 + (smooth - R0) * (1 - relief) * mask, R0)


_HS_R0 = _flatten_face(_HS_R0, _HS_OK)
_HS_TOP = np.array([_HS_V[_HS_OK[:, j]].max() for j in range(len(_HS_PHI))])  # 向きごとの測れた上端
_HS_RTOP = np.array([_HS_R[_HS_OK[:, j], j][-1] for j in range(len(_HS_PHI))])  # その高さでの半径
HS_BOTTOM = float(_HS_V[0])  # 表の下端（首の付け根、目の線〜あご先 = 1 の単位）


def _hs_lookup(v, aphi, v_crown):
    """表を双線形補間した半径（目の線〜あご先で割った値）。aphi は正面から測った角（0〜π）。
    測れた上端より上（頭頂側）は、上端の半径から頭頂 v_crown で 0 になる楕円の弧で閉じる。"""
    v, aphi = np.broadcast_arrays(np.asarray(v, float), np.asarray(aphi, float))
    dv, dp = _HS_V[1] - _HS_V[0], _HS_PHI[1] - _HS_PHI[0]
    fv = np.clip((v - _HS_V[0]) / dv, 0, len(_HS_V) - 1 - 1e-9)
    fp = np.clip(aphi / dp, 0, len(_HS_PHI) - 1 - 1e-9)
    i, j = fv.astype(int), fp.astype(int)
    a, b = fv - i, fp - j

    def g(T):
        return T[i, j] * (1 - a) * (1 - b) + T[i + 1, j] * a * (1 - b) + T[i, j + 1] * (1 - a) * b + T[i + 1, j + 1] * a * b
    r = g(_HS_R0) / np.maximum(g(_HS_OK.astype(float)), 1e-9)
    top = np.interp(aphi, _HS_PHI, _HS_TOP)
    rtop = np.interp(aphi, _HS_PHI, _HS_RTOP)
    ext = rtop * np.sqrt(np.clip(1 - ((v - top) / np.maximum(v_crown - top, 1e-6)) ** 2, 0, 1))
    return np.where(v > top - 1e-9, ext, r)


def _mix_c(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


# 目の輪郭（目の板の (A, B) 単位）を中心から見た角度ごとの半径。くぼみをこの形で彫る
_EO = eye_outline()
_EO_T = np.arctan2(_EO[:, 1], _EO[:, 0])
_EO_ORDER = np.argsort(_EO_T)
_EO_T, _EO_R = _EO_T[_EO_ORDER], np.hypot(_EO[:, 0], _EO[:, 1])[_EO_ORDER]


def _eye_radius(a, b):
    """点 (a, b) が目の輪郭の何倍の所にあるか（輪郭の上で 1）。"""
    t = np.arctan2(b, a)
    R = np.interp(t, _EO_T, _EO_R, period=2 * np.pi)
    return np.hypot(a, b) / np.maximum(R, 1e-6)


# 目の板：顔の面から奥へ下げる量・板の向き（目の向きの何割外を向くか）・板の裏で肌を下げる量
EYE_INSET = 0.0006
EYE_TURN = 0.85
EYE_SOCKET_DEPTH = 0.0006
EYE_DOME = 0.0022  # 目の板の真ん中のふくらみ（眼球のゆるい丸み）
EYE_LID = 0.0016   # 上まぶたの帯の盛り上がり


# 萌えキャラの頭の比率（あご 0〜頭頂 1）：眉間 0.5、目尻 0.375、鼻先 0.25、口 0.125（頭頂〜あごの半分が眉間、眉間〜あごの半分が鼻先…）。
# 頭蓋は大人より横に広く丸く、目から下の顔は短く、あごへ向かって細く小さくなる（赤ちゃんの頭蓋骨に近い）。
# 横幅は既製の萌えキャラ（音街ウナ公式モデル）の正面の輪郭に合わせる：あご〜頭頂を 1 として、
# 顔の幅は口の高さ 0.76・目の高さ 0.76・おでこ 0.83（頬の丸い横に広い顔。大人のモデルは口の高さで細い V 字）。
HEAD_WIDTH = 0.52   # 横の大きさ（表の 1 が頭の高さの何倍か）。表の元の大人のモデルは 0.424
HEAD_DEPTH = 0.45   # 前後の大きさ（同じく）
JAW_X = -0.16       # 目の下からあごへ向かって横幅を変える割合（負で広げる＝頬を丸く張らせる）
JAW_Z = 0.0         # 同じく前後
NECK_W = 0.40       # あごの下〜首の横の大きさ（顔を広げても首は細いまま）
NECK_D = 0.40       # 同じく前後


class Head:
    """頭（肌の面）。既製の美少女モデルを一周輪切りにした表（_hs_lookup）を、萌えキャラの比率に伸び縮みさせて作る。
    高さ h はあご先 0〜頭頂 1。あごの下から首の付け根（h < 0）まで作り、首の付け根は体の首の中に隠れる。
    eye_h は目の線の高さで、表の「目の線〜あご先」をこの長さに合わせる（元の大人のモデルは 0.424、萌えキャラは 0.375）。
    横幅と奥行きは高さと別に HEAD_WIDTH・HEAD_DEPTH で決め、目から下は JAW_X・JAW_Z だけ細める。
    z は表の軸（目の高さで頭の前後の真ん中）の前後位置。元のモデルと同じく、首の中心より少し前に置く。
    r は髪の面を置くための大きさの目安（楕円体の半径）。"""

    def __init__(self, chin_y=0.860, height=0.138, z=-0.0075, eye_h=0.375, width=HEAD_WIDTH, jaw_x=JAW_X):
        self.H, self.y0, self.eye_h = height, chin_y, eye_h
        self.width, self.jaw_x = width, jaw_x
        self.L = eye_h * height  # 目の線〜あご先
        self.zc = z
        self.vc = 1.0 / eye_h - 1.0  # 頭頂（目の線からの高さ、目の線〜あご先 = 1）
        self.h_min = eye_h * (1.0 + HS_BOTTOM)  # 首の付け根
        self.c = np.array([0.0, chin_y + 0.52 * height, z])
        self.sockets = []
        vs = np.linspace(-1.0, _HS.V_CROWN, 60)
        side = _hs_lookup(vs, np.pi / 2, _HS.V_CROWN).max()
        fb = (_hs_lookup(vs, 0.0, _HS.V_CROWN).max() + _hs_lookup(vs, np.pi, _HS.V_CROWN).max()) / 2
        self.r = np.array([side * width, 0.50, fb * HEAD_DEPTH]) * height

    def _v(self, h):
        """高さ h を表の高さ v（目の線 0、あご先 -1）に直す。目の線より上は、表の頭頂がこの頭の頭頂に来るよう伸び縮みさせる。"""
        v = np.clip(h, self.h_min, 1) / self.eye_h - 1.0
        return np.where(v > 0, v * (_HS.V_CROWN / self.vc), v)

    def scales(self, h):
        """高さ h での表の 1 あたりの横・前後の長さ。"""
        t = smoothstep(-0.25, -0.85, self._v(h))
        return self.width * (1 - self.jaw_x * t) * self.H, HEAD_DEPTH * (1 - JAW_Z * t) * self.H

    def _neck(self, h, phi):
        """あごの下・首の所（1）か顔（0）か。あごの下の線はあご先（目の線〜あご先 = -1）から耳の下（-0.6）へ斜めに上がる。"""
        aphi = np.abs(np.arctan2(np.sin(phi), np.cos(phi)))
        v_line = -1.0 + 0.40 * np.sin(np.minimum(aphi, np.pi / 2)) ** 2
        return smoothstep(v_line + 0.02, v_line - 0.14, self._v(h))

    def table_phi(self, h, x, z):
        """点 (x, z) が表のどの向きにあるか（横・前後の伸び縮みを戻して測る）。"""
        sx, sz = self.scales(h)
        return np.arctan2(x / sx, (self.zc - z) / sz)

    def radius(self, h, phi):
        """高さ h（あご 0〜頭頂 1）、表の向き phi での断面の半径（表の値。横は scales の sx、前後は sz 倍して使う）。"""
        aphi = np.abs(np.arctan2(np.sin(phi), np.cos(phi)))
        return _hs_lookup(self._v(h), aphi, _HS.V_CROWN)

    def surface(self, h, phi):
        h_c = np.clip(h, self.h_min, 1)
        rho = self.radius(h_c, phi)
        sx, sz = self.scales(h_c)
        # あごの下と首は、顔を広げたぶんを戻して細いままにする（顔の貼り物を置く point・project は顔の所しか使わない）
        nk = self._neck(h_c, phi)
        sx = sx * (1 - nk) + NECK_W * self.H * nk
        sz = sz * (1 - nk) + NECK_D * self.H * nk
        x, zf = rho * np.sin(phi) * sx, rho * np.cos(phi) * sz
        y = self.y0 + h_c * self.H
        p = np.stack(np.broadcast_arrays(x, y, self.zc - zf), -1)
        return self._carve(p)

    def add_socket(self, p0, n, e1, rx, ry, depth, rim=1.30, lid=0.0):
        """目のくぼみ：目の板（p0 を通り n を向く平面、横 e1・縦 UP に半径 rx・ry）の裏へ、肌を depth だけ下げる。
        くぼみの形は目の輪郭（台形）どおりで、外へ rim 倍の所までなだらかに戻す。
        上の縁には lid だけ外へ盛り上がる帯（上まぶた）を付け、目の上にかぶさる段にする。"""
        self.sockets.append((np.asarray(p0, float), np.asarray(n, float), np.asarray(e1, float), rx, ry, depth, rim, lid))

    def _carve(self, p):
        for p0, n, e1, rx, ry, depth, rim, lid in getattr(self, "sockets", []):
            rel = p - p0
            a, b, d = rel @ e1 / rx, rel[..., 1] / ry, rel @ n
            r = _eye_radius(a, b)
            w = 1.0 - smoothstep(1.0, rim, r)
            push = np.maximum(d + depth, 0.0) * w * (d > -4 * depth)
            # 上まぶたの帯：目の上の縁のすぐ外で外へふくらむ（目頭・目尻の方は細る）
            top = smoothstep(0.15, 0.7, b / np.maximum(np.hypot(a, b), 1e-9))
            ridge = lid * np.exp(-((r - 1.12) / 0.13) ** 2) * top * (d > -4 * depth)
            p = p - push[..., None] * n + ridge[..., None] * n
        return p

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
            sx, sz = self.scales(h)
            phi = np.arctan2(p[..., 0] / sx, (self.zc - p[..., 2]) / sz)
            inside = (np.hypot(p[..., 0] / sx, (p[..., 2] - self.zc) / sz) < self.radius(h, phi)) & (h <= 1) & (h >= self.h_min)
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
        接平面上の高さ q_y をそのまま頭の表面の高さ h に対応させて歪みを防ぐ。
        横方向も、頭の軸から放射状に当てると目尻のように面が奥へ回り込む所で内へ縮み、目がしずく形にゆがむので、
        板の向き（水平成分）に沿ってまっすぐ押し当てる。"""
        p0 = self.point(d0)
        pdir = d0 if plane_dir is None else plane_dir
        nrm = norm(np.array([pdir[0], 0.0, pdir[2]]))
        e1 = norm(np.cross(nrm, UP))
        e2 = UP
        ab = np.asarray(ab, float)
        q = p0 + ab[..., :1] * e1 + ab[..., 1:2] * e2
        h_q = np.clip((q[..., 1] - self.y0) / self.H, self.h_min, 1.0)
        line = q
        for _ in range(6):
            phi_q = self.table_phi(h_q, line[..., 0], line[..., 2])
            p = self.surface(h_q, phi_q)
            # 面の点の、板の向きに沿った深さだけ線上を動かす（y は動かさない）
            t = (p - q) @ nrm
            line = q + t[..., None] * nrm
        phi_q = self.table_phi(h_q, line[..., 0], line[..., 2])
        p = self.surface(h_q, phi_q)
        out = norm(np.stack([p[..., 0], (p[..., 1] - self.c[1]) * 0.25, p[..., 2] - self.zc], -1))
        return p + out * (off * 0.085)

    def mesh(self, nh=160, nphi=192):
        hs = np.linspace(self.h_min, 1, nh)
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
        self._hilite = {}  # 瞳の白い光の中心（目の板の上の座標。調整ページが使う）
        model.tex_set = tex
        self.hair_color = hair_color
        # 顔の調整値（mmdgen/face.py。調整ページで人が決めてキャラごとの json に保存したもの）
        F = _face.get
        self.head = Head(eye_h=F("eye_height"), width=F("face_width"), jaw_x=F("cheek"))
        self.skin_color = skin_color
        orig_eye = self._orig_eye = "#" + "".join(f"{int(round(c * 255)):02x}" for c in eye_color)
        if F("eye_color"):
            eye_color = tuple(int(F("eye_color")[i:i + 2], 16) / 255 for i in (1, 3, 5))
        self.eye_color = np.array(eye_color, float)
        self.bust = bust
        # 目・鼻・口の位置は頭の形の表（美少女モデル）から決める：目は表の目のくぼみ（瞳の中心の向き EYE_YAW）、
        # 鼻先は目の線〜あご先の 0.35、口は 0.66 だけ目の線より下（立ち絵の比率）。高さはあご先からの高さ（頭の肌の高さ比）
        hd = self.head
        # 目の中心は美少女モデルの瞳の向きより少し外（既製モデルのずんだもん・ミクは目と目の間が目 1 つ分近く空く）
        p_eye = hd.surface(hd.eye_h, np.radians(_HS.EYE_YAW * F("eye_spread")))
        eye_yaw = float(np.arctan2(p_eye[0], hd.c[2] - p_eye[2]))
        self.eye = dict(yaw=eye_yaw, h=hd.eye_h, rx=0.0172, ry=0.0063, top=0.88, tilt=0.0, iris=0.70, iris_h=1.05, iris_y=-0.06, iris_in=0.05)
        self.eye.update(eye or {})
        E = self.eye
        E["rx"] *= F("eye_size")
        E["ry"] *= F("eye_size") * F("eye_aspect")
        E["tilt"] += F("eye_tilt")
        E["iris"] *= F("iris_w")
        E["iris_h"] *= F("iris_h")
        E["iris_y"] += F("iris_y")
        # 調整ページに今の値を見せるため、効いている値を残す
        _face.EFFECTIVE = dict(eye_color="#" + "".join(f"{int(round(c * 255)):02x}" for c in self.eye_color), orig_eye_color=orig_eye)
        self.eye["pitch"] = self.head.pitch_at(self.eye["h"], self.eye["yaw"])
        self.mouth = dict(rx=0.0080, ry=0.0010, smile=1.0, open_bottom=1.0, h=hd.eye_h * (1 - 0.66),
                          nose_h=hd.eye_h * (1 - 0.35), fang=False)
        self.mouth.update(mouth or {})
        self.mouth["pitch"] = self.head.pitch_at(self.mouth["h"])
        self.mouth["nose_pitch"] = self.head.pitch_at(self.mouth["nose_h"], 0.02)
        # 目は顔の曲面に沿わせず、少し外を向いた平らな板にして、顔に彫ったくぼみにはめる（既製モデルのずんだもんと同じ作り）。
        # 斜めから見ても目は平らな楕円として縮むだけで、頬の丸みに巻きつかない
        for _, _, s in SIDES:
            p0, n, e1 = self.eye_frame(s)
            hd.add_socket(p0, n, e1, self.eye["rx"] * 1.04, self.eye["ry"] * 1.04, EYE_SOCKET_DEPTH, lid=EYE_LID)
        m = model
        m.default_toon_tex = tex.toon("toon_cloth.png", (0.82, 0.80, 0.89))
        # 肌の影は灰色に濁らせず、赤みのある暖色にする
        skin_shadow = np.clip(np.asarray(skin_color, float) * np.array([0.97, 0.80, 0.77]), 0, 1)
        m.material("skin", "肌", skin_color, toon_tex=tex.toon("toon_skin.png", (1.0, 0.85, 0.80)), shadow_color=skin_shadow)
        m.material("ear_in", "耳の内側", np.asarray(skin_color) * np.array([0.88, 0.72, 0.70]),
                   toon_tex=tex.toon("toon_skin.png", (1.0, 0.85, 0.80)), edge=False)
        self._rig()
        self._chains = 0
        self._head_info = None
        physics().body_rigids(self)

    # ---- ボーン ----
    def eye_frame(self, s):
        """目の板の中心・向き（外向きの法線）・横向き（目尻の方）。板は顔の面の目の中心から EYE_INSET だけ奥に置く。"""
        E = self.eye
        d0 = Head.dir(s * E["yaw"], E["pitch"])
        pdir = Head.dir(s * E["yaw"] * EYE_TURN, E["pitch"] * 0.75)
        n = norm(np.array([pdir[0], 0.0, pdir[2]]))
        e1 = norm(np.cross(n, UP))
        p0 = self.head.point(d0)
        # 目の中心の肌は表のくぼみで奥にあるので、板の高さは目のまわり（輪郭の少し外）の肌の平均にそろえる。
        # 中心の肌に合わせると目全体が穴の底に沈んで見える
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = p0 + 1.15 * (np.cos(t)[:, None] * E["rx"] * e1 + np.sin(t)[:, None] * E["ry"] * UP)
        rim = self.head.point(ring - self.head.c)
        p0 = p0 + n * (float(np.mean((rim - p0) @ n)) - EYE_INSET)
        return p0, n, e1

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

        geo_n = _merge_seams(hmesh[0], vertex_normals(*hmesh))
        geo_lut = {tuple(np.round(p, 7)): n for p, n in zip(hmesh[0], geo_n)}

        def head_normals(v):
            # 法線は形どおり（美少女モデルと同じく目のくぼみ・鼻筋・口元の凹凸が陰影に出る）にし、
            # 楕円体の滑らかな法線を 3 割だけ混ぜて、トゥーンの影が細かく割れないようにする
            base_n = norm((v - h.c) / h.r)
            gn = np.array([geo_lut[tuple(np.round(p, 7))] for p in v])
            return norm(gn * 0.7 + base_n * 0.3)

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
        # 首：頭の表の首の付け根（頭の肌のいちばん下の輪）から、胸の中の細い輪までをなめらかにつなぐ。
        # 上の輪は頭の肌の下の輪と同じ点なので、頭を動かしても継ぎ目が開かない（上の端は頭に 100% 付ける）
        nphi = 192
        phis = np.arange(nphi) / nphi * TAU
        top = h.surface(h.h_min, phis)
        y_top, y_bot = float(top[0, 1]), 0.780
        r_bot, z_bot = 0.0170, 0.004
        bot = np.stack([r_bot * np.sin(phis), np.full(nphi, y_bot), z_bot - r_bot * np.cos(phis)], 1)
        rings = []
        for t in np.linspace(0, 1, 10):
            k = smoothstep(0, 1, t)
            rings.append(bot * (1 - k) + top * k)
        upper = m.skin_vertical("頭", "首", y_top, NECK_Y)
        lower = m.skin_auto(["上半身2", "首"])

        def neck_skin(p):
            a, b = upper(p), lower(p)
            return [a[i] if p[i, 1] >= NECK_Y else b[i] for i in range(len(p))]
        m.add(neck_mat, loft(rings, closed=True), neck_skin)

        # 胴体（胸・谷間・くびれ・へそを一体成形した滑らかな高密度メッシュ：楕円体の継ぎ目溝や2重メッシュの突き抜けが一切出ない）
        chest_ys = np.linspace(0.660, 0.780, 36)
        navel_ys = np.linspace(0.594, 0.618, 9)
        shoulder_ys = np.linspace(0.760, 0.840, 41)  # 首の付け根〜肩先の丸みを細かく刻む
        extra_keys = list(chest_ys) + list(navel_ys) + list(shoulder_ys) + [b[0] for b in torso_bands]
        ys = _samples([t[0] for t in TORSO], extra_keys, 60)
        n_ang = 160
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
            ys = _samples([t[0] for t in LEG], [b[0] for b in leg_bands], 90)
            rings, centers = [], []
            for y in ys:
                r, x, z = (np.interp(-y, [-t[0] for t in LEG], [t[k] for t in LEG]) for k in (1, 2, 3))
                c = np.array([s * x, y, z])
                pts = ring(c, [1, 0, 0], [0, 0, 1], r, r * 1.02, 64)
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
            ts = _samples([a[0] for a in ARM], [b[0] for b in arm_bands], 60, descending=False)
            rings, centers = [], []
            for t in ts:
                r = np.interp(t, _ARM_T, _ARM_R)
                c = j["腕"] + d * t
                rings.append(ring(c, u, v, r, r, 48))
                centers.append(c)
            self._banded(rings, centers, ts, arm_bands, self.skin_arm(jp), cap0=True, cap1=True, descending=False)
            sh_mat = _band_at(arm_bands, 0.0, descending=False)
            # 肩の丸み（三角筋）：腕の向きに長い楕円体にして、玉ではなく腕へ流れる形にする
            m.add(sh_mat, ellipsoid(j["腕"] + d * 0.004 + np.array([-s * 0.004, 0.001, 0]), [0.024, 0.019, 0.020],
                                    rot_axes(d, v, np.cross(d, v)), nu=48, nv=28),
                  m.skin_auto(["上半身2", jp + "肩", jp + "腕"], {"上半身2": 1.2}))
            self.add_hand(jp, s, hand_mat)

    def add_hand(self, jp, s, mat):
        """立体的な 5 本指の手：手首から段差なくつながる掌（母指球のふくらみ、甲の指の付け根の小さな山）と、
        指ごとに向きを持った断面の 5 本の指。素手（肌）なら指先に爪を付ける。"""
        m = self.m
        j = joints(s)
        d, up, bk = hand_axes(s)
        w = j["手首"]

        # 1) 掌（腕の先の中 u=-0.012 から指の付け根 u=0.032 まで）。腕の先（半径 0.0092）と同じ太さで始めて段を出さない
        palm_slices = np.array([
            # (u, v_center, z_center, rv, rz_thumb, rz_pinky, thumb_bulge)
            (-0.012,  0.0000,  0.0000, 0.0084, 0.0086, 0.0086, 0.0000),
            (-0.003,  0.0000,  0.0000, 0.0091, 0.0096, 0.0096, 0.0000),
            ( 0.002, -0.0004,  0.0002, 0.0093, 0.0114, 0.0116, 0.0012),
            ( 0.008, -0.0012,  0.0006, 0.0086, 0.0140, 0.0136, 0.0032),
            ( 0.015, -0.0014,  0.0015, 0.0075, 0.0148, 0.0152, 0.0028),
            ( 0.023, -0.0012,  0.0022, 0.0066, 0.0143, 0.0164, 0.0010),
            ( 0.029, -0.0011,  0.0024, 0.0057, 0.0137, 0.0168, 0.0000),
            ( 0.0325, -0.0011, 0.0024, 0.0043, 0.0127, 0.0156, 0.0000),
        ])
        us = np.concatenate([np.linspace(-0.012, 0.029, 22), [0.031, 0.0325]])
        cols = [np.interp(us, palm_slices[:, 0], palm_slices[:, k]) for k in range(1, 7)]
        n_ring = 40
        ang = np.arange(n_ring) / n_ring * TAU
        ca, sa = np.cos(ang), np.sin(ang)  # ca: bk (+Z 小指 / -Z 親指), sa: up (+甲 / -掌)
        knuckle_z = [pts[0][2] for fn, _, _, pts, _ in FINGER_SPECS if fn != "親指"]
        p_rings, p_centers = [], []
        for u, vc, zc, rv, rzt, rzp, tbulge in zip(us, *cols):
            rz = np.where(ca < 0, rzt, rzp)
            # 掌側（sa < 0）かつ親指側（ca < 0）に母指球の柔らかなふくらみを足し、小指側は少し手前に引く
            bulge_v = -tbulge * np.clip(-ca, 0, 1) ** 1.4 * np.clip(-sa, 0, 1)
            arch_u = -0.0035 * (max(u, 0) / 0.032) * np.clip(ca, 0, 1) ** 1.6
            # 甲側：指の付け根の関節の小さな山と、その間の谷
            zz = zc + ca * rz
            kn = sum(np.exp(-((zz - kz) / 0.0030) ** 2) for kz in knuckle_z)
            knuckle = 0.0009 * (kn - 0.45) * np.clip(sa, 0, 1) * np.exp(-((u - 0.0285) / 0.0050) ** 2)
            c = w + d * u + up * vc + bk * zc
            pts = (c + np.outer(arch_u, d)
                   + np.outer(sa * rv + bulge_v + knuckle, up)
                   + np.outer(ca * rz, bk))
            p_rings.append(pts)
            p_centers.append(c)
        palm_bones = [jp + "ひじ", jp + "手首", jp + "親指０", jp + "人指１", jp + "中指１", jp + "薬指１", jp + "小指１"]
        palm_skin = self.smooth_joints(
            m.skin_auto(palm_bones, {jp + "ひじ": 1.35, jp + "親指０": 1.25}),
            [(jp + "ひじ", jp + "手首", 0.018)],
        )
        m.add(mat, loft(p_rings, cap0=True, cap1=True, centers=p_centers), palm_skin)

        # 2) 5 本の指。断面は指ごとの向き（指の背 n・横 sd）で作る。親指は爪が外上を向くよう 55° ほどねじる
        nf = 18
        fa = np.arange(nf) / nf * TAU
        fca, fsa = np.cos(fa), np.sin(fa)
        bare = mat == "skin"
        if bare and "nail" not in m._mat:
            nail_c = np.clip(_mix_c(self.skin_color, (1.0, 0.70, 0.72), 0.55), 0, 1)
            m.material("nail", "爪", nail_c, edge=False)
            m._mat["nail"].specular, m._mat["nail"].power = (0.22, 0.20, 0.20), 40.0
        for fn, _, nums, pts_spec, rads_spec in FINGER_SPECS:
            pts = [w + d * u + up * v + bk * z for u, v, z in pts_spec]
            f_dir = norm(pts[1] - pts[0])
            root_pt = pts[0] - f_dir * (0.0045 if fn == "親指" else 0.0055)
            chain_pts = [root_pt] + pts
            chain_rads = [rads_spec[0]] + list(rads_spec)
            # 指のパスを細かく補間する（関節でわずかに太らせる）
            sub_pts, sub_rv, sub_rz = [], [], []
            segs = len(chain_pts) - 1
            for k in range(segs):
                steps = 6
                for s_i in range(steps):
                    t = s_i / steps
                    sub_pts.append(chain_pts[k] * (1 - t) + chain_pts[k + 1] * t)
                    rv0, rz0 = chain_rads[k]
                    rv1, rz1 = chain_rads[k + 1]
                    knuckle = 1.0 + (0.05 if s_i == 0 and k in (1, 2) else 0.0)
                    sub_rv.append((rv0 * (1 - t) + rv1 * t) * knuckle)
                    sub_rz.append((rz0 * (1 - t) + rz1 * t) * knuckle)
            n_body = len(sub_pts)
            # 指先は丸いドーム
            tip_dir = norm(chain_pts[-1] - chain_pts[-2])
            rv_tip, rz_tip = chain_rads[-1]
            for tip_t in (0.0, 0.30, 0.55, 0.76, 0.91, 0.99):
                sub_pts.append(chain_pts[-1] + tip_dir * (rv_tip * 1.25 * tip_t))
                sc = np.sqrt(max(1.0 - tip_t ** 2, 0.0))
                sub_rv.append(rv_tip * max(sc, 0.10))
                sub_rz.append(rz_tip * max(sc, 0.10))
            sub_pts = np.array(sub_pts)
            # 関節の折れを少しならす（指が節ごとにかくかく曲がって見えないように）
            sm = sub_pts.copy()
            sm[1:-1] = 0.25 * sub_pts[:-2] + 0.5 * sub_pts[1:-1] + 0.25 * sub_pts[2:]
            sub_pts = sm
            tg = norm(np.gradient(sub_pts, axis=0))
            n0, sd0 = (norm(0.57 * up - 0.82 * bk), norm(0.82 * up + 0.57 * bk)) if fn == "親指" else (up, bk)
            nn = norm(n0 - (tg @ n0)[:, None] * tg)
            sd = sd0 - (tg @ sd0)[:, None] * tg
            sd = norm(sd - np.einsum("ij,ij->i", sd, nn)[:, None] * nn)

            f_rings = []
            for p_i, rv_i, rz_i, n_i, s_i in zip(sub_pts, sub_rv, sub_rz, nn, sd):
                # 指の腹側（fsa < 0）を少しふっくらさせ、甲側をすっきりさせる
                pad = 1.0 + 0.10 * np.clip(-fsa, 0, 1)
                f_rings.append(p_i + np.outer(fsa * rv_i * pad, n_i) + np.outer(fca * rz_i, s_i))

            fbones = [jp + "手首"] + [jp + fn + ch for ch in nums]
            fpairs = [(fbones[k], fbones[k + 1], 0.005) for k in range(len(fbones) - 1)]
            fskin = self.smooth_joints(m.skin_auto(fbones, {jp + "手首": 1.35}), fpairs)
            m.add(mat, loft(f_rings, cap0=True, cap1=True, centers=list(sub_pts)), fskin)

            if bare:
                # 爪：いちばん先の節の背に沿う、丸みのある小さな板。根元と先を丸め、肌からわずかに浮かせる
                i0, i1 = n_body - 4, len(sub_pts) - 3
                rows = []
                for q in np.linspace(0, 1, 9):
                    x = i0 + (i1 - i0) * q
                    k0 = int(np.floor(x))
                    k1 = min(k0 + 1, len(sub_pts) - 1)
                    f = x - k0
                    pc = sub_pts[k0] * (1 - f) + sub_pts[k1] * f
                    rv_q = sub_rv[k0] * (1 - f) + sub_rv[k1] * f
                    rz_q = sub_rz[k0] * (1 - f) + sub_rz[k1] * f
                    n_q, s_q = norm(nn[k0] * (1 - f) + nn[k1] * f), norm(sd[k0] * (1 - f) + sd[k1] * f)
                    half = np.radians(58) * max(np.sqrt(max(1 - (2 * q - 1) ** 2, 0.0)) ** 0.55, 0.22)
                    ph = np.linspace(-half, half, 9)
                    rows.append(pc + np.outer(np.cos(ph) * (rv_q + 0.00035), n_q) + np.outer(np.sin(ph) * (rz_q + 0.00035), s_q))
                cen = [sub_pts[int(round(i0 + (i1 - i0) * q))] for q in np.linspace(0, 1, 9)]
                m.add("nail", loft(rows, closed=False, centers=cen), fskin)

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
        """目は白目・瞳・ハイライト・下まぶた・上まつ毛・二重の 6 層。口はリップライン・唇の影・口内奥・舌・歯の 5 層。
        部品を分けてあるので、表情ごとに別々に動かせる（じと目は上まつ毛だけ下げる、ハイライト消しは光だけ消す、など）。"""
        m = self.m
        h = self.head
        E = self.eye
        M = self.mouth
        T = self.tex
        F = _face.get
        is_closed_mouth = M["ry"] < 0.0022
        # 目や頬は陰影で灰色にならないよう、真っ白なトゥーンと環境色 1 で描いた色そのままにする
        noedge = dict(edge=False, shadow=False, shade=1.0, toon_tex=T.toon("toon_flat.png", (1, 1, 1)))
        # 3 層の板の範囲（A は rx 単位で目頭 - → 目尻 +、B は ry 単位で上 +）。テクスチャは同じ輪郭をこの範囲に描く
        k = E["rx"] / E["ry"]
        ext = {"eye_white": (-1.0, 1.0, -1.0, 1.0),
               # 瞳は幅の 1.05 倍の縦長で、目の高さいっぱい。上はまつ毛、下は下の縁で少し切られる
                              # 瞳の幅は目の幅の iris 倍、高さは目の高さの iris_h 倍。既製の萌えキャラと同じく目の高さいっぱいの縦長にし、上はまつ毛の下に入り、下は下まぶたに触れる
               "iris": (-E["iris_in"] - E["iris"], -E["iris_in"] + E["iris"],
                        E["iris_y"] - E["iris_h"], E["iris_y"] + E["iris_h"]),
               "lash": (-1.14, 1.45, -1.0, 1.85)}
        m.material("eye_white", "白目", (1, 1, 1), tex=T.eye_white(ext["eye_white"], k), **noedge)
        m.material("iris", "瞳", self.eye_color, tex=T.iris("iris.png", self.eye_color, ext["iris"], k), **noedge)
        # まつ毛は真っ黒にせず、瞳の色を混ぜた暗い紫寄りの色（既製の萌えキャラと同じ。黒と白目の強い差は怖く見える）
        lash_c = np.clip(_mix_c(self.eye_color * 0.45, (0.30, 0.20, 0.34), 0.6) * F("lash_light"), 0, 1)
        m.material("hilite", "ハイライト", (1, 1, 1), tex=T.hilite(), **noedge)
        m.material("lowlash", "下まつ毛", lash_c, tex=T.lash(ext["lash"], k, "lash_lower.png", color=lash_c, part="lower"), **noedge)
        m.material("lash", "まつ毛", lash_c, tex=T.lash(ext["lash"], k, color=lash_c), **noedge)
        m.material("eyelid", "二重", (0.36, 0.22, 0.20), tex=T.lash(ext["lash"], k, "eyelid.png", color=lash_c, part="crease"), **noedge)
        # 眉は髪色そのままでなく、赤みのある焦げ茶に寄せる（既製モデルの眉の色）
        bc = brow_color if brow_color is not None else _mix_c(np.asarray(self.hair_color) * 0.60, (0.40, 0.20, 0.20), 0.5)
        m.material("brow", "眉", bc, tex=T.brow("brow.png", bc), **noedge)
        if is_closed_mouth:
            m.material("mouth", "口", closed_mouth_color, tex=T.mouth_line("mouth_line.png", closed_mouth_color), **noedge)
        else:
            m.material("mouth", "口", (0.28, 0.10, 0.11), **noedge)
        m.material("lip", "唇", self.skin_color, tex=T.lip("lip.png", self.skin_color), **noedge)
        m.material("mouth_in", "口内", (1, 1, 1), tex=T.mouth_interior("mouth_in.png"), **noedge)
        m.material("tongue", "舌", (1, 1, 1), tex=T.mouth_interior("tongue.png", part="tongue"), **noedge)
        m.material("teeth", "歯", (1, 1, 1), tex=T.mouth_interior("teeth.png", fang=M.get("fang", False), part="teeth"), **noedge)
        m.material("nose", "鼻", self.skin_color, tex=T.nose("nose.png", self.skin_color), **noedge)
        if blush:
            m.material("blush", "頬", (1.0, 0.5, 0.5), tex=T.blush(), **noedge)
        m.material("hatch", "照れ", (1, 1, 1), tex=T.hatch(), **noedge)
        for name, en, panel in [("まばたき", "blink", 2), ("笑い", "smile", 2), ("ウィンク", "wink", 2),
                                ("ウィンク右", "wink R", 2), ("じと目", "jito", 2), ("びっくり", "surprised", 2),
                                ("瞳小", "pupil small", 2), ("ハイライト消し", "no highlight", 2),
                                ("あ", "a", 3), ("い", "i", 3), ("う", "u", 3),
                                ("え", "e", 3), ("お", "o", 3), ("口角上げ", "mouth up", 3), ("口角下げ", "mouth down", 3),
                                ("にこり", "cheerful", 1), ("怒り", "angry", 1), ("困る", "sad", 1),
                                ("上", "brow up", 1), ("下", "brow down", 1), ("照れ", "blush", 4)]:
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

            # 瞳の白い光：見る人から見て左上（-X 側）に大きな楕円、反対の下に小さな丸。両目とも同じ側に置く
            hs = F("highlight")
            ic_a, ic_b = -s * E["iris_in"] * rx, E["iris_y"] * ry  # 瞳の中心
            iw, ih = E["iris"] * rx, E["iris_h"] * ry              # 瞳の半径
            hl_centers = [(ic_a - 0.27 * iw, ic_b + 0.42 * ih), (ic_a + 0.30 * iw, ic_b - 0.44 * ih)]
            hl_sizes = [(0.21 * iw * hs, 0.27 * ih * hs), (0.085 * iw * hs, 0.095 * ih * hs)]
            self._hilite[jp] = [(a * s, b) for a, b in hl_centers]
            layers = [
                ("eye_white", layer("eye_white", 13, 11), 0.0025, "頭"),
                ("iris", layer("iris", 11, 11), 0.0048, jp + "目"),
            ] + [("hilite", plane(wa, -wb, wb, 1.0, cx=ca_, cy=cb_, cols=5, rows=5), 0.0058, jp + "目")
                 for (ca_, cb_), (wa, wb) in zip(hl_centers, hl_sizes) if hs > 0.01] + [
                ("lowlash", layer("lash", 17, 15), 0.0066, "頭"),
                ("lash", layer("lash", 17, 15), 0.0072, "頭"),
                ("eyelid", layer("lash", 17, 15), 0.0074, "頭"),
            ]
            upper = ("lash", "eyelid", "lowlash")  # まばたきで閉じた線に重なる部品
            pupil = ("iris", "hilite")

            def jito(mat):
                """じと目：上まつ毛と二重だけを半分下ろし、白目と瞳は上を押しつぶしてまつ毛の下に隠す。下まぶたは動かさない。"""
                def f(r):
                    b = r[:, 1]
                    if mat in ("lash", "eyelid"):
                        r[:, 1] = b - 0.50 * ry * smoothstep(-0.8 * ry, 0.8 * ry, b)
                    elif mat != "lowlash":
                        c = 0.36 * ry
                        r[:, 1] = np.where(b > c, c + (b - c) * 0.16, b)
                    return r
                return f

            def wide(mat):
                """びっくり：目を縦に開き、瞳は小さく。"""
                def f(r):
                    r[:, 1] = r[:, 1] * 1.12
                    if mat in pupil:
                        r[:, 0] = ic_a + (r[:, 0] - ic_a) * 0.84
                        r[:, 1] = ic_b + (r[:, 1] - ic_b) * 0.76
                    return r
                return f

            def small(r):
                r[:, 0] = ic_a + (r[:, 0] - ic_a) * 0.72
                r[:, 1] = ic_b + (r[:, 1] - ic_b) * 0.72
                return r
            p0, n_eye, e1_eye = self.eye_frame(s)

            def plate(rows, off):
                """目の板（顔の曲面に貼らない）。真ん中だけ眼球のようにゆるくふくらませる。"""
                def dome(r):
                    q = np.clip(1 - (r[:, 0] / rx) ** 2 - (r[:, 1] / ry) ** 2, 0, None)
                    return (EYE_DOME * q + off * 0.085)[:, None]
                rings = [p0 + r[:, :1] * e1_eye + r[:, 1:2] * UP + n_eye * dome(r) for r in rows]
                return _flat_loft(rings, n_eye, False)

            for mat, rows0, off, bone in layers:
                rows = tilted(rows0)
                shut = dict(keep=0.2, pivot=1.0 * ry) if mat in upper else {}
                base = m.add(mat, plate(rows, off), m.skin_fixed(bone))
                blink = plate(remap_ab(rows, closer(-0.20, **shut)), off)[0]
                m.add_morph_part("まばたき", base, blink)
                m.add_morph_part("笑い", base, plate(remap_ab(rows, closer(0.28, **shut)), off)[0])
                m.add_morph_part("ウィンク" if s > 0 else "ウィンク右", base, blink)
                # 傾きを掛ける前の形を変えてから傾ける（部品どうしがずれない）
                m.add_morph_part("じと目", base, plate(tilted(remap_ab(rows0, jito(mat))), off)[0])
                m.add_morph_part("びっくり", base, plate(tilted(remap_ab(rows0, wide(mat))), off)[0])
                if mat in pupil:
                    m.add_morph_part("瞳小", base, plate(tilted(remap_ab(rows0, small)), off)[0])
                if mat == "hilite":
                    ctr = np.mean(np.vstack(rows0), axis=0)
                    m.add_morph_part("ハイライト消し", base, plate(tilted(remap_ab(rows0, lambda r, c=ctr: c + (r - c) * 0.0)), off)[0])
            # 眉は目の上に少し離して（美少女モデルの眉の高さ）、目頭の上から目尻の少し手前まで長くかける
            bd = Head.dir(s * E["yaw"], E["pitch"] + 0.42)
            bpdir = Head.dir(s * E["yaw"] * 0.38, E["pitch"] + 0.36)
            brow_rows = plane(rx * 0.92, -0.0040, 0.0040, s, cx=-s * 0.02 * rx)
            b_base = m.add("brow", h.decal(bd, brow_rows, 0.0085, closed=False, plane_dir=bpdir), m.skin_fixed("頭"))
            # 眉の表情モーフ（にこり・怒り・困る・まばたき・笑い）
            m.add_morph_part("笑い", b_base, h.decal(bd, remap_ab(brow_rows, lambda r: r + [0, -0.0025]), 0.0085, closed=False, plane_dir=bpdir)[0])
            m.add_morph_part("にこり", b_base, h.decal(bd, remap_ab(brow_rows, lambda r: np.stack([r[:, 0], r[:, 1] - 0.002 - 0.12 * (r[:, 0] / 0.023) ** 2 * 0.005], 1)), 0.0085, closed=False, plane_dir=bpdir)[0])
            m.add_morph_part("怒り", b_base, h.decal(bd, remap_ab(brow_rows, lambda r, s=s: np.stack([r[:, 0], r[:, 1] + 0.22 * r[:, 0] * s - 0.002], 1)), 0.0085, closed=False, plane_dir=bpdir)[0])
            m.add_morph_part("困る", b_base, h.decal(bd, remap_ab(brow_rows, lambda r, s=s: np.stack([r[:, 0], r[:, 1] - 0.22 * r[:, 0] * s + 0.002], 1)), 0.0085, closed=False, plane_dir=bpdir)[0])
            for nm, dy in (("上", 0.0035), ("下", -0.0030), ("びっくり", 0.0030), ("じと目", -0.0018)):
                m.add_morph_part(nm, b_base, h.decal(bd, remap_ab(brow_rows, lambda r, dy=dy: r + [0, dy]), 0.0085, closed=False, plane_dir=bpdir)[0])
            # 照れの斜線：頬の上
            hh = E["h"] - 0.105
            hd_ = Head.dir(s * 0.50, self.head.pitch_at(hh, 0.50))
            hp_ = Head.dir(s * 0.30, self.head.pitch_at(hh, 0.30))
            hatch_rows = plane(0.0125, -0.0048, 0.0048, s, cols=9, rows=5)
            # ふだんは肌の内側に沈めて隠し、モーフで表へ出す
            hb = m.add("hatch", h.decal(hd_, hatch_rows, -0.0060, closed=False, plane_dir=hp_), m.skin_fixed("頭"))
            m.add_morph_part("照れ", hb, h.decal(hd_, hatch_rows, 0.0030, closed=False, plane_dir=hp_)[0])
            if blush:
                # 目の下の縁から口の高さまで、頬に広くふんわり置く
                bh = E["h"] - 0.12
                cd = Head.dir(s * 0.52, self.head.pitch_at(bh, 0.52))
                cpdir = Head.dir(s * 0.30, self.head.pitch_at(bh, 0.30))
                m.add("blush", h.decal(cd, plane(0.0180, -0.0085, 0.0085, s, cols=25, rows=19), 0.0020, closed=False, plane_dir=cpdir),
                      m.skin_fixed("頭"))

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
            in_off = 0.0010
        else:
            # 普段から開いた元気な口（キリコ・フェリス）：外周のリップ縁 ("mouth") と内側の歯・舌 ("mouth_in")
            lip_base_ab = ellipse_ab(0, 0, M["rx"], M["ry"] * 1.06, top=0.45, bottom=M["open_bottom"], curve=M["smile"] * 0.35)
            in_base_ab = mouth_grid_ab(M["rx"] * 0.93, M["ry"], M["smile"], top=0.45, bottom=M["open_bottom"])
            base_lip = m.add("mouth", h.decal(md, lip_base_ab, 0.0028), m.skin_fixed("頭"))
            in_off = 0.0048
        # 口の中は同じ形の板を 3 枚重ねる：奥、舌、歯（後ろから順に）
        in_layers = [(mat, m.add(mat, h.decal(md, in_base_ab, in_off + dz, closed=False), m.skin_fixed("頭")), dz)
                     for mat, dz in (("mouth_in", 0.0), ("tongue", 0.0003), ("teeth", 0.0006))]
        # 唇の影：口のすぐ下の肌に置く。口が開くとあごと一緒に下がる
        lip_y = -(0.0042 if is_closed_mouth else M["ry"] * M["open_bottom"] + 0.0030)
        lip_ab = plane(M["rx"] * 0.80, -0.0034, 0.0034, 1.0, cy=lip_y, cols=13, rows=5)
        base_shadow = m.add("lip", h.decal(md, lip_ab, 0.0016, closed=False), m.skin_fixed("頭"))

        # 口角の上げ下げ：口の端ほど大きく上下させる
        def corner(dy):
            return lambda r: np.stack([r[:, 0], r[:, 1] + dy * np.clip(np.abs(r[:, 0]) / max(M["rx"], 1e-6), 0, 1.3) ** 2], 1)
        for nm, dy in (("口角上げ", 0.0030), ("口角下げ", -0.0030)):
            lip_off = 0.0075 if is_closed_mouth else 0.0028
            m.add_morph_part(nm, base_lip, h.decal(md, remap_ab(lip_base_ab, corner(dy)), lip_off, closed=is_closed_mouth is False)[0])
            for mat, bi, dz in in_layers:
                m.add_morph_part(nm, bi, h.decal(md, remap_ab(in_base_ab, corner(dy)), in_off + dz, closed=False)[0])

        self._write_face_meta(ext, lash_c)

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
                for mat, bi, dz in in_layers:
                    m.add_morph_part(name, bi, h.decal(md, in_v_ab, 0.0042 + dz, closed=False)[0])
            else:
                lip_v_ab = ellipse_ab(0, 0, rx_v, ry_v * 1.06, top=0.45, bottom=1.0, curve=sm * 0.35)
                in_v_ab = mouth_grid_ab(rx_v * 0.93, ry_v, sm, top=0.45, bottom=1.0)
                m.add_morph_part(name, base_lip, h.decal(md, lip_v_ab, 0.0028)[0])
                for mat, bi, dz in in_layers:
                    m.add_morph_part(name, bi, h.decal(md, in_v_ab, 0.0048 + dz, closed=False)[0])
            # 唇の影は下唇について下がる
            drop = ry_v - (0.0015 if is_closed_mouth else M["ry"] * M["open_bottom"])
            m.add_morph_part(name, base_shadow, h.decal(md, remap_ab(lip_ab, lambda r, d=float(np.clip(drop, 0.0, 0.0035)): r - [0, d]), 0.0016, closed=False)[0])

            # 口を大きく開けるモーフでは、あご先の肌も自然に少し下がるようにする
            if self._head_info is not None and jaw_drop > 0:
                hv = self._head_mesh[0].copy()
                h_rel = (hv[:, 1] - h.y0) / h.H
                front_mask = np.clip((-hv[:, 2] - 0.015) / 0.035, 0.0, 1.0)
                chin_mask = np.clip((0.20 - h_rel) / 0.20, 0.0, 1.0) ** 1.3 * np.exp(-(hv[:, 0] / 0.032) ** 2)
                # 下げるのは開いた口の下端より下だけ。口の高さの肌まで下げると、あごへ回り込む面が前へ出て口の中にかぶさる
                mouth_bottom = h.point(md)[1] - ry_v
                below = np.clip((mouth_bottom - 0.0015 - hv[:, 1]) / 0.004, 0.0, 1.0)
                hv[:, 1] -= jaw_drop * front_mask * chin_mask * below
                m.add_morph_part(name, self._head_info, hv)

    def _write_face_meta(self, ext, lash_c):
        """顔の寸法を models/<キー>/face.json に書く。調整ページはこれを使い、作り直さずにブラウザでその場で目や顔を動かす。
        座標は PMX と同じ（MMD 単位、顔は -Z 向き）。"""
        import json
        import os
        s, h, E = self.m.scale, self.head, self.eye
        sides = {}
        for _, _, sg in SIDES:
            p0, n, e1 = self.eye_frame(sg)
            jp = "左" if sg > 0 else "右"
            sides["L" if sg > 0 else "R"] = dict(p0=(p0 * s).tolist(), n=n.tolist(), e1=e1.tolist(), sign=sg,
                                                 hilite=[[a * s, b * s] for a, b in self._hilite.get(jp, [])])
        meta = dict(
            scale=s,
            values={k: _face.get(k) for k in _face.DEFAULTS},
            spec=[dict(name=n, label=l, min=a, max=b, step=st, default=d, help=hp) for n, l, a, b, st, d, hp in _face.SPEC],
            head=dict(y0=h.y0 * s, H=h.H * s, eye_h=h.eye_h, zc=h.zc * s, width=h.width, jaw_x=h.jaw_x),
            eye={k: (E[k] * s if k in ("rx", "ry") else E[k]) for k in ("rx", "ry", "tilt", "iris", "iris_h", "iris_y", "iris_in")},
            ext={k: list(v) for k, v in ext.items()},
            outline=eye_outline().tolist(),
            eye_color="#" + "".join(f"{int(round(c * 255)):02x}" for c in self.eye_color),
            orig_eye_color=self._orig_eye,
            lash_color=[float(c) for c in lash_c],
            sides=sides,
        )
        out = os.path.dirname(self.tex.dir)
        with open(os.path.join(out, "face.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    # ---- 髪 ----
    def hair_material(self, key, name, color, tip=None, flecks=None):
        """筋の入った髪テクスチャと、上寄りに光の輪が出るスフィアを付けた材質。flecks に色を渡すと、その色の筋が混じる。"""
        T = self.tex
        return self.m.material(key, name, color, tex=T.hair(f"hair_{key}.png", color, tip, flecks=flecks),
                               sphere=T.hair_sphere(), sphere_mode=2, bake="v", edge_size=0.8)

    def hair_surface(self, scale=(1.06, 1.04, 1.07), lift=(0, 0.002, 0.006)):
        return self.head.c + np.array(lift), self.head.r * np.array(scale)

    def hair_cap(self, mat, keep=None, scale=(1.06, 1.04, 1.07)):
        c, r = self.hair_surface(scale)
        if keep is None:
            def keep(d):
                face = (d[:, 2] < -0.22) & (np.abs(d[:, 0]) < 0.78) & (d[:, 1] < 0.48)
                return ~face & (d[:, 1] > -0.40)
        self.m.add(mat, ellipsoid(c, r, nu=36, nv=22, keep=keep), self.m.skin_fixed("頭"))

    def strand(self, path, widths, thick, sides, mats, blunt=False, skin=None, wrap=None):
        """髪の束。頭頂〜額はしっかり厚みを保って肌を覆い、毛先近くで薄く絞る。
        wrap に頭の中心を渡すと、束の幅を平らな板にせず、その縦軸のまわりに巻いて頭や首の丸みに沿わせる
        （幅の広い束を板のまま並べると、頭のまわりが角ばって段になる）。"""
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
        seg = 12 if wrap is None else 24
        a = np.arange(seg) / seg * TAU
        ca, sa = np.cos(a), np.sin(a)
        if wrap is None:
            rings = [path[i] + np.outer(ca * widths[i], sides[i])
                     + np.outer(sa * thick_arr[i] - 1.3 * thick_arr[i] * (1 - ca ** 2), nn[i]) for i in range(n)]
        else:
            rings = []
            for i in range(n):
                dx, dz = path[i][0] - wrap[0], path[i][2] - wrap[2]
                rho = max(np.hypot(dx, dz), 1e-6)
                th = np.arctan2(dx, -dz)
                sgn = 1.0 if sides[i] @ np.array([np.cos(th), 0.0, np.sin(th)]) >= 0 else -1.0
                q = sa * thick_arr[i] - 1.3 * thick_arr[i] * (1 - ca ** 2)  # 内向きの厚み
                ang = th + sgn * np.clip(ca * widths[i] / rho, -0.75, 0.75)
                rad = rho + q * (nn[i] @ np.array([np.sin(th), 0.0, -np.cos(th)]))
                rings.append(np.stack([wrap[0] + rad * np.sin(ang), path[i][1] + q * nn[i][1],
                                       wrap[2] - rad * np.cos(ang)], 1))
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
              scale=(1.06, 1.04, 1.07), mats_fn=None, clearance=(0.100, 0.090, 0.074), wave=0.0, bones=0, splay=0.0,
              tuck=0.0):
        """頭から垂れ下がる髪の束。体（肩・胸）をよけて外側に広がる。bones を渡すと物理で揺れる。
        splay はあごの高さから下へ行くほど外へ開く量（姫毛を顔から離して広げる）。
        tuck は頭の真ん中の高さから毛先へ向かって内へ丸める割合（ボブの毛先を首の方へ入れる）。
        短い髪で肩をよける必要がないときは clearance=(0, 0, 0) にする（頭の一番広い所からまっすぐ下ろす）。"""
        c, r = self.hair_surface(scale)
        h = np.array([np.sin(theta), 0.0, -np.cos(theta)])
        pts = []
        # 頭の面に沿って e0 の高さから頭の真ん中の高さまで下りる（e0 が真ん中より下なら、そこから下へだけ下ろす）
        for phi in np.linspace(e0, 0.0, 8, endpoint=False) if e0 > 0 else [e0]:
            d = np.array([np.sin(theta) * np.cos(phi), np.sin(phi), -np.cos(theta) * np.cos(phi)])
            pts.append(c + d * r)
        req = 1.0 / np.sqrt((np.sin(theta) / r[0]) ** 2 + (np.cos(theta) / r[2]) ** 2)
        b = clearance[1] if np.cos(theta) > 0 else clearance[2]
        rcl = 1.0 / np.sqrt((np.sin(theta) / max(clearance[0], 1e-6)) ** 2 + (np.cos(theta) / max(b, 1e-6)) ** 2)
        ys = np.linspace(min(c[1], pts[-1][1] - 0.004), y_end, 16)
        for i, y in enumerate(ys):
            t = 1 - smoothstep(SHOULDER_Y + 0.01, c[1], y)
            rad = req + (max(req, rcl * 1.06) - req) * t + flare * max(0.0, SHOULDER_Y - y) + splay * max(0.0, c[1] - 0.06 - y)
            rad *= 1.0 - tuck * smoothstep(c[1], y_end, y) ** 1.5
            wv = wave * np.sin((c[1] - y) * 40 + theta * 3) * min(1.0, (c[1] - y) * 8)
            pts.append(np.array([c[0], y, c[2]]) + h * (rad + wv))
        path = catmull(pts, 30)
        n = len(path)
        tt = np.linspace(0, 1, n)
        w = width * (1 - 0.9 * smoothstep(0.75, 1.0, tt)) if pointy else np.full(n, width)
        w = w * (0.75 + 0.25 * smoothstep(0.0, 0.25, tt))
        sides = np.tile([np.cos(theta), 0.0, np.sin(theta)], (n, 1))
        mats = [mats_fn(p) for p in path] if mats_fn else mat
        self.strand(path, w, thick, sides, mats, blunt=not pointy, wrap=c,
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
    """塗りを焼き込み、表示枠を整える。"""
    m = model
    if m.tex_set is not None:
        _bake.bake(m, face_y=0.862)
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
