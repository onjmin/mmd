"""足と靴：足の木型（かかと〜つま先の断面の並び）から、足・靴の甲革・靴底・履き口の縁・ストラップを作る。

既製の MMD モデルの靴は、靴底・かかと・甲・履き口・留め具が別々の形でできている。卵形のかたまり 1 個では靴に見えない。
座標は体と同じ（つま先が -Z、地面が y=0）。t は かかとの後ろ 0 → つま先 1。断面の角 a は 0 が体の外でも内でもなく +X 側、
π/2 が上、3π/2 が下。
"""

import numpy as np

from .body import joints
from .geom import loft, tube, TAU


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Last:
    """片足の木型。s は左右（左 +1）。wide・high で幅と甲の高さを、long でつま先側の長さを変える。
    round_toe が大きいほどつま先が丸く太い（ブーツ）、小さいほど細い。"""

    def __init__(self, s, wide=1.0, high=1.0, long=1.0, round_toe=1.0, lift=0.0):
        self.s = s
        self.x0 = joints(s)["足首"][0]
        self.z_heel, self.z_toe = 0.034, 0.034 - 0.120 * long
        self.wide, self.high, self.round_toe, self.lift = wide, high, round_toe, lift

    # ---- 断面の寸法 ----
    def z(self, t):
        return self.z_heel + (self.z_toe - self.z_heel) * t

    def _ends(self, t):
        """かかととつま先の丸め（0〜1）。"""
        t = np.asarray(t, float)
        heel = np.sqrt(np.clip(1 - (1 - t / 0.11) ** 2, 0, 1))
        k = 0.20
        toe = np.clip(1 - ((t - (1 - k)) / k) ** 2, 0, 1) ** (0.5 / self.round_toe)
        return np.where(t < 0.11, heel, np.where(t > 1 - k, toe, 1.0))

    def width(self, t):
        w = np.interp(t, [0.0, 0.12, 0.40, 0.68, 0.86, 1.0], [0.0180, 0.0188, 0.0210, 0.0262, 0.0245, 0.0215])
        return w * self.wide * np.maximum(self._ends(t), 0.04)

    def bottom(self, t):
        return self.lift + 0.0040 * _smooth(0.74, 1.0, t)  # つま先は少し反り上がる

    def top(self, t):
        h = np.interp(t, [0.0, 0.10, 0.30, 0.45, 0.65, 0.85, 1.0], [0.044, 0.050, 0.052, 0.041, 0.028, 0.0215, 0.0175])
        e = self._ends(t)
        e = np.where(np.asarray(t) < 0.5, 0.60 + 0.40 * e, np.maximum(e, 0.05))
        return self.bottom(t) + h * self.high * e

    def cx(self, t):
        return self.x0 + self.s * 0.0045 * np.asarray(t)  # つま先は少し外を向く

    def point(self, t, a, grow=0.0):
        """木型の面の点。grow だけ外へふくらませる（靴の革の厚み）。t, a は同じ形の配列。"""
        t, a = np.asarray(t, float), np.asarray(a, float)
        w, bot, top = self.width(t) + grow, self.bottom(t) - grow * 0.2, self.top(t) + grow
        yc = bot + 0.36 * (top - bot)
        ca, sa = np.cos(a), np.sin(a)
        # 上半分は丸い甲、下半分は角を丸めた平らな底
        x = np.sign(ca) * np.abs(ca) ** np.where(sa >= 0, 1.0, 0.62) * w
        y = np.where(sa >= 0, yc + (top - yc) * sa, yc + (yc - bot) * np.sign(sa) * np.abs(sa) ** 0.62)
        return np.stack([self.cx(t) + x, y, self.z(t)], -1)

    def _rings(self, ts, n, grow):
        a = np.arange(n) / n * TAU
        return [self.point(np.full(n, t), a, grow) for t in ts]

    # ---- 部品 ----
    def foot(self, grow=0.0, nt=44, n=40):
        """足そのもの（靴下や素足、またはブーツの足の部分）。閉じた形。"""
        ts = np.linspace(0.0, 1.0, nt)
        rings = self._rings(ts, n, grow)
        cen = [np.array([self.cx(t), (self.bottom(t) + self.top(t)) / 2, self.z(t)]) for t in ts]
        return loft(rings, cap0=True, cap1=True, centers=cen)

    def upper(self, rim, grow=0.0018, nt=44, n=41):
        """甲革：足を包む殻。rim(t) の高さより上は開ける（履き口）。rim が甲の高さ以上の所は閉じる。
        返り値は (面, 履き口の縁の点列)。縁は閉じた輪で、縁取りの管を通すのに使う。"""
        ts = np.linspace(0.0, 1.0, nt)
        rings, left, right, cen = [], [], [], []
        for t in ts:
            bot, top = self.bottom(t) - grow * 0.2, self.top(t) + grow
            yc = bot + 0.36 * (top - bot)
            q = float(np.clip((rim(t) - yc) / max(top - yc, 1e-6), -0.2, 1.0))
            al = np.arcsin(np.clip(q, -1, 1)) if q >= 0 else -np.arcsin(np.clip(-q, 0, 1) ** (1 / 0.62))
            a = np.linspace(np.pi - al, TAU + al, n)  # 片側の縁から底を回って反対の縁へ
            pts = self.point(np.full(n, t), a, grow)
            rings.append(pts)
            left.append(pts[0])
            right.append(pts[-1])
            cen.append(np.array([self.cx(t), (bot + top) / 2, self.z(t)]))
        mesh = loft(rings, closed=False, centers=cen)
        open_ = [i for i in range(nt) if np.linalg.norm(left[i] - right[i]) > 1e-4]
        i0, i1 = open_[0], open_[-1]
        loop = left[i0:i1 + 1] + right[i0:i1 + 1][::-1]
        return mesh, np.array(loop)

    def sole(self, thick=0.0065, welt=0.0022, heel=0.0045, n=72):
        """靴底：足の輪郭より少し張り出した板。土踏まずを heel だけ切り上げて、かかとの段を付ける。"""
        ts = np.concatenate([np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2)])[:-1]
        side = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])[:-1]
        w = self.width(ts) + welt
        x = self.cx(ts) + side * w
        z = self.z(ts) + np.where(ts < 0.05, welt * (1 - ts / 0.05), 0.0) - np.where(ts > 0.95, welt * (ts - 0.95) / 0.05, 0.0)
        arch = heel * _smooth(0.26, 0.31, ts) * (1 - _smooth(0.50, 0.60, ts))
        spring = 0.0040 * _smooth(0.74, 1.0, ts)
        y_top = self.lift - 0.0002 + spring
        y_bot = y_top - thick + arch
        cen = np.array([self.x0, 0, (self.z_heel + self.z_toe) / 2])
        rings = []
        for k, (yy, inset) in enumerate([(y_top, 0.7), (y_top, 1.0), (y_bot, 1.0), (y_bot, 0.55)]):
            pts = np.stack([x, np.broadcast_to(yy, x.shape), z], 1)
            rings.append(cen + (pts - cen) * [inset, 1, inset] + [0, 0, 0])
            rings[-1][:, 1] = yy
        cs = [np.array([cen[0], float(np.mean(r[:, 1])), cen[2]]) for r in rings]
        return loft(rings, cap0=True, cap1=True, centers=cs)

    def band(self, t, half=0.0045, thick=0.0013, grow=0.0026, a0=np.pi * 0.08, a1=np.pi * 0.92, n=28):
        """足の甲を横切る帯（ストラップ）。t の位置で角 a0〜a1 を渡る平たい管。"""
        a = np.linspace(a0, a1, n)
        path = self.point(np.full(n, t), a, grow)
        return tube(path, half, thick, n=10, side=np.array([0, 0, 1.0]), cap0=True, cap1=True)

    def line(self, ts, a, grow=0.002):
        """木型の面の上を通る点列（縫い目・ひもを通す道）。"""
        return self.point(np.asarray(ts, float), np.asarray(a, float), grow)
