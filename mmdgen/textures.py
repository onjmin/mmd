"""Pillow で描くテクスチャ：目・口・鼻・髪・トゥーン・スフィア。モデルと同じフォルダの tex/ に書き出す。"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4  # 描いてから縮小してアンチエイリアスをかける倍率


def _rgb(c, a=255):
    return tuple(int(max(0, min(1, x)) * 255) for x in c) + (int(a),)


def _mix(a, b, t):
    return tuple(np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t)


class TextureSet:
    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, "tex")
        os.makedirs(self.dir, exist_ok=True)

    def save(self, img, name):
        img.save(os.path.join(self.dir, name))
        return "tex\\" + name

    # ---- トゥーン（上が光、下が影。境目をなめらかにつなぐ） ----
    def toon(self, name, shadow):
        h = 64
        y = np.linspace(0, 1, h)
        t = np.clip((y - 0.44) / 0.16, 0, 1)
        t = t * t * (3 - 2 * t)
        col = np.outer(1 - t, [1, 1, 1]) + np.outer(t, shadow)
        img = np.repeat((col * 255).astype(np.uint8)[:, None, :], 32, axis=1)
        return self.save(Image.fromarray(img, "RGB"), name)

    # ---- 髪のつや（加算スフィア：上寄りに天使の輪） ----
    def hair_sphere(self, name="hair_sphere.png", strength=0.50):
        n = 256
        y, x = np.mgrid[0:n, 0:n] / (n - 1) * 2 - 1
        r = np.sqrt(x * x + y * y)
        band = np.exp(-((r - 0.60) / 0.085) ** 2) * np.clip(-y + 0.12, 0, 1) ** 0.8
        v = (band * strength * 255).astype(np.uint8)
        return self.save(Image.fromarray(np.stack([v, v, v], -1), "RGB"), name)

    # ---- 髪（束に沿った繊細な筋と、頭頂のハイライト・根元の影） ----
    def hair(self, name, base, tip=None, tip_from=0.72):
        w, h = 256, 512
        rng = np.random.default_rng(len(name) + 11)
        u = np.linspace(0, 1, w)
        stripe = np.zeros(w)
        for _ in range(42):
            c, s, a = rng.uniform(0, 1), rng.uniform(0.003, 0.016), rng.uniform(-0.10, 0.13)
            stripe += a * np.exp(-((u - c) / s) ** 2)
        # 束の両端をわずかに締めて立体感を出しつつ、パイプ状の暗い筋にならないよう控えめにする
        edge_darken = 1.0 - 0.08 * (2 * u - 1) ** 4
        v = np.linspace(0, 1, h)
        root = 0.88 + 0.12 * np.clip(v / 0.20, 0, 1)
        sheen = 0.07 * np.exp(-((v - 0.24) / 0.08) ** 2)
        img = np.zeros((h, w, 3))
        base = np.asarray(base, float)
        for i in range(h):
            col = base if tip is None else _mix(base, tip, np.clip((v[i] - tip_from) / (1 - tip_from), 0, 1))
            row = np.outer((1 + stripe) * edge_darken, col) * root[i] + sheen[i]
            img[i] = row
        return self.save(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB"), name)

    # ---- 目：白目、瞳、まつ毛の 3 枚 ----
    def eye_white(self, name="eye_white.png"):
        """上まつ毛の弧と下まぶたの弧に目頭・目尻までぴったり収まるアーモンド形の白目。上部にまぶたの落ち影を入れる。"""
        n = 256
        v, u = np.mgrid[0:n, 0:n] / (n - 1)
        x = (u - 0.5) * 2.0  # -1 (目頭) .. +1 (目尻)
        # 目頭・目尻 (x = ±0.91) で上下の弧が自然に一点で交わるアーモンド形
        w = np.maximum(0.0, 1.0 - (x / 0.91) ** 2)
        y_mid = 0.53 + 0.03 * x
        y_top = y_mid - 0.38 * (w ** 0.72)
        y_bot = y_mid + 0.36 * (w ** 0.78)
        span = np.maximum(y_bot - y_top, 1e-4)
        tv = (v - y_top) / span
        inside = (w > 0.0) & (tv >= 0.0) & (tv <= 1.0)
        dist_edge = np.minimum(v - y_top, y_bot - v)
        alpha = np.clip(dist_edge * 55.0, 0.0, 1.0) * inside

        # 上まぶたの青みがかった落ち影＋目頭のほんのり赤い粘膜感
        shadow_t = np.clip((0.36 - tv) / 0.36, 0.0, 1.0) ** 1.4
        col = np.ones((n, n, 3), float) * np.array([0.99, 0.99, 1.00])
        col = col * (1 - shadow_t[..., None]) + np.array([0.76, 0.78, 0.88]) * shadow_t[..., None]
        caruncle = np.clip((0.15 - u) / 0.11, 0.0, 1.0) * np.clip(1.0 - np.abs(tv - 0.55) / 0.40, 0.0, 1.0)
        col = col * (1 - 0.35 * caruncle[..., None]) + np.array([0.96, 0.68, 0.70]) * (0.35 * caruncle[..., None])

        arr = np.dstack([np.clip(col, 0, 1), alpha])
        return self.save(Image.fromarray((arr * 255).astype(np.uint8), "RGBA"), name)

    def iris(self, name, color):
        """既製アニメモデル風の描き込み瞳：濃い外縁リング、上半分を覆う落ち影、下側の発光三日月、
        瞳孔まわりの明るい縁取り、大小2つのメインハイライト＋下部ハイライト。"""
        n = 256 * S
        color = np.asarray(color, float)
        y, x = np.mgrid[0:n, 0:n] / (n - 1)
        dx = (x - 0.50) / 0.485
        dy = (y - 0.50) / 0.485
        r = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)

        # ベース色：暗い上部シェードと鮮やかな下部グラデーション
        dark = color * 0.22 + np.array([0.02, 0.02, 0.04])
        mid = color * 0.95
        bright = np.clip(color * 1.45 + 0.28, 0.0, 1.0)

        # 上部（y < 0.44）はまぶたの影、下半分は鮮やかなグラデーション
        upper_line = 0.43 + 0.05 * (dx ** 2)
        t_upper = np.clip((y - upper_line) / 0.08, 0.0, 1.0)
        t_grad = np.clip((y - 0.42) / 0.45, 0.0, 1.0) ** 1.15
        lower_col = mid * (1 - t_grad[..., None]) + bright * t_grad[..., None]
        col = dark * (1 - t_upper[..., None]) + lower_col * t_upper[..., None]

        # 虹彩の放射状の繊細な筋
        spokes = 0.5 + 0.5 * np.sin(ang * 38.0)
        spoke_mask = np.clip(1.0 - np.abs(r - 0.58) / 0.24, 0.0, 1.0) * t_upper
        col = np.clip(col + (spokes * spoke_mask)[..., None] * 0.10 * bright, 0.0, 1.0)

        # 下側の強い発光三日月（r = 0.55..0.82）
        crescent = np.exp(-((r - 0.68) / 0.12) ** 2) * np.clip((dy - 0.12) / 0.65, 0.0, 1.0) ** 1.2
        col = np.clip(col + crescent[..., None] * (bright * 0.45 + 0.18), 0.0, 1.0)

        # 瞳孔（横方向に少し広げて3D空間で正円に見えるようにする）とその周囲の明るいリング
        pr = np.sqrt((dx / 0.34) ** 2 + ((dy + 0.02) / 0.32) ** 2)
        pupil_rim = np.exp(-((pr - 1.05) / 0.18) ** 2) * np.clip(dy + 0.25, 0.0, 1.0)
        col = np.clip(col + pupil_rim[..., None] * (bright * 0.50 + 0.15), 0.0, 1.0)
        pupil_mask = np.clip((1.0 - pr) * 14.0, 0.0, 1.0)
        pupil_col = color * 0.08 + np.array([0.01, 0.01, 0.02])
        col = col * (1 - pupil_mask[..., None]) + pupil_col * pupil_mask[..., None]

        # 外縁の濃いリム（輪郭をくっきり締める）
        limbal = np.clip((r - 0.80) / 0.18, 0.0, 1.0) ** 1.3
        col = col * (1 - 0.82 * limbal[..., None])

        alpha = np.clip((1.0 - r) * 32.0, 0.0, 1.0)
        img = Image.fromarray((np.dstack([np.clip(col, 0, 1), alpha]) * 255).astype(np.uint8), "RGBA")
        d = ImageDraw.Draw(img)
        # ハイライト：左上（目頭側上）に大きな楕円、右上（目尻側上）に中楕円、左下に小ハイライト
        d.ellipse([n * 0.20, n * 0.22, n * 0.45, n * 0.37], fill=_rgb((1, 1, 1), 252))
        d.ellipse([n * 0.56, n * 0.25, n * 0.75, n * 0.36], fill=_rgb((1, 1, 1), 245))
        d.ellipse([n * 0.24, n * 0.71, n * 0.39, n * 0.79], fill=_rgb((1, 1, 1), 235))
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def lash(self, name="lash.png", color=(0.08, 0.05, 0.06)):
        """既製アニメモデル風の上まつ毛（eye_white の上縁アーチに目頭から目尻まで完全一致し、目尻の束スパイク・二重線・下まつ毛を備える）。"""
        n = 256 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # 上まつ毛のメインアーチ（eye_white の y_top = y_mid - 0.38 * w^0.72 に沿って目頭〜目尻を完全に覆う）
        ts = np.linspace(0.0, 1.0, 90)
        pts_top, pts_bot = [], []
        for t in ts:
            x = -0.92 + 1.84 * t
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.92) ** 2)
            y_mid = 0.53 + 0.03 * x
            y_arch = y_mid - 0.38 * (w ** 0.72)
            env = np.sin(np.pi * t) ** 0.50
            th_up = (0.052 + 0.048 * (t ** 1.0)) * env + 0.008
            th_dn = (0.045 + 0.038 * np.sin(np.pi * t)) * env + 0.006
            pts_top.append((u * n, (y_arch - th_up) * n))
            pts_bot.append((u * n, (y_arch + th_dn) * n))

        warm_dark = _rgb((0.24, 0.06, 0.07), 255)
        core_black = _rgb(color, 255)
        d.polygon(pts_top + pts_bot[::-1], fill=warm_dark)

        # 目尻のハネと束スパイク
        spikes = [
            [(0.84 * n, 0.25 * n), (0.98 * n, 0.31 * n), (0.88 * n, 0.32 * n)],
            [(0.88 * n, 0.32 * n), (1.00 * n, 0.41 * n), (0.91 * n, 0.40 * n)],
            [(0.90 * n, 0.41 * n), (0.99 * n, 0.50 * n), (0.92 * n, 0.52 * n)],
        ]
        for sp in spikes:
            d.polygon(sp, fill=warm_dark)

        # 中央の黒いコア部分
        c_lo, c_hi = int(len(ts) * 0.14), int(len(ts) * 0.80)
        d.polygon(pts_top[c_lo:c_hi] + pts_bot[c_lo:c_hi][::-1], fill=core_black)

        # 二重まぶたの線（上まつ毛の上に沿う繊細な弧）
        crease_pts = []
        for t in np.linspace(0.16, 0.78, 40):
            x = -0.92 + 1.84 * t
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.92) ** 2)
            y_c = (0.53 + 0.03 * x) - 0.38 * (w ** 0.72) - 0.095 * np.sin(np.pi * t) ** 0.6
            crease_pts.append((u * n, y_c * n))
        d.line(crease_pts, fill=_rgb((0.26, 0.10, 0.11), 185), width=max(2, int(n * 0.0085)))

        # 下まつ毛（白目の下縁 y_bot = y_mid + 0.36 * w^0.78 に沿う細いライン＋小さなまつ毛の毛先）
        low_pts = []
        for t in np.linspace(0.22, 0.80, 36):
            x = -0.91 + 1.82 * t
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.91) ** 2)
            y_l = (0.53 + 0.03 * x) + 0.36 * (w ** 0.78)
            low_pts.append((u * n, y_l * n))
        d.line(low_pts, fill=_rgb((0.24, 0.09, 0.10), 195), width=max(2, int(n * 0.009)))
        for t_tick in (0.32, 0.43, 0.54, 0.65, 0.75):
            x = -0.91 + 1.82 * t_tick
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.91) ** 2)
            y_l = (0.53 + 0.03 * x) + 0.36 * (w ** 0.78)
            dx_t = (t_tick - 0.45) * 0.022
            d.polygon([
                ((u - 0.007) * n, y_l * n),
                ((u + 0.007) * n, y_l * n),
                ((u + dx_t) * n, (y_l + 0.028) * n),
            ], fill=_rgb((0.24, 0.09, 0.10), 170))

        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def brow(self, name, color):
        n = 128 * S
        img = Image.new("RGBA", (n * 2, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        pts_top, pts_bot = [], []
        for t in np.linspace(0, 1, 50):
            x = 0.06 + 0.88 * t
            y = 0.54 - 0.24 * np.sin(np.pi * (0.20 + 0.70 * t))
            th = 0.12 * np.sin(np.pi * t) ** 0.7 * (1.0 - 0.35 * t) + 0.012
            pts_top.append((x * n * 2, (y - th / 2) * n))
            pts_bot.append((x * n * 2, (y + th / 2) * n))
        d.polygon(pts_top + pts_bot[::-1], fill=_rgb(color, 240))
        return self.save(img.resize((256, 128), Image.LANCZOS), name)

    def nose(self, name="nose.png", skin_color=(1.0, 0.86, 0.76)):
        """アニメ風の繊細な鼻筋・鼻先の影線（楕円のシールではなく、細い縦の影線と下の淡い落ち影）。"""
        n = 128 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # 鼻先の下側の淡い肌影
        shadow_col = _rgb(np.asarray(skin_color) * np.array([0.78, 0.62, 0.60]), 110)
        d.ellipse([n * 0.30, n * 0.54, n * 0.72, n * 0.84], fill=shadow_col)
        img = img.filter(ImageFilter.GaussianBlur(n * 0.04))
        d = ImageDraw.Draw(img)
        # 鼻先のシャープな細い線（くの字に近い縦線）
        line_col = _rgb(np.asarray(skin_color) * np.array([0.42, 0.28, 0.26]), 220)
        stroke = [
            (n * 0.48, n * 0.16),
            (n * 0.43, n * 0.56),
            (n * 0.48, n * 0.76),
            (n * 0.54, n * 0.55),
        ]
        d.polygon(stroke, fill=line_col)
        return self.save(img.resize((128, 128), Image.LANCZOS), name)

    def blush(self, name="blush.png", color=(1.0, 0.44, 0.48)):
        """ふんわりぼかしたチーク＋小さな白いハイライト点（目の白目に重ならない）。"""
        w, h = 256, 128
        y, x = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None] * 2 - 1
        r2 = x * x + (y * 1.15) ** 2
        a = np.clip(1.0 - r2, 0.0, 1.0) ** 2.0 * 0.34
        col = np.broadcast_to(np.asarray(color, float), (h, w, 3)).copy()
        # 頬の上寄りに小さな白いツヤ点
        hl = np.exp(-(((x + 0.18) / 0.10) ** 2 + ((y + 0.22) / 0.14) ** 2))
        col = col * (1 - hl[..., None]) + np.ones((h, w, 3)) * hl[..., None]
        a = np.maximum(a, hl * 0.65)
        arr = np.dstack([np.clip(col, 0, 1), np.clip(a, 0, 1)])
        return self.save(Image.fromarray((arr * 255).astype(np.uint8), "RGBA"), name)

    def mouth_line(self, name="mouth_line.png", color=(0.42, 0.18, 0.19)):
        """閉じた口の繊細なリップライン（中央と口角に抑揚があり、下唇にほんのり血色が入る）。"""
        w, h = 256 * S, 128 * S
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # 下唇の淡いピンクのツヤ・血色
        lip_tint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dt = ImageDraw.Draw(lip_tint)
        dt.ellipse([w * 0.25, h * 0.46, w * 0.75, h * 0.82], fill=_rgb((0.95, 0.58, 0.58), 75))
        lip_tint = lip_tint.filter(ImageFilter.GaussianBlur(h * 0.08))
        img = Image.alpha_composite(img, lip_tint)
        d = ImageDraw.Draw(img)

        pts_top, pts_bot = [], []
        for t in np.linspace(0, 1, 60):
            u = 0.05 + 0.90 * t
            x = (t - 0.5) * 2.0
            # 口角がキュッと上がる微笑みライン
            y = 0.52 - 0.12 * (x ** 2)
            # 口角と中央を少し強調し、端は細く抜く
            env = np.sin(np.pi * t) ** 0.65
            th = (0.065 + 0.035 * np.exp(-((x / 0.25) ** 2)) + 0.045 * np.exp(-(((abs(x) - 0.78) / 0.15) ** 2))) * env
            pts_top.append((u * w, (y - th * 0.45) * h))
            pts_bot.append((u * w, (y + th * 0.55) * h))
        d.polygon(pts_top + pts_bot[::-1], fill=_rgb(color, 245))
        return self.save(img.resize((256, 128), Image.LANCZOS), name)

    def mouth_interior(self, name="mouth_in.png", fang=False):
        """開いた口の中（上部に白い歯並、奥に暗い口内、下部に丸いピンクの舌を一枚のUVに描く）。"""
        n = 256 * S
        v, u = np.mgrid[0:n, 0:n] / (n - 1)
        x = (u - 0.5) * 2.0
        # 口内のベース（上が暗く奥まった赤茶、中央がコーラルレッド）
        top_dark = np.array([0.38, 0.12, 0.14])
        mid_cavity = np.array([0.62, 0.22, 0.25])
        tc = np.clip((v - 0.18) / 0.45, 0.0, 1.0)[..., None]
        col = top_dark * (1 - tc) + mid_cavity * tc

        # 下部の丸いピンクの舌（ドーム形）
        tongue_top = 0.50 + 0.18 * (x ** 2)
        in_tongue = v >= tongue_top
        tt = np.clip((v - tongue_top) / 0.42, 0.0, 1.0)
        tongue_col = np.array([0.86, 0.42, 0.45]) * (1 - tt[..., None]) + np.array([0.95, 0.56, 0.58]) * tt[..., None]
        # 舌の中央のわずかな溝の影
        groove = np.exp(-((x / 0.12) ** 2)) * np.clip(1.0 - tt, 0.0, 1.0) * 0.12
        tongue_col = np.clip(tongue_col - groove[..., None], 0.0, 1.0)
        t_blend = np.clip((v - tongue_top) * 35.0, 0.0, 1.0)[..., None]
        col = col * (1 - t_blend) + tongue_col * t_blend

        # 上部の白い歯並び（u 方向にわずかな陰影、下縁にすっきりしたライン）
        teeth_bot = 0.25 + 0.04 * (x ** 2)
        if fang:
            # 左口角寄り（x ≈ -0.46）に小さな八重歯
            teeth_bot = teeth_bot + 0.11 * np.maximum(0.0, 1.0 - np.abs(x + 0.45) / 0.13)
        t_teeth = np.clip((teeth_bot - v) * 40.0, 0.0, 1.0)[..., None]
        tooth_shade = 1.0 - 0.14 * (x ** 2)[..., None]
        tooth_col = np.array([0.98, 0.97, 0.96]) * tooth_shade
        col = col * (1 - t_teeth) + tooth_col * t_teeth

        # 口の輪郭の細いリップエッジ（周囲を引き締める）
        r_edge = np.sqrt(x * x + ((v - 0.5) * 2.0) ** 2)
        rim = np.clip((r_edge - 0.88) / 0.10, 0.0, 1.0)[..., None]
        col = col * (1 - 0.65 * rim) + np.array([0.32, 0.12, 0.13]) * (0.65 * rim)

        img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGB")
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def solid(self, name, color, noise=0.0, pattern=None):
        """服の色。pattern(u, v) で模様（縁取りの線など）を足せる。"""
        n = 256
        v, u = np.mgrid[0:n, 0:n] / (n - 1)
        col = np.ones((n, n, 3)) * np.asarray(color, float)
        if noise:
            col *= 1 + noise * np.random.default_rng(1).normal(0, 1, (n, n, 1))
        if pattern is not None:
            col = pattern(u, v, col)
        return self.save(Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGB"), name)

    # ---- 模様と印 ----
    def meander(self, name, bg, fg, tiles=1):
        """横に続く雷文（四角い渦）の帯。横方向にくり返して貼る。"""
        w, h = 64 * S, 64 * S
        img = Image.new("RGB", (w, h), _rgb(bg)[:3])
        d = ImageDraw.Draw(img)
        lw = max(1, int(h * 0.07))
        g = h / 8  # 1 マスの大きさ
        pts = [(0, 6), (6, 6), (6, 1), (1, 1), (1, 4), (4, 4), (4, 3), (3, 3)]
        d.line([(x * g, y * g) for x, y in pts], fill=_rgb(fg)[:3], width=lw, joint="curve")
        d.line([(6 * g, 6 * g), (8 * g, 6 * g)], fill=_rgb(fg)[:3], width=lw)
        for y in (0.4, 7.6):  # 上下の細い線
            d.line([(0, y * g), (w, y * g)], fill=_rgb(fg)[:3], width=max(1, lw // 2))
        return self.save(img.resize((64, 64), Image.LANCZOS), name)

    def flame(self, name, top, mid, tip, tongues=6):
        """裾の炎：上から top→mid→tip の色に変わり、下の方は舌の形に透けて消える。横にくり返して貼る。"""
        w, h = 128 * S, 128 * S
        v, u = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None]
        top, mid, tip = (np.asarray(x, float) for x in (top, mid, tip))
        t1 = np.clip(v / 0.5, 0, 1)[..., None]
        t2 = np.clip((v - 0.5) / 0.5, 0, 1)[..., None]
        col = np.where(v[..., None] < 0.5, top * (1 - t1) + mid * t1, mid * (1 - t2) + tip * t2)
        reach = 0.62 + 0.30 * np.abs(np.sin(u * np.pi * tongues)) ** 2  # 炎の舌の先の高さ
        alpha = np.clip((reach - v) / 0.18, 0, 1) ** 0.8
        img = Image.fromarray((np.dstack([col, alpha]) * 255).astype(np.uint8), "RGBA")
        return self.save(img.resize((128, 128), Image.LANCZOS), name)

    def mark(self, name, text, color, font="arialbd.ttf"):
        """肩などに入れる文字の印（背景は透明）。"""
        from PIL import ImageFont
        w, h = 128 * S, 64 * S
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(font, int(h * 0.8))
        except OSError:
            f = ImageFont.load_default()
        box = d.textbbox((0, 0), text, font=f)
        d.text(((w - box[2] - box[0]) / 2, (h - box[3] - box[1]) / 2), text, font=f, fill=_rgb(color))
        return self.save(img.resize((128, 64), Image.LANCZOS), name)
