"""Pillow で描くテクスチャ：目・口・鼻・へそ・髪・トゥーン・スフィア。モデルと同じフォルダの tex/ に書き出す。"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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

    # ---- トゥーン（上が光、下が影。境目をなめらかにつなぐ正方形テクスチャ） ----
    def toon(self, name, shadow):
        h = 32
        y = np.linspace(0, 1, h)
        t = np.clip((y - 0.44) / 0.16, 0, 1)
        t = t * t * (3 - 2 * t)
        col = np.outer(1 - t, [1, 1, 1]) + np.outer(t, shadow)
        img = np.repeat((col * 255).astype(np.uint8)[:, None, :], h, axis=1)
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
        edge_darken = 1.0 - 0.06 * (2 * u - 1) ** 4
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

    # ---- 目：白目、瞳、まつ毛の 3 枚（既製アニメモデル KK/N4D と立ち絵の形状・配色に準拠） ----
    def eye_white(self, name="eye_white.png"):
        """上が丸いドーム状、下が緩やかな弧を描くアニメの白目。上部にまぶたの落ち影を入れる。"""
        n = 256
        v, u = np.mgrid[0:n, 0:n] / (n - 1)
        x = (u - 0.5) * 2.0  # -1 (目頭) .. +1 (目尻)
        w = np.maximum(0.0, 1.0 - (x / 0.84) ** 2)
        y_base = 0.62 - 0.05 * x
        y_top = y_base - 0.50 * (w ** 0.45)
        y_bot = y_base + 0.30 * (w ** 0.65)
        span = np.maximum(y_bot - y_top, 1e-4)
        tv = (v - y_top) / span
        inside = (w > 0.0) & (tv >= 0.0) & (tv <= 1.0)
        dist_edge = np.minimum(v - y_top, y_bot - v)
        alpha = np.clip(dist_edge * 55.0, 0.0, 1.0) * inside

        # 上まぶたの落ち影
        shadow_t = np.clip((0.32 - tv) / 0.32, 0.0, 1.0) ** 1.3
        col = np.ones((n, n, 3), float) * np.array([0.98, 0.98, 0.99])
        col = col * (1 - shadow_t[..., None]) + np.array([0.80, 0.81, 0.86]) * shadow_t[..., None]
        arr = np.dstack([np.clip(col, 0, 1), alpha])
        return self.save(Image.fromarray((arr * 255).astype(np.uint8), "RGBA"), name)

    def iris(self, name, color):
        """既製アニメモデル（KK_EyeL）・立ち絵準拠の瞳：
        太い黒の外縁リング、上半分を覆う漆黒のシェード、中央の縦長楕円の黒瞳孔、
        下半分をU字に彩る鮮やかなグラデーション、上部左右2つの白い楕円ハイライト＋左下の小ハイライト。"""
        n = 256 * S
        color = np.asarray(color, float)
        y, x = np.mgrid[0:n, 0:n] / (n - 1)
        dx = (x - 0.50) / 0.48
        dy = (y - 0.50) / 0.48
        r = np.sqrt(dx * dx + dy * dy)

        dark_hood = color * 0.06 + np.array([0.01, 0.02, 0.02])
        outer_black = np.array([0.02, 0.02, 0.03])
        mid_col = np.clip(color * 0.88 + 0.06, 0.0, 1.0)
        bright_col = np.clip(color * 1.25 + 0.36, 0.0, 1.0)
        rim_col = np.clip(color * 0.68 + 0.04, 0.0, 1.0)

        # 下半分（y >= 0.52）のU字グラデーション
        t_v = np.clip((y - 0.52) / 0.36, 0.0, 1.0)
        lower_col = mid_col * (1 - t_v[..., None]) + bright_col * t_v[..., None]
        # 下縁の内側リム（r = 0.76..0.85）を少し落ち着いた色にして立体感を出す
        inner_rim = np.clip((r - 0.74) / 0.09, 0.0, 1.0)
        lower_col = lower_col * (1 - 0.55 * inner_rim[..., None]) + rim_col * (0.55 * inner_rim[..., None])

        # 上半分（y < 0.52）は漆黒のシェード
        t_hood = np.clip((y - 0.51) * 45.0, 0.0, 1.0)[..., None]
        col = dark_hood * (1 - t_hood) + lower_col * t_hood

        # 中央の大きな縦長黒瞳孔（U字の内側を作る）とその周囲のわずかな中間色リング
        pr = np.sqrt((dx / 0.24) ** 2 + ((dy - 0.02) / 0.34) ** 2)
        p_halo = np.clip((1.22 - pr) * 10.0, 0.0, 1.0) * t_hood[..., 0]
        col = col * (1 - 0.35 * p_halo[..., None]) + mid_col * (0.35 * p_halo[..., None])
        p_core = np.clip((1.0 - pr) * 18.0, 0.0, 1.0)[..., None]
        col = col * (1 - p_core) + dark_hood * p_core

        # 外周の太い黒リング（r = 0.83..1.00）
        limbal = np.clip((r - 0.82) * 28.0, 0.0, 1.0)[..., None]
        col = col * (1 - limbal) + outer_black * limbal

        alpha = np.clip((1.0 - r) * 36.0, 0.0, 1.0)
        img = Image.fromarray((np.dstack([np.clip(col, 0, 1), alpha]) * 255).astype(np.uint8), "RGBA")
        d = ImageDraw.Draw(img)
        # 上部左右に並ぶ2つの白い横長楕円ハイライト＋左下の小ハイライト（KK_EyeL 準拠）
        d.ellipse([n * 0.16, n * 0.24, n * 0.41, n * 0.35], fill=_rgb((1, 1, 1), 255))
        d.ellipse([n * 0.59, n * 0.24, n * 0.84, n * 0.35], fill=_rgb((1, 1, 1), 255))
        d.ellipse([n * 0.19, n * 0.77, n * 0.37, n * 0.84], fill=_rgb((1, 1, 1), 250))
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def lash(self, name="lash.png", color=(0.06, 0.04, 0.05)):
        """既製アニメモデル（KK_Eyeline_up）・立ち絵準拠のアイライン：
        上まぶたの太いドームアーチ、目頭・目尻の赤茶グラデーション、目尻の外側〜下まで包み込むラインとまつ毛スパイク、
        上部の二重まぶた線、下まぶた中央の短いアイライン。"""
        n = 256 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        warm_red = _rgb((0.34, 0.05, 0.06), 255)
        core_black = _rgb(color, 255)

        # 1) 上まつ毛のメインアーチ（目頭 x=-0.88 から上ドームを通り、目尻 x=+0.88 の下まで包み込む）
        ts = np.linspace(0.0, 1.0, 100)
        pts_top, pts_bot = [], []
        for t in ts:
            x = -0.88 + 1.76 * t
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.85) ** 2)
            y_base = 0.62 - 0.05 * x
            y_arch = y_base - 0.50 * (w ** 0.45)
            env = np.sin(np.pi * t) ** 0.42
            th_up = (0.080 + 0.050 * (t ** 1.1)) * env + 0.010
            th_dn = (0.065 + 0.075 * (t ** 1.5)) * env + 0.008
            pts_top.append((u * n, (y_arch - th_up) * n))
            pts_bot.append((u * n, (y_arch + th_dn) * n))

        d.polygon(pts_top + pts_bot[::-1], fill=warm_red)

        # 目尻の下まで回り込むアイラインの垂れ（KK_Eyeline_up 右下の包みを再現）
        outer_hood = [
            (0.76 * n, 0.24 * n),
            (0.95 * n, 0.48 * n),
            (0.88 * n, 0.56 * n),
            (0.76 * n, 0.72 * n),
            (0.68 * n, 0.78 * n),
            (0.76 * n, 0.56 * n),
            (0.74 * n, 0.36 * n),
        ]
        d.polygon(outer_hood, fill=warm_red)

        # 目頭・目尻の繊細なまつ毛スパイク
        spikes = [
            [(0.12 * n, 0.38 * n), (0.04 * n, 0.47 * n), (0.14 * n, 0.44 * n)],
            [(0.10 * n, 0.46 * n), (0.03 * n, 0.55 * n), (0.12 * n, 0.52 * n)],
            [(0.78 * n, 0.22 * n), (0.95 * n, 0.32 * n), (0.83 * n, 0.30 * n)],
            [(0.83 * n, 0.31 * n), (0.97 * n, 0.41 * n), (0.86 * n, 0.39 * n)],
        ]
        for sp in spikes:
            d.polygon(sp, fill=warm_red)

        # 中央の黒いコア部分
        c_lo, c_hi = int(len(ts) * 0.16), int(len(ts) * 0.82)
        d.polygon(pts_top[c_lo:c_hi] + pts_bot[c_lo:c_hi][::-1], fill=core_black)

        # 2) 二重まぶたの線（上まつ毛の少し上に沿う細い弧）
        crease_pts = []
        for t in np.linspace(0.08, 0.70, 45):
            x = -0.88 + 1.76 * t
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.85) ** 2)
            y_c = (0.62 - 0.05 * x) - 0.50 * (w ** 0.45) - 0.115 * (np.sin(np.pi * t) ** 0.55)
            crease_pts.append((u * n, y_c * n))
        d.line(crease_pts, fill=warm_red, width=max(2, int(n * 0.011)))

        # 3) 下まぶた中央の短いアイライン（KK_Eyeline_up 下部準拠）
        low_pts = []
        for t in np.linspace(0.32, 0.66, 24):
            x = -0.88 + 1.76 * t
            u = 0.5 + 0.5 * x
            w = max(0.0, 1.0 - (x / 0.85) ** 2)
            y_l = (0.62 - 0.05 * x) + 0.30 * (w ** 0.65)
            low_pts.append((u * n, y_l * n))
        d.line(low_pts, fill=warm_red, width=max(3, int(n * 0.015)))

        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def brow(self, name, color):
        """目頭側から目尻へなだらかに弧を描いて細く抜ける繊細な眉。"""
        w, h = 256 * S, 64 * S
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        ts = np.linspace(0.0, 1.0, 50)
        pts_top, pts_bot = [], []
        for t in ts:
            u = 0.06 + 0.88 * t
            y_c = 0.56 - 0.22 * np.sin(np.pi * (t ** 0.85))
            thick = 0.16 * (np.sin(np.pi * t) ** 0.75) * (1.0 - 0.35 * t)
            pts_top.append((u * w, (y_c - thick) * h))
            pts_bot.append((u * w, (y_c + thick) * h))
        d.polygon(pts_top + pts_bot[::-1], fill=_rgb(color, 235))
        return self.save(img.resize((256, 64), Image.LANCZOS), name)

    def nose(self, name="nose.png", skin_color=(1.0, 0.86, 0.76)):
        """既製アニメモデル（KK_Nose）・立ち絵準拠の控えめで繊細な鼻先チョン点＋淡い立体影。"""
        n = 128 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        sc = np.asarray(skin_color, float)
        # 鼻先の下のほんのり淡い立体影
        shade_col = _rgb(sc * np.array([0.82, 0.68, 0.64]), 70)
        d.ellipse([n * 0.36, n * 0.52, n * 0.64, n * 0.82], fill=shade_col)
        img = img.filter(ImageFilter.GaussianBlur(n * 0.05))
        d = ImageDraw.Draw(img)
        # 鼻先の小さなチョン点（主張しすぎない暖色ブラウン）
        line_col = _rgb(sc * np.array([0.46, 0.28, 0.25]), 185)
        stroke = [
            (n * 0.48, n * 0.46),
            (n * 0.52, n * 0.50),
            (n * 0.52, n * 0.68),
            (n * 0.49, n * 0.76),
            (n * 0.47, n * 0.66),
        ]
        d.polygon(stroke, fill=line_col)
        return self.save(img.resize((128, 128), Image.LANCZOS), name)

    def blush(self, name="blush.png", color=(1.0, 0.44, 0.48)):
        """ふんわりぼかしたチーク＋小さな白いハイライト点（KK_Face_light 準拠）。"""
        w, h = 256, 128
        y, x = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None] * 2 - 1
        r2 = x * x + (y * 1.15) ** 2
        a = np.clip(1.0 - r2, 0.0, 1.0) ** 2.0 * 0.30
        col = np.broadcast_to(np.asarray(color, float), (h, w, 3)).copy()
        hl = np.exp(-(((x + 0.18) / 0.09) ** 2 + ((y + 0.24) / 0.13) ** 2))
        col = col * (1 - hl[..., None]) + np.ones((h, w, 3)) * hl[..., None]
        a = np.maximum(a, hl * 0.70)
        arr = np.dstack([np.clip(col, 0, 1), np.clip(a, 0, 1)])
        return self.save(Image.fromarray((arr * 255).astype(np.uint8), "RGBA"), name)

    def mouth_line(self, name="mouth_line.png", color=(0.32, 0.14, 0.14)):
        """立ち絵（ref/_mouth_chin.png）準拠の繊細でくっきりした微笑みリップライン＋控えめな下唇の血色。"""
        w, h = 256 * S, 128 * S
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # 下唇のごく淡く引き締まった血色（広がりすぎないようにする）
        lip_tint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dt = ImageDraw.Draw(lip_tint)
        dt.ellipse([w * 0.28, h * 0.52, w * 0.72, h * 0.74], fill=_rgb((0.94, 0.62, 0.60), 65))
        lip_tint = lip_tint.filter(ImageFilter.GaussianBlur(h * 0.045))
        img = Image.alpha_composite(img, lip_tint)
        d = ImageDraw.Draw(img)

        pts_top, pts_bot = [], []
        for t in np.linspace(0, 1, 60):
            u = 0.04 + 0.92 * t
            x = (t - 0.5) * 2.0
            # 中央は穏やかで、口角だけキュッと上がる自然なアニメの微笑み線
            y = 0.54 - 0.09 * (x ** 2) - 0.04 * (x ** 4)
            env = np.sin(np.pi * t) ** 0.50
            th = (0.048 + 0.025 * np.exp(-(((abs(x) - 0.75) / 0.18) ** 2))) * env
            pts_top.append((u * w, (y - th * 0.48) * h))
            pts_bot.append((u * w, (y + th * 0.52) * h))
        d.polygon(pts_top + pts_bot[::-1], fill=_rgb(color, 235))
        return self.save(img.resize((256, 128), Image.LANCZOS), name)

    def mouth_interior(self, name="mouth_in.png", fang=False):
        """開いた口の中（上部に白い歯並、奥に暗い口内、下部に丸いピンクの舌を一枚のUVに描く）。"""
        n = 256 * S
        v, u = np.mgrid[0:n, 0:n] / (n - 1)
        x = (u - 0.5) * 2.0
        top_dark = np.array([0.38, 0.12, 0.14])
        mid_cavity = np.array([0.62, 0.22, 0.25])
        tc = np.clip((v - 0.18) / 0.45, 0.0, 1.0)[..., None]
        col = top_dark * (1 - tc) + mid_cavity * tc

        tongue_top = 0.50 + 0.18 * (x ** 2)
        tt = np.clip((v - tongue_top) / 0.42, 0.0, 1.0)
        tongue_col = np.array([0.86, 0.42, 0.45]) * (1 - tt[..., None]) + np.array([0.95, 0.56, 0.58]) * tt[..., None]
        groove = np.exp(-((x / 0.12) ** 2)) * np.clip(1.0 - tt, 0.0, 1.0) * 0.12
        tongue_col = np.clip(tongue_col - groove[..., None], 0.0, 1.0)
        t_blend = np.clip((v - tongue_top) * 35.0, 0.0, 1.0)[..., None]
        col = col * (1 - t_blend) + tongue_col * t_blend

        teeth_bot = 0.25 + 0.04 * (x ** 2)
        if fang:
            teeth_bot = teeth_bot + 0.11 * np.maximum(0.0, 1.0 - np.abs(x + 0.45) / 0.13)
        t_teeth = np.clip((teeth_bot - v) * 40.0, 0.0, 1.0)[..., None]
        tooth_shade = 1.0 - 0.14 * (x ** 2)[..., None]
        tooth_col = np.array([0.98, 0.97, 0.96]) * tooth_shade
        col = col * (1 - t_teeth) + tooth_col * t_teeth

        r_edge = np.sqrt(x * x + ((v - 0.5) * 2.0) ** 2)
        rim = np.clip((r_edge - 0.88) / 0.10, 0.0, 1.0)[..., None]
        col = col * (1 - 0.65 * rim) + np.array([0.32, 0.12, 0.13]) * (0.65 * rim)

        img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGB")
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def navel(self, name="navel.png", skin_color=(1.0, 0.86, 0.76)):
        """おなかの中央の縦長のおへそ（cf_m_body テクスチャと立ち絵の腹筋・へそ影に準拠）。"""
        w, h = 128 * S, 256 * S
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sc = np.asarray(skin_color, float)
        d = ImageDraw.Draw(img)
        # へその上〜みぞおちへ伸びる淡い腹筋の縦影
        d.ellipse([w * 0.36, h * 0.08, w * 0.64, h * 0.78], fill=_rgb(sc * np.array([0.84, 0.68, 0.62]), 55))
        # へそ周りの柔らかな窪み影
        d.ellipse([w * 0.28, h * 0.46, w * 0.72, h * 0.78], fill=_rgb(sc * np.array([0.78, 0.58, 0.52]), 95))
        img = img.filter(ImageFilter.GaussianBlur(w * 0.08))
        d = ImageDraw.Draw(img)
        # へその中心の縦長スリット影
        d.polygon([
            (w * 0.50, h * 0.50),
            (w * 0.56, h * 0.58),
            (w * 0.54, h * 0.69),
            (w * 0.50, h * 0.74),
            (w * 0.46, h * 0.69),
            (w * 0.44, h * 0.58),
        ], fill=_rgb(sc * np.array([0.52, 0.32, 0.28]), 215))
        return self.save(img.resize((128, 256), Image.LANCZOS), name)

    def buckle(self, name="buckle.png", gold=(0.78, 0.64, 0.42), dark=(0.44, 0.34, 0.22)):
        """立ち絵（ref/_crotch.png）準拠の四角い金バックル（渦巻き雲の彫り模様入り）。"""
        w, h = 256, 192
        img = Image.new("RGBA", (w, h), _rgb(gold, 255))
        d = ImageDraw.Draw(img)
        # 外枠の彫り線
        d.rectangle([10, 10, w - 11, h - 11], outline=_rgb(dark, 255), width=8)
        d.rectangle([22, 22, w - 23, h - 23], outline=_rgb(dark, 180), width=4)
        # 中央の雲渦模様
        ts = np.linspace(0, 3.2 * np.pi, 90)
        pts = []
        for t in ts:
            r = 0.04 + 0.24 * (t / (3.2 * np.pi))
            pts.append((w * (0.54 + r * np.cos(t)), h * (0.52 + r * 1.15 * np.sin(t))))
        d.line(pts, fill=_rgb(dark, 235), width=14)
        d.ellipse([w * 0.26, h * 0.36, w * 0.52, h * 0.68], fill=_rgb(dark, 220))
        d.ellipse([w * 0.33, h * 0.43, w * 0.45, h * 0.58], fill=_rgb(gold, 255))
        return self.save(img, name)

    def solid(self, name, color, noise=0.0, pattern=None):
        """服の色。pattern(u, v) で模様（縁取りの線など）を足せる。"""
        n = 256
        v, u = np.mgrid[0:n, 0:n] / (n - 1)
        img = np.broadcast_to(np.asarray(color, float), (n, n, 3)).copy()
        if noise:
            rng = np.random.default_rng(len(name))
            g = Image.fromarray((rng.normal(0.5, 0.2, (n // 4, n // 4)).clip(0, 1) * 255).astype(np.uint8))
            g = np.asarray(g.resize((n, n), Image.BILINEAR), float) / 255 - 0.5
            img += g[:, :, None] * noise
        if pattern is not None:
            pattern(img, u, v)
        return self.save(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB"), name)

    def meander(self, name, bg, fg):
        """腰布の裾の白い雷文（四角い渦巻きが横に並ぶ帯）。"""
        w, h = 256, 128
        img = Image.new("RGB", (w, h), _rgb(bg)[:3])
        d = ImageDraw.Draw(img)
        c = _rgb(fg)[:3]
        lw = 10
        d.line([(0, 12), (w, 12)], fill=c, width=lw)
        d.line([(0, h - 12), (w, h - 12)], fill=c, width=lw)
        for x0 in (0, w // 2):
            pts = [(x0 + 12, h - 12), (x0 + 12, 32), (x0 + 108, 32), (x0 + 108, 96),
                   (x0 + 44, 96), (x0 + 44, 60), (x0 + 76, 60)]
            d.line(pts, fill=c, width=lw, joint="curve")
        return self.save(img, name)

    def flame(self, name, c_top, c_mid, c_tip):
        """裾の炎：上が赤、中がピンク、下が淡いピンクで、下端は炎の舌の形に透明へ抜ける。"""
        w, h = 512, 256
        v, u = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None]
        t1 = np.clip(v / 0.45, 0, 1)[..., None]
        t2 = np.clip((v - 0.45) / 0.45, 0, 1)[..., None]
        col = np.asarray(c_top) * (1 - t1) + np.asarray(c_mid) * t1
        col = col * (1 - t2) + np.asarray(c_tip) * t2
        wave = (0.55 + 0.22 * np.sin(u * 2 * np.pi * 4 + np.sin(v * 6) * 0.8)
                + 0.12 * np.sin(u * 2 * np.pi * 9 - 0.7) + 0.08 * np.cos(u * 2 * np.pi * 13 + 1.2))
        alpha = np.clip((wave - v) / 0.10, 0, 1)
        rgba = np.dstack([np.clip(col, 0, 1), alpha])
        return self.save(Image.fromarray((rgba * 255).astype(np.uint8), "RGBA"), name)

    def mark(self, name, text, color, bg=None, font="C:/Windows/Fonts/arialbd.ttf"):
        """服に貼る文字や模様（肩の OJ、胸の 炎）。bg=None なら背景は透明。"""
        n = 256
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0) if bg is None else _rgb(bg))
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(font, int(n * 0.72))
        bb = d.textbbox((0, 0), text, font=f)
        d.text(((n - (bb[2] - bb[0])) / 2 - bb[0], (n - (bb[3] - bb[1])) / 2 - bb[1]),
               text, font=f, fill=_rgb(color))
        return self.save(img, name)

    def yinyang(self, name="yinyang.png", c1=(0.36, 0.78, 0.60), c2=(0.96, 0.95, 0.93)):
        """結い髪の陰陽玉（太極図のS字勾玉模様と2つの目点）。"""
        n = 256
        py, px = np.mgrid[0:n, 0:n] / (n - 1) * 2.0 - 1.0
        r_top = np.hypot(px, py + 0.5)
        r_bot = np.hypot(px, py - 0.5)
        is_c1 = np.where(r_top < 0.5, True, np.where(r_bot < 0.5, False, px < 0))
        is_c1 = np.where(r_top < 0.16, False, np.where(r_bot < 0.16, True, is_c1))
        col = np.where(is_c1[..., None], np.asarray(c1, float), np.asarray(c2, float))
        return self.save(Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGB"), name)
