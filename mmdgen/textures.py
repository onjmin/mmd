"""Pillow で描くテクスチャ：目・髪・トゥーン・スフィア。モデルと同じフォルダの tex/ に書き出す。"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 4  # 描いてから縮小してアンチエイリアスをかける倍率


def _rgb(c, a=255):
    return tuple(int(max(0, min(1, x)) * 255) for x in c) + (a,)


def _mix(a, b, t):
    return tuple(np.asarray(a) * (1 - t) + np.asarray(b) * t)


class TextureSet:
    def __init__(self, out_dir):
        self.dir = os.path.join(out_dir, "tex")
        os.makedirs(self.dir, exist_ok=True)

    def save(self, img, name):
        img.save(os.path.join(self.dir, name))
        return "tex\\" + name

    # ---- トゥーン（上が光、下が影。2 段で境目は少しぼかす） ----
    def toon(self, name, shadow):
        h = 32
        y = np.linspace(0, 1, h)
        t = np.clip((y - 0.42) / 0.12, 0, 1)
        col = np.outer(1 - t, [1, 1, 1]) + np.outer(t, shadow)
        img = np.repeat((col * 255).astype(np.uint8)[:, None, :], 32, axis=1)
        return self.save(Image.fromarray(img, "RGB"), name)

    # ---- 髪のつや（加算スフィア：上寄りに細い光の帯） ----
    def hair_sphere(self, name="hair_sphere.png", strength=0.55):
        n = 256
        y, x = np.mgrid[0:n, 0:n] / (n - 1) * 2 - 1
        r = np.sqrt(x * x + y * y)
        band = np.exp(-((r - 0.62) / 0.10) ** 2) * np.clip(-y + 0.15, 0, 1) ** 0.7
        v = (band * strength * 255).astype(np.uint8)
        return self.save(Image.fromarray(np.stack([v, v, v], -1), "RGB"), name)

    # ---- 髪（束に沿った筋と、根元の影・毛先の明るさ） ----
    def hair(self, name, base, tip=None, tip_from=0.75):
        w, h = 128, 512
        rng = np.random.default_rng(len(name))
        u = np.linspace(0, 1, w)
        stripe = np.zeros(w)
        for _ in range(28):
            c, s, a = rng.uniform(0, 1), rng.uniform(0.004, 0.02), rng.uniform(-0.16, 0.22)
            stripe += a * np.exp(-((u - c) / s) ** 2)
        stripe += 0.015 * np.cos(u * np.pi * 2)  # 束の端をごく少し暗く
        v = np.linspace(0, 1, h)
        root = 0.82 + 0.18 * np.clip(v / 0.25, 0, 1)
        img = np.zeros((h, w, 3))
        base = np.asarray(base, float)
        for i in range(h):
            col = base if tip is None else _mix(base, tip, np.clip((v[i] - tip_from) / (1 - tip_from), 0, 1))
            img[i] = np.outer(1 + stripe, col) * root[i]
        return self.save(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGB"), name)

    # ---- 目：白目、瞳、まつ毛の 3 枚 ----
    def eye_white(self, name="eye_white.png"):
        n = 128 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([n * 0.04, n * 0.06, n * 0.96, n * 0.98], fill=_rgb((1, 1, 1)))
        shade = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        ds = ImageDraw.Draw(shade)
        ds.ellipse([n * 0.0, -n * 0.35, n * 1.0, n * 0.30], fill=_rgb((0.80, 0.82, 0.92), 55))
        shade = shade.filter(ImageFilter.GaussianBlur(n * 0.06))
        img = Image.alpha_composite(img, Image.composite(shade, Image.new("RGBA", (n, n)), img))
        return self.save(img.resize((128, 128), Image.LANCZOS), name)

    def iris(self, name, color):
        """上（まぶたの影）が暗く下が明るい瞳。瞳孔は小さめで、光は右上に大、左下に小。"""
        n = 256 * S
        color = np.asarray(color, float)
        y, x = np.mgrid[0:n, 0:n] / (n - 1)
        r = np.sqrt((x - 0.5) ** 2 + ((y - 0.5) * 1.0) ** 2) / 0.5
        dark = color * 0.28
        light = np.clip(color * 1.30 + 0.15, 0, 1)
        t = np.clip((y - 0.20) / 0.70, 0, 1)[..., None] ** 1.2
        col = dark * (1 - t) + light * t
        col = col * (1 - 0.45 * np.clip((r - 0.82) / 0.18, 0, 1))[..., None]  # 縁だけ少し締める
        # 下側の明るい三日月
        glow = np.clip(1 - np.abs(r - 0.62) / 0.16, 0, 1) * np.clip((y - 0.55) / 0.3, 0, 1)
        col = np.clip(col + glow[..., None] * 0.25 * (light - col + 0.1), 0, 1)
        alpha = np.clip((1.0 - r) * n * 0.02, 0, 1)
        img = Image.fromarray((np.dstack([col, alpha]) * 255).astype(np.uint8), "RGBA")
        d = ImageDraw.Draw(img)
        d.ellipse([n * 0.38, n * 0.33, n * 0.62, n * 0.62], fill=_rgb(color * 0.10, 235))  # 瞳孔
        d.ellipse([n * 0.54, n * 0.16, n * 0.80, n * 0.40], fill=_rgb((1, 1, 1)))
        d.ellipse([n * 0.24, n * 0.64, n * 0.36, n * 0.75], fill=_rgb((1, 1, 1), 210))
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def lash(self, name="lash.png", color=(0.10, 0.06, 0.07)):
        """上まつ毛（太い弧＋目尻の跳ね）と下まつ毛の細い線。左目用で、右目は UV を反転して使う。"""
        n = 256 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c = _rgb(color)
        # 上まつ毛：外側（右）ほど太く、目尻で跳ねる
        pts_top, pts_bot = [], []
        for t in np.linspace(0, 1, 60):
            a = np.pi * (0.97 - 0.92 * t)
            x = 0.5 + 0.47 * np.cos(a)
            y = 0.52 - 0.44 * np.sin(a)
            th = 0.085 + 0.075 * t ** 1.5  # 太く下へ伸ばし、瞳の上にかぶせる
            pts_top.append((x * n, (y - th * 0.3) * n))
            pts_bot.append((x * n, (y + th) * n))
        flick = [(0.99 * n, 0.36 * n), (1.0 * n, 0.44 * n), (0.95 * n, 0.52 * n)]
        d.polygon(pts_top + flick + pts_bot[::-1], fill=c)
        # 二重まぶたの線
        d.arc([0.14 * n, 0.00 * n, 0.90 * n, 0.62 * n], 205, 320, fill=_rgb(color, 110), width=int(n * 0.010))
        # 下まつ毛
        d.arc([0.18 * n, 0.30 * n, 0.90 * n, 0.98 * n], 25, 150, fill=_rgb(color, 200), width=int(n * 0.012))
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def brow(self, name, color):
        n = 128 * S
        img = Image.new("RGBA", (n * 2, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        pts_top, pts_bot = [], []
        for t in np.linspace(0, 1, 40):
            x = 0.06 + 0.88 * t
            y = 0.55 - 0.25 * np.sin(np.pi * (0.25 + 0.65 * t))
            th = 0.16 * (1 - t) ** 0.6 + 0.03
            pts_top.append((x * n * 2, (y - th / 2) * n))
            pts_bot.append((x * n * 2, (y + th / 2) * n))
        d.polygon(pts_top + pts_bot[::-1], fill=_rgb(color))
        return self.save(img.resize((256, 128), Image.LANCZOS), name)

    def blush(self, name="blush.png", color=(1.0, 0.45, 0.50)):
        n = 128
        y, x = np.mgrid[0:n, 0:n] / (n - 1) * 2 - 1
        a = np.clip(1 - (x * x + (y * 1.0) ** 2), 0, 1) ** 1.6 * 0.55
        lines = (np.abs(((x + y * 0.6) * 3.5) % 1 - 0.5) < 0.07) & (np.abs(x) < 0.55) & (np.abs(y) < 0.35)
        a = np.maximum(a, lines * 0.55)
        arr = np.dstack([np.full((n, n), c) for c in color] + [a])
        return self.save(Image.fromarray((arr * 255).astype(np.uint8), "RGBA"), name)

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
