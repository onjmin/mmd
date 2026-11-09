"""Pillow で描くテクスチャ：目・口・鼻・へそ・髪・トゥーン・スフィア。モデルと同じフォルダの tex/ に書き出す。"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 4  # 描いてから縮小してアンチエイリアスをかける倍率


def _rgb(c, a=255):
    return tuple(int(max(0, min(1, x)) * 255) for x in c) + (int(a),)


def _mix(a, b, t):
    return tuple(np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t)


# ---- 目の形：白目・瞳・まつ毛が同じ輪郭を共有する ----
# 座標 (A, B) は目の板の上の位置。A は rx 単位で目頭 -1 → 目尻 +1、B は ry 単位で上が +。
# 白目の板は A, B とも ±1、絵の輪郭はその 92%（measure.py が見える目の端をそう数える）。
# 既製モデル（Tda式ミク系）の目は丸ではなく、上が広く下が狭い台形：
# 上の縁はほぼまっすぐで目尻へ少し上がり、目尻の上に鋭い角。外側の縁はそこから内へ斜めに短く下り、
# 下の縁は目尻へ向かって上がる（目頭側の縦の縁は長く、目尻側は短い）。目頭は上の縁の端で尖る。
# 横:縦はおよそ 2:1（rx/ry と合わせて決める）。同じ点を 2 回置いた所は角として残る（_chaikin が角を丸めきらない）。
EYE_CTRL = [(-0.94, 0.55), (-0.80, 0.76), (-0.45, 0.88), (0.10, 0.93), (0.55, 0.96), (0.88, 0.95), (0.88, 0.95),
            (0.80, 0.50), (0.68, 0.00), (0.58, -0.45), (0.46, -0.66), (0.20, -0.80), (-0.15, -0.90),
            (-0.52, -0.96), (-0.76, -0.93), (-0.87, -0.62), (-0.92, -0.10), (-0.94, 0.55)]
EYE_CORNER = (0.88, 0.95)  # 目尻の上の角（まつ毛がいちばん太く、ハネが出る所）


def _chaikin(p, it=4):
    """端点を動かさない角切り。目頭（始点＝終点）だけ尖ったまま残る。"""
    p = np.asarray(p, float)
    for _ in range(it):
        q = np.empty((2 * len(p) - 2, 2))
        q[0::2] = 0.75 * p[:-1] + 0.25 * p[1:]
        q[1::2] = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def eye_outline():
    return _chaikin(EYE_CTRL)


def eye_top_arc(k):
    """上まつ毛の通り道：目頭の角の少し先（細く尖って終わる所）から上の縁を通り、
    目尻の角を回って外側の縁を半分ほど下まで包む。"""
    o = eye_outline()
    end = np.argmax((o[:, 0] > 0.4) & (o[:, 1] < -0.25))
    arc = o[:end + 1]
    d = (arc[0] - arc[3]) * [k, 1.0]
    tail = arc[0] + d / np.linalg.norm(d) * 0.12 / [k, 1.0]
    return np.vstack([tail, arc])


def _offset(arc, th, k):
    """arc を外側（目の外）へ th だけずらした線。th は ry 単位、A は k=rx/ry 倍して実寸で法線を取る。"""
    P = arc * [k, 1.0]
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    return (P + nrm * np.asarray(th)[:, None]) / [k, 1.0]


def _to_px(ext, n):
    A0, A1, B0, B1 = ext
    return lambda pts: [((a - A0) / (A1 - A0) * (n - 1), (B1 - b) / (B1 - B0) * (n - 1)) for a, b in pts]


def _mask(poly_px, n):
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).polygon(poly_px, fill=255)
    return np.asarray(img, float) / 255


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

    # ---- 目：白目、瞳、まつ毛の 3 枚（既製MMDモデル・立ち絵の形状・配色に準拠） ----
    def eye_white(self, ext, k, name="eye_white.png"):
        """白目：共通の輪郭を塗りつぶす。上はまつ毛の落ち影で青みの灰色、下の縁だけ淡く締める。
        内側の白目は瞳が目頭の際まで寄るので、ここでは形だけ用意すれば細い筋しか見えない。"""
        n = 256 * S
        to_px = _to_px(ext, n)
        m = _mask(to_px(eye_outline()), n)
        # 各列の白目の上端からの深さ（上まつ毛の影）と下端までの近さ
        rows = np.arange(n)[:, None]
        inside = m > 0.5
        top = np.where(inside.any(0), inside.argmax(0), n)[None, :]
        bot = np.where(inside.any(0), n - 1 - inside[::-1].argmax(0), 0)[None, :]
        px = (n - 1) / (ext[3] - ext[2])  # ry 1 つぶんの画素数
        depth, near_bot = (rows - top) / px, (bot - rows) / px
        shade = np.clip(1 - depth / 0.90, 0, 1) ** 1.3
        col = np.array([1.0, 1.0, 1.0]) * (1 - shade[..., None]) + np.array([0.70, 0.72, 0.84]) * shade[..., None]
        edge = np.clip(1 - near_bot / 0.14, 0, 1) ** 2 * 0.35
        col = col * (1 - edge[..., None]) + np.array([0.70, 0.66, 0.70]) * edge[..., None]
        img = Image.fromarray((np.dstack([np.clip(col, 0, 1), m]) * 255).astype(np.uint8), "RGBA")
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def iris(self, name, color, ext, k):
        """瞳：縦長の楕円（板が縦長なので円で描けば縦長に貼られる）。
        上 4 割はまぶたの影で濃く、下へ明るく抜ける。瞳孔・濃い外周・左上の大きな光と右下の小さな光。
        正面を向いたときの白目の輪郭で切り抜き、目頭の外の肌に瞳がはみ出さないようにする。"""
        n = 256 * S
        color = np.asarray(color, float)
        y, x = np.mgrid[0:n, 0:n] / (n - 1)
        dy = (y - 0.50) / 0.47
        dx = (x - 0.50) / 0.47 + 0.12 * dy  # 上ほど目尻側へ少し傾ける（上の縁の傾きに合わせる）
        r = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)

        top_shade = np.clip(color * 0.34 + 0.03, 0.0, 1.0)
        mid_col = np.clip(color * 0.80 + 0.04, 0.0, 1.0)
        bright_col = np.clip(color * 1.25 + 0.38, 0.0, 1.0)
        rim_col = np.clip(color * 0.22 + np.array([0.02, 0.03, 0.10]), 0.0, 1.0)

        t_upper = np.clip((y - 0.32) / 0.26, 0.0, 1.0)
        t_upper = t_upper * t_upper * (3 - 2 * t_upper)
        col = top_shade * (1 - t_upper[..., None]) + mid_col * t_upper[..., None]
        # 下半分の三日月形の明るみ
        crescent = np.clip((y - 0.50) / 0.30, 0.0, 1.0) * np.exp(-((r - 0.62) / 0.22) ** 2)
        col = col * (1 - crescent[..., None]) + bright_col * crescent[..., None]
        rays = (0.5 + 0.5 * np.cos(ang * 26.0)) * np.exp(-((r - 0.55) / 0.20) ** 2) * np.clip((y - 0.42) / 0.40, 0.0, 1.0)
        col = np.clip(col + rays[..., None] * (bright_col - mid_col) * 0.20, 0.0, 1.0)

        # 瞳孔（縦長）とまわりの淡い輪
        pr = np.sqrt((dx / 0.19) ** 2 + ((dy + 0.02) / 0.25) ** 2)
        halo = np.exp(-((pr - 1.25) / 0.30) ** 2) * np.clip((y - 0.38) / 0.40, 0.0, 1.0)
        col = col * (1 - 0.30 * halo[..., None]) + bright_col * (0.30 * halo[..., None])
        core = np.clip((1.0 - pr) * 14.0, 0.0, 1.0)[..., None]
        col = col * (1 - 0.85 * core) + np.clip(color * 0.12 + 0.02, 0, 1) * (0.85 * core)

        # 外周のくっきりした濃い縁
        rim = np.clip((r - 0.90) * 30.0, 0.0, 1.0)[..., None]
        col = col * (1 - rim) + rim_col * rim

        alpha = np.clip((1.0 - r) * 40.0, 0.0, 1.0)
        # 白目の輪郭で切り抜く（少しだけ広げて、まつ毛の下で途切れないように）
        out = _offset(eye_outline(), np.full(len(eye_outline()), 0.04), k)
        alpha = alpha * _mask(_to_px(ext, n)(out), n)
        img = Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(img)
        # 光：板の上部はまつ毛に隠れるので、見える高さに置く
        d.ellipse([n * 0.12, n * 0.33, n * 0.42, n * 0.45], fill=(255, 255, 255))  # まつ毛の際の横長の光
        d.ellipse([n * 0.60, n * 0.66, n * 0.74, n * 0.78], fill=(250, 250, 250))
        d.ellipse([n * 0.66, n * 0.35, n * 0.74, n * 0.41], fill=(235, 235, 235))
        img.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
        return self.save(img.resize((256, 256), Image.LANCZOS), name)

    def lash(self, ext, k, name="lash.png", color=(0.05, 0.04, 0.05)):
        """上まつ毛：白目の上の縁に沿う帯。太さは白目の高さの 2 割弱で、
        目頭の少し上で細い鉤として始まり、上の縁で太く、目尻の角でいちばん太くなって外側の縁を下の方まで包んで細る。
        目尻の角から外へ短いハネを 1 本。上に細い二重の線、下まぶたは目尻側の半分だけ淡い茶の線。"""
        n = 512 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        to_px = _to_px(ext, n)
        arc = eye_top_arc(k)
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(arc * [k, 1], axis=0), axis=1))])
        s /= s[-1]
        ic = int(np.argmin(np.linalg.norm(arc - EYE_CORNER, axis=1)))
        sc = s[ic]  # 目尻の角
        th = np.interp(s, [0.00, 0.06, 0.15, 0.32, sc - 0.08, sc, sc + (1 - sc) * 0.35, sc + (1 - sc) * 0.7, 1.00],
                          [0.00, 0.17, 0.27, 0.32, 0.34, 0.40, 0.16, 0.07, 0.00])
        # 白目の縁に少しかぶせて隙間を出さない（両端は細く尖らせる）
        inner = _offset(arc, -0.06 * np.clip(th / 0.12, 0, 1), k)
        outer = _offset(arc, th, k)
        band = [tuple(p) for p in outer] + [tuple(p) for p in inner[::-1]]
        edge = _rgb(_mix(color, (0.30, 0.18, 0.17), 0.55))
        d.polygon(to_px(band), fill=edge)
        core_o = _offset(arc, th * 0.70, k)
        core_i = _offset(arc, -0.04 * np.clip(th / 0.12, 0, 1), k)
        sel = (s > 0.03) & (s < 0.97)
        d.polygon(to_px([tuple(p) for p in core_o[sel][::-1]] + [tuple(p) for p in core_i[sel]]), fill=_rgb(color))

        # 目尻のハネ：上の縁が外側へ下りはじめる角から、横へ細く尖らせる
        i0, i1, im = (int(np.argmin(np.abs(s - t))) for t in (sc - 0.07, sc + 0.06, sc))
        tip = outer[im] + np.array([0.34 / k, 0.08])
        base0, base1 = outer[i0] - (outer[i0] - arc[i0]) * 0.3, outer[i1] - (outer[i1] - arc[i1]) * 0.3
        q = np.linspace(0, 1, 12)[:, None]
        up = (1 - q) ** 2 * base0 + 2 * q * (1 - q) * ((base0 + tip) / 2 + [0.0, 0.03]) + q ** 2 * tip
        dn = (1 - q) ** 2 * tip + 2 * q * (1 - q) * ((tip + base1) / 2 + [-0.04, 0.0]) + q ** 2 * base1
        d.polygon(to_px([tuple(p) for p in np.vstack([up, dn])]), fill=_rgb(color))

        # 二重の線：まつ毛の上、目頭寄りから目尻手前まで細く
        sel = (s > 0.22) & (s < sc - 0.03)
        crease = _offset(arc, th + 0.24, k)[sel]
        w = np.sin(np.linspace(0, np.pi, len(crease))) ** 0.6
        for a, b, ww in zip(crease[:-1], crease[1:], w[:-1]):
            d.line(to_px([tuple(a), tuple(b)]), fill=_rgb((0.36, 0.22, 0.20), 180), width=max(1, int(n * 0.007 * (0.4 + 0.6 * ww))))

        # 下まぶた：目尻側の下の縁に沿う淡い茶の線（目頭側は描かない）
        o = eye_outline()
        lo = o[(o[:, 1] < -0.62) & (o[:, 0] > 0.0) & (o[:, 0] < 0.55)]
        lo = lo[np.argsort(-lo[:, 0])]
        w = np.sin(np.linspace(0, np.pi, len(lo))) ** 0.8
        for a, b, ww in zip(lo[:-1], lo[1:], w[:-1]):
            d.line(to_px([tuple(a), tuple(b)]), fill=_rgb((0.46, 0.28, 0.26), int(185 * ww)), width=max(1, int(n * 0.011 * ww)))
        return self.save(img.resize((n // S, n // S), Image.LANCZOS), name)

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
        """既製MMDモデル準拠の控えめで可愛らしい鼻先ハイライト＆淡い立体影点。"""
        n = 128 * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        sc = np.asarray(skin_color, float)
        # 鼻先の下のほんのり淡いピーチ影
        shade_col = _rgb(sc * np.array([0.85, 0.70, 0.66]), 60)
        d.ellipse([n * 0.38, n * 0.54, n * 0.62, n * 0.80], fill=shade_col)
        img = img.filter(ImageFilter.GaussianBlur(n * 0.05))
        d = ImageDraw.Draw(img)
        # 鼻先の小さなチョン点（濃すぎない柔らかい暖色ブラウン）
        line_col = _rgb(sc * np.array([0.52, 0.34, 0.30]), 165)
        stroke = [
            (n * 0.48, n * 0.47),
            (n * 0.52, n * 0.50),
            (n * 0.52, n * 0.66),
            (n * 0.49, n * 0.73),
            (n * 0.47, n * 0.64),
        ]
        d.polygon(stroke, fill=line_col)
        return self.save(img.resize((128, 128), Image.LANCZOS), name)

    def blush(self, name="blush.png", color=(1.0, 0.46, 0.50)):
        """ふんわりぼかしたピーチピンクのチーク。"""
        w, h = 256, 128
        y, x = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None] * 2 - 1
        r2 = x * x + (y * 1.15) ** 2
        a = np.clip(1.0 - r2, 0.0, 1.0) ** 2.2 * 0.24
        col = np.broadcast_to(np.asarray(color, float), (h, w, 3)).copy()
        arr = np.dstack([np.clip(col, 0, 1), np.clip(a, 0, 1)])
        return self.save(Image.fromarray((arr * 255).astype(np.uint8), "RGBA"), name)

    def mouth_line(self, name="mouth_line.png", color=(0.48, 0.22, 0.22)):
        """既製MMDモデル準拠の優しく微笑む可愛らしいリップライン（口角がほんのり上がり、唇に淡い血色ティントが乗る）。"""
        w, h = 256 * S, 128 * S
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        lip_tint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dt = ImageDraw.Draw(lip_tint)
        dt.ellipse([w * 0.30, h * 0.48, w * 0.70, h * 0.72], fill=_rgb((0.96, 0.64, 0.62), 55))
        lip_tint = lip_tint.filter(ImageFilter.GaussianBlur(h * 0.05))
        img = Image.alpha_composite(img, lip_tint)
        d = ImageDraw.Draw(img)

        pts_top, pts_bot = [], []
        for t in np.linspace(0, 1, 60):
            u = 0.06 + 0.88 * t
            x = (t - 0.5) * 2.0
            # 口角がキュッと優しく上がる美少女の微笑みアーチ
            y = 0.55 - 0.11 * (x ** 2) + 0.02 * np.cos(np.pi * x)
            env = np.sin(np.pi * t) ** 0.55
            th = (0.058 + 0.016 * np.exp(-((x / 0.35) ** 2))) * env
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
