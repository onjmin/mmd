"""書き出した PMX を正面から平行投影で描く（ブラウザなしで顔を見比べる用）。

python -m mmdgen.render <キー> [出力.png] [face|bare|full]（bare は髪と服を消して輪郭を見る）
材質を PMX の順に描き、テクスチャのアルファで重ねる。陰影は法線と正面上からの光で軽く付けるだけ。
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .inspect_pmx import read

ROOT = Path(__file__).resolve().parent.parent
BARE = {"肌", "白目", "瞳", "まつ毛", "眉", "口", "口内", "鼻", "頬"}


def render(path, view="face", size=800, yaw=0.0, clay=False, keep=None, frame=None):
    """yaw は体を上から見て左回りに回す角（度、正面 0・横顔 90）。clay は色を消して陰影だけで凹凸を見る。
    keep は描く材質名の集合（bare は BARE）。frame=(cx, cy, half) で描く範囲を直接決める。"""
    info = read(path)
    P, N, UV = info["positions"].copy(), info["normals"].copy(), info["uvs"]
    H = P[:, 1].max()
    if view in ("face", "bare"):
        cy, half = H * 0.865, H * 0.085
    else:
        cy, half = H * 0.5, H * 0.52
    cx = 0.0
    if frame is not None:
        cx, cy, half = frame
    if yaw:
        a = np.radians(yaw)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        pivot = np.array([0.0, cy, 0.0])
        P = (P - pivot) @ R.T + pivot
        N = N @ R.T
    W = Hh = size
    sx = lambda x: ((x - cx) / half * 0.5 + 0.5) * (W - 1)
    sy = lambda y: (0.5 - (y - cy) / half * 0.5) * (Hh - 1)
    X, Y = sx(P[:, 0]), sy(P[:, 1])
    Z = P[:, 2]  # 小さいほど手前（顔は -Z 向き）
    img = np.zeros((Hh, W, 3)) + np.array([0.86, 0.88, 0.92])
    zbuf = np.full((Hh, W), np.inf)
    light = np.array([-0.55, 0.45, -0.75]) if clay else np.array([0.0, 0.45, -1.0])  # 粘土は斜め上から当てて凹凸を出す
    light /= np.linalg.norm(light)
    base = Path(path).parent
    start = 0
    for m in info["materials"]:
        idx = info["indices"][start:start + m["faces"] * 3].reshape(-1, 3)
        start += m["faces"] * 3
        if (view == "bare" and keep is None and m["name"] not in BARE) or (keep is not None and m["name"] not in keep):
            continue
        tex = None
        if m["tex"]:
            tp = base / m["tex"].replace("\\", "/")
            if tp.exists():
                tex = np.asarray(Image.open(tp).convert("RGBA"), float) / 255
        dif = np.asarray(m["diffuse"][:3])
        if clay:
            tex, dif = None, np.array([0.85, 0.82, 0.80])
        alpha_m = m["diffuse"][3]
        if alpha_m <= 0:
            continue
        flat = "白目" in m["name"] or "瞳" in m["name"] or "まつ" in m["name"] or "眉" in m["name"] or "口" in m["name"] \
            or "鼻" in m["name"] or "頬" in m["name"]
        flat = flat and not clay
        for f in idx:
            xs, ys = X[f], Y[f]
            x0, x1 = int(max(0, np.floor(xs.min()))), int(min(W - 1, np.ceil(xs.max())))
            y0, y1 = int(max(0, np.floor(ys.min()))), int(min(Hh - 1, np.ceil(ys.max())))
            if x0 > x1 or y0 > y1:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, bx, cx_), (ay, by, cy_) = xs, ys
            d = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
            if abs(d) < 1e-9:
                continue
            l0 = ((by - cy_) * (gx - cx_) + (cx_ - bx) * (gy - cy_)) / d
            l1 = ((cy_ - ay) * (gx - cx_) + (ax - cx_) * (gy - cy_)) / d
            l2 = 1 - l0 - l1
            ins = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
            if not ins.any():
                continue
            z = l0 * Z[f[0]] + l1 * Z[f[1]] + l2 * Z[f[2]]
            zb = zbuf[y0:y1 + 1, x0:x1 + 1]
            ok = ins & (z <= zb + 1e-4)
            if not ok.any():
                continue
            col = np.broadcast_to(dif, ok.shape + (3,)).copy()
            a = np.full(ok.shape, alpha_m)
            if tex is not None:
                u = l0 * UV[f[0], 0] + l1 * UV[f[1], 0] + l2 * UV[f[2], 0]
                v = l0 * UV[f[0], 1] + l1 * UV[f[1], 1] + l2 * UV[f[2], 1]
                th, tw = tex.shape[:2]
                ti = tex[np.clip((v % 1.0001) * (th - 1), 0, th - 1).astype(int), np.clip((u % 1.0001) * (tw - 1), 0, tw - 1).astype(int)]
                col = ti[..., :3] if flat else col * ti[..., :3]
                a = a * ti[..., 3]
            if not flat:
                n = l0[..., None] * N[f[0]] + l1[..., None] * N[f[1]] + l2[..., None] * N[f[2]]
                n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
                sh = np.clip(n @ light, 0, 1)
                if clay:
                    col = col * (0.25 + 0.75 * sh)[..., None]
                else:
                    col = col * (0.72 + 0.28 * np.clip((sh - 0.15) / 0.2, 0, 1))[..., None]
            ok &= a > 0.02
            sub = img[y0:y1 + 1, x0:x1 + 1]
            sub[ok] = sub[ok] * (1 - a[ok, None]) + col[ok] * a[ok, None]
            # 不透明な所だけ奥行きを書く（半透明の板は下を隠さない）
            zw = ok & (a > 0.5)
            zb[zw] = np.minimum(zb[zw], z[zw])
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    key = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"{key}_{sys.argv[3] if len(sys.argv) > 3 else 'face'}.png"
    view = sys.argv[3] if len(sys.argv) > 3 else "face"
    render(ROOT / "models" / key / f"{key}.pmx", view).save(out)
    print(out)
