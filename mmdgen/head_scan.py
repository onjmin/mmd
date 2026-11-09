"""美少女モデル（models/tabane_roze_n4d/、gitignore）の顔の肌を輪切りにして、頭の形の表 mmdgen/head_shape.py を書き出す。

python -m mmdgen.head_scan

頭の中心軸から水平に光線を飛ばし、顔の肌（material_face）に当たるまでの距離を高さ・向きごとに測る。
長さは「目の線〜あご先」で割り、高さは目の線を 0（上が +）にする。左右は平均して左右対称にする。
目のくぼみ・鼻筋・口元・あご先の凹凸はこの表から作る（body.Head）。後頭部や頭頂は顔の肌がないので測らない。
"""

from pathlib import Path

import numpy as np

from .inspect_pmx import read
from .measure import material_vertices

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "tabane_roze_n4d" / "tabane_roze_n4d.pmx"
OUT = Path(__file__).resolve().parent / "head_shape.py"

# 表の目盛り：高さ（目の線〜あご先 = 1、上が +）と、正面から測った角（度）
SCAN_V = np.round(np.arange(-1.10, 1.46, 0.05), 2)
SCAN_PHI = np.arange(0, 96, 5)
SOCKET = 0.35  # 目の穴を白目の奥へ下げる割合（0 で肌どうしを平らにつなぐ、1 で白目の板そのまま）


def _hits(tri, org, dirs):
    """水平な光線（org から dirs へ）が三角形に当たる距離の最大値（当たらなければ nan）。"""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.full(len(dirs), np.nan)
    for k, d in enumerate(dirs):
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = org[k] - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if hit.any():
            out[k] = t[hit].max()
    return out


def scan():
    info = read(SRC)
    mv = material_vertices(info)
    eye_y = np.vstack([mv["material_eyer"], mv["material_eyel"]])[:, 1].mean()
    face = mv["material_face"]
    front = face[np.abs(face[:, 0]) < 0.05]
    chin_y = front[front[:, 2] < np.percentile(front[:, 2], 60), 1].min()
    L = eye_y - chin_y
    # 顔の肌・鼻の三角形と、白目の三角形（顔の肌は目の所が穴）
    start, tris, whites = 0, [], []
    for m in info["materials"]:
        idx = info["indices"][start:start + m["faces"] * 3].reshape(-1, 3)
        start += m["faces"] * 3
        if m["name"] in ("material_face", "material_nose"):
            tris.append(info["positions"][idx])
        elif m["name"] == "material_eyewhites":
            whites.append(info["positions"][idx])
    tris, whites = np.vstack(tris), np.vstack(whites)
    # 中心軸：目の高さで顔がいちばん横に広い所の奥行き（耳は除く）。ここで測った 90 度の値がそのまま頭の横幅になる
    band = face[(np.abs(face[:, 1] - eye_y) < 0.03 * L) & (face[:, 2] < np.percentile(face[:, 2], 70))]
    zc = band[np.argmax(np.abs(band[:, 0])), 2]
    iris = np.vstack([mv["material_eyer"], mv["material_eyel"]])
    iris = iris[iris[:, 0] > 0]
    eye_yaw = float(np.degrees(np.arctan2(iris[:, 0].mean(), zc - iris[:, 2].mean())))
    def measure(T):
        table = np.full((len(SCAN_V), len(SCAN_PHI)), np.nan)
        for i, v in enumerate(SCAN_V):
            y = eye_y + v * L
            near = T[(T[:, :, 1].min(1) <= y + 1e-6) & (T[:, :, 1].max(1) >= y - 1e-6)]
            if not len(near):
                continue
            for j, ph in enumerate(SCAN_PHI):
                r = []
                for sg in (1, -1):
                    a = np.radians(ph) * sg
                    d = np.array([np.sin(a), 0.0, -np.cos(a)])
                    r.append(_hits(near, np.array([[0.0, y, zc]]), d[None, :])[0])
                r = [x for x in r if np.isfinite(x)]
                if r:
                    table[i, j] = np.mean(r) / L
        return table
    table, white = measure(tris), measure(whites)
    # 目の穴は、まわりの肌を向きにそってつないだ面から、白目の奥へ少しだけ下げる。
    # 白目の板そのままだと深い穴になり、斜め・横から見たとき目の所が大きくへこんで見える
    for i in range(len(SCAN_V)):
        row, ok = table[i], np.isfinite(table[i])
        hole = ~ok & np.isfinite(white[i])
        if hole.any() and ok.sum() >= 2:
            fill = np.interp(SCAN_PHI, SCAN_PHI[ok], row[ok])
            row[hole] = fill[hole] - SOCKET * np.clip(fill[hole] - white[i][hole], 0, None)
    # 肌の三角形の縁や細かな段で面がでこぼこしないよう、測れた所だけで軽くならす
    ok = np.isfinite(table)
    t0 = np.where(ok, table, 0.0)
    k = np.array([0.25, 0.5, 0.25])
    def blur(T):
        T = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 1, mode="edge"), k, "valid"), 0, T)
        return np.apply_along_axis(lambda c: np.convolve(np.pad(c, 1, mode="edge"), k, "valid"), 1, T)
    sm = blur(t0) / np.maximum(blur(ok.astype(float)), 1e-9)
    table = np.where(ok, sm, np.nan)
    return table, eye_yaw


def main():
    table, eye_yaw = scan()
    rows = ",\n    ".join("[" + ", ".join("nan" if not np.isfinite(x) else f"{x:.4f}" for x in row) + "]" for row in table)
    OUT.write_text(f'''"""頭の形の表（python -m mmdgen.head_scan で美少女モデルの顔の肌から測って書き出したもの。手で直さない）。

RHO[i][j]：目の線から高さ V[i]（目の線〜あご先 = 1、上が +）、正面から PHI[j] 度の向きでの、中心軸から肌までの距離
（同じく目の線〜あご先で割った値）。nan は顔の肌がなく測れない所（頭頂・後頭部）。
"""

from math import nan

EYE_YAW = {eye_yaw:.2f}  # 瞳の中心の向き（度）
V = {SCAN_V.tolist()}
PHI = {SCAN_PHI.tolist()}
RHO = [
    {rows},
]
''', encoding="utf-8")
    print(OUT, table.shape, "測れた点", int(np.isfinite(table).sum()))


if __name__ == "__main__":
    main()
