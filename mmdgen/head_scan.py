"""既製の美少女モデルの顔の肌を輪切りにして、頭の形の表 mmdgen/head_shape.py を書き出す。

python -m mmdgen.head_scan [uka|n4d]（既定は uka）

uka：CG アバター「うか」（https://github.com/mmdagent-ex/uka、CC-BY 4.0）を models/uka_ref/ に clone したもの。
  CG-CA Uka (c) 2023-2024 by Nagoya Institute of Technology, Moonshot R&D Goal 1 Avatar Symbiotic Society
n4d：Neural4D で作った models/tabane_roze_n4d/（再配布条件が不明なので手元だけ）。
どちらも gitignore で、リポジトリには表（数値）だけを入れる。

頭の中心軸から水平に光線を飛ばし、顔の肌（material_face）に当たるまでの距離を高さ・向きごとに測る。
長さは「目の線〜あご先」で割り、高さは目の線を 0（上が +）にする。左右は平均して左右対称にする。
頭は前から後ろまで一周（0〜180 度）、あごの下・首の付け根から頭頂まで測り、頭の肌はこの表だけで作る（body.Head）。
"""

from pathlib import Path

import numpy as np

from .inspect_pmx import read
from .measure import material_vertices

ROOT = Path(__file__).resolve().parent.parent
# 元のモデル：ファイル・顔の肌の材質・目の材質（白目と瞳。顔の肌の目の穴を埋めるのに使う）
SOURCES = {
    "uka": dict(path=ROOT / "models" / "uka_ref" / "MS_Uka.pmx", face=("頭",), eyes=("目",)),
    "n4d": dict(path=ROOT / "models" / "tabane_roze_n4d" / "tabane_roze_n4d.pmx",
                face=("material_face", "material_nose"), eyes=("material_eyewhites",), iris=("material_eyer", "material_eyel")),
}
OUT = Path(__file__).resolve().parent / "head_shape.py"

# 表の目盛り：高さ（目の線〜あご先 = 1、上が +）と、正面から測った角（度）
SCAN_V = np.round(np.arange(-1.50, 1.46, 0.02), 2)
SCAN_PHI = np.arange(0, 180.1, 2.5)
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


def _chin(tris, eye_y, top):
    """あご先の高さ：顔の真ん中の縦の線を目の線から下へたどり、前の面が首の方へ大きく引っ込む直前の高さ。
    頭の後ろから前へ光線を通し、いちばん前の肌までの距離で前の面を測る。"""
    span = top - eye_y
    back = tris[:, :, 2].max() + 1.0
    d = np.array([[0.0, 0.0, -1.0]])
    step = 0.005 * span
    prev, y = None, eye_y
    while y > tris[:, :, 1].min():
        near = tris[(tris[:, :, 1].min(1) <= y) & (tris[:, :, 1].max(1) >= y)]
        t = _hits(near, np.array([[0.0, y, back]]), d)[0] if len(near) else np.nan
        if not np.isfinite(t) and prev is not None:  # 首の肌がない（服で隠れている）
            return y + step
        if np.isfinite(t):
            z = back - t
            if prev is not None and z - prev > 0.08 * span:
                return y + step
            prev = z
        y -= step
    raise ValueError("あご先が見つからない")


def scan(key="uka"):
    src = SOURCES[key]
    info = read(src["path"])
    mv = material_vertices(info)
    eyes = np.vstack([mv[k] for k in src.get("iris", src["eyes"])])
    eye_y = eyes[:, 1].mean()
    face = np.vstack([mv[k] for k in src["face"]])
    # 顔の肌の三角形と、白目の三角形（顔の肌は目の所が穴）
    start, tris, whites = 0, [], []
    for m in info["materials"]:
        idx = info["indices"][start:start + m["faces"] * 3].reshape(-1, 3)
        start += m["faces"] * 3
        if m["name"] in src["face"]:
            tris.append(info["positions"][idx])
        elif m["name"] in src["eyes"]:
            whites.append(info["positions"][idx])
    tris, whites = np.vstack(tris), np.vstack(whites)
    chin_y = _chin(tris, eye_y, face[:, 1].max())
    L = eye_y - chin_y
    # 中心軸：目の高さの輪切りの前後の真ん中。首の中を通るので、あごの下から首の付け根まで同じ軸で測れる
    near = tris[(tris[:, :, 1].min(1) <= eye_y) & (tris[:, :, 1].max(1) >= eye_y)]
    far = 10.0 * L
    z_front = far - _hits(near, np.array([[0.0, eye_y, far]]), np.array([[0.0, 0.0, -1.0]]))[0]
    z_back = _hits(near, np.array([[0.0, eye_y, -far]]), np.array([[0.0, 0.0, 1.0]]))[0] - far
    zc = (z_front + z_back) / 2
    iris = eyes[eyes[:, 0] > 0]
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
    # 目の穴は、白目の丸い面（眼球）からの浮きを穴の縁の肌から 2 次元でなめらかに広げて埋め、白目の奥へ少しだけ下げる。
    # 行ごとに埋めると面がでこぼこし、上に貼る目の板がゆがむ。白目の板そのままだと深い穴になり、斜め・横から目の所がへこむ
    hole = ~np.isfinite(table) & np.isfinite(white)
    rim = np.isfinite(table) & np.isfinite(white)
    d = np.where(rim, table - white, 0.0)
    known = rim.copy()
    for _ in range(2000):
        pad = np.pad(d, 1, mode="edge")
        avg = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
        d = np.where(hole, avg, d)
    table = np.where(hole, white + (1 - SOCKET) * d, table)
    # 肌の三角形の縁や細かな段で面がでこぼこしないよう、測れた所だけで軽くならす
    ok = np.isfinite(table)
    t0 = np.where(ok, table, 0.0)
    k = np.array([0.25, 0.5, 0.25])
    def blur(T):
        T = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 1, mode="edge"), k, "valid"), 0, T)
        return np.apply_along_axis(lambda c: np.convolve(np.pad(c, 1, mode="edge"), k, "valid"), 1, T)
    w0 = ok.astype(float)
    for _ in range(3):  # 目盛りが細かい（2.5 度・0.02）ので 3 回ならす
        t0, w0 = blur(t0), blur(w0)
    sm = t0 / np.maximum(w0, 1e-9)
    table = np.where(ok, sm, np.nan)
    print(key, "目の線", round(eye_y, 3), "あご先", round(chin_y, 3), "頭頂", round(face[:, 1].max(), 3),
          "目の線の高さ（あご 0〜頭頂 1）", round(L / (face[:, 1].max() - chin_y), 3))
    return table, eye_yaw, (face[:, 1].max() - eye_y) / L


def main(key="uka"):
    table, eye_yaw, v_crown = scan(key)
    rows = ",\n    ".join("[" + ", ".join("nan" if not np.isfinite(x) else f"{x:.4f}" for x in row) + "]" for row in table)
    OUT.write_text(f'''"""頭の形の表（python -m mmdgen.head_scan で美少女モデルの顔の肌から測って書き出したもの。手で直さない）。

元のモデル：{key}（mmdgen/head_scan.py の SOURCES）

RHO[i][j]：目の線から高さ V[i]（目の線〜あご先 = 1、上が +）、正面から PHI[j] 度の向きでの、中心軸から肌までの距離
（同じく目の線〜あご先で割った値）。nan は顔の肌がなく測れない所（頭頂・後頭部）。
"""

from math import nan

EYE_YAW = {eye_yaw:.2f}  # 瞳の中心の向き（度）
V_CROWN = {v_crown:.4f}  # 頭頂の高さ（目の線から、目の線〜あご先 = 1）
V = {SCAN_V.tolist()}
PHI = {SCAN_PHI.tolist()}
RHO = [
    {rows},
]
''', encoding="utf-8")
    print(OUT, table.shape, "測れた点", int(np.isfinite(table).sum()))


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])
