"""書き出した PMX を正面から見たときの顔と体の比率を測る。python -m mmdgen.measure <キー> [...]

立ち絵で測った値（checklists/<キー>.md の「比率」表）と同じ定義で並べ、外れた項目に印を付ける。
長さはすべて「頭の高さ（髪のてっぺん〜あご）」か「目の幅」で割った比にする。
顔の幅は絵では髪に隠れて定義がぶれるので使わない。
"""

import re
import sys
from pathlib import Path

import numpy as np

from .inspect_pmx import read

ROOT = Path(__file__).resolve().parent.parent


def material_vertices(info):
    """材質名 → その材質の面が使う頂点の座標。"""
    out, start = {}, 0
    for m in info["materials"]:
        idx = info["indices"][start:start + m["faces"] * 3]
        start += m["faces"] * 3
        v = info["positions"][np.unique(idx)]
        out[m["name"]] = np.vstack([out[m["name"]], v]) if m["name"] in out else v
    return out


def measure(path):
    info = read(path)
    info["positions"] = info["positions"] / info["positions"][:, 1].max()  # 身長 1 にそろえる（生成時の座標と同じ）
    mv = material_vertices(info)
    allv = info["positions"]
    skin, hair = mv["肌"], np.vstack([v for k, v in mv.items() if k.startswith("髪")])
    iris, white = mv["瞳"], mv["白目"]
    top = hair[np.abs(hair[:, 0]) < 0.015, 1].max()  # 頭頂の真ん中（左右の跳ね毛や飾りは数えない）
    eye_y = iris[:, 1].mean()
    face = skin[(skin[:, 2] < -0.035) & (skin[:, 1] > eye_y - 0.10) & (np.abs(skin[:, 0]) < 0.02)]
    chin = face[:, 1].min()
    head = top - chin
    eyes = {}
    for side, sel_i, sel_w in [("L", iris[:, 0] > 0, white[:, 0] > 0), ("R", iris[:, 0] < 0, white[:, 0] < 0)]:
        i, w = iris[sel_i], white[sel_w]
        # 白目の板は絵の楕円より 4% ずつ広い（textures.eye_white）ので、見える目の端に詰める
        a0, a1 = np.abs(w[:, 0]).min(), np.abs(w[:, 0]).max()
        mid, half = (a0 + a1) / 2, (a1 - a0) / 2 * 0.92
        eyes[side] = dict(cx=i[:, 0].mean(), iw=np.ptp(i[:, 0]), x0=mid - half, x1=mid + half, h=np.ptp(w[:, 1]) * 0.92)
    ew = np.mean([e["x1"] - e["x0"] for e in eyes.values()])
    mid = np.vstack([mv["肌"], mv.get("服", mv["肌"])])
    mid = mid[(np.abs(mid[:, 0]) < 0.006) & (mid[:, 1] > 0.40) & (mid[:, 1] < 0.65)]
    crotch = mid[:, 1].min()
    mouth_y, nose_y = mv["口"][:, 1].mean(), mv["鼻"][:, 1].mean()
    height = allv[:, 1].max()
    return {
        "目と目のすき間 / 目の幅": (eyes["L"]["x0"] + eyes["R"]["x0"]) / ew,
        "瞳の中心の間隔 / 目の幅": (eyes["L"]["cx"] - eyes["R"]["cx"]) / ew,
        "瞳の幅 / 目の幅": np.mean([e["iw"] for e in eyes.values()]) / ew,
        "目の高さ / 目の幅": np.mean([e["h"] for e in eyes.values()]) / ew,
        "目の線の高さ（頭のてっぺんから / 頭の高さ）": (top - eye_y) / head,
        "鼻（頭のてっぺんから / 頭の高さ）": (top - nose_y) / head,
        "口（頭のてっぺんから / 頭の高さ）": (top - mouth_y) / head,
        "頭のてっぺん〜股 / 頭の高さ": (top - crotch) / head,
        "頭身（身長 / 頭の高さ）": height / head,
    }


def reference(key):
    """checklists/<キー>.md の「| 項目 | 立ち絵 | 許容 |」表を読む。"""
    p = ROOT / "checklists" / f"{key}.md"
    ref = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|\s*±([\d.]+)\s*\|", line)
            if m:
                ref[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return ref


def main(keys):
    for key in keys:
        got, ref = measure(ROOT / "models" / key / f"{key}.pmx"), reference(key)
        print(f"== {key}")
        for name, v in got.items():
            if name in ref:
                want, tol = ref[name]
                mark = "○" if abs(v - want) <= tol else "×"
                print(f"  {mark} {name}: {v:.3f}（立ち絵 {want:.3f} ±{tol}）")
            else:
                print(f"  - {name}: {v:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["tabane_roze", "chikune_kiriko", "feris"])
