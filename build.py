"""python build.py で models/ に PMX を書き出す。"""

import os
import sys

from mmdgen import body, face
from mmdgen.characters import feris, kiriko, roze

TARGETS = {"tabane_roze": roze, "chikune_kiriko": kiriko, "feris": feris}
DEFAULT_HEAD = "sculpt"  # 頭の作り方：sculpt は設計曲線から、scan は既製モデルを測った表から


def main(names):
    root = os.path.dirname(os.path.abspath(__file__))
    for key in names or TARGETS:
        out = os.path.join(root, "models", key)
        os.makedirs(out, exist_ok=True)
        # <キー>_sculpt / <キー>_scan で頭の作り方を指定して別フォルダに書き出す（見比べ用）
        base, _, style = key.rpartition("_") if key.endswith(("_sculpt", "_scan")) else (key, "", "")
        body.HEAD_STYLE = style or DEFAULT_HEAD
        face.use(face.load(base))  # 調整ページで保存した顔の調整値（mmdgen/characters/face/<キー>.json）
        model = TARGETS[base].build(out)
        path = os.path.join(out, key + ".pmx")
        model.save(path)
        nf = sum(len(mt.faces) for mt in model.materials)
        print(f"{path}: {len(model.positions)} 頂点 / {nf} 面 / {len(model.bones)} ボーン / {len(model.morphs)} モーフ")


if __name__ == "__main__":
    main(sys.argv[1:])
