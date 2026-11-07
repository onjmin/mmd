"""python build.py で models/ に PMX を書き出す。"""

import os
import sys

from mmdgen.characters import feris, kiriko, roze

TARGETS = {"tabane_roze": roze, "chikune_kiriko": kiriko, "feris": feris}


def main(names):
    root = os.path.dirname(os.path.abspath(__file__))
    for key in names or TARGETS:
        out = os.path.join(root, "models", key)
        os.makedirs(out, exist_ok=True)
        model = TARGETS[key].build(out)
        path = os.path.join(out, key + ".pmx")
        model.save(path)
        nf = sum(len(mt.faces) for mt in model.materials)
        print(f"{path}: {len(model.positions)} 頂点 / {nf} 面 / {len(model.bones)} ボーン / {len(model.morphs)} モーフ")


if __name__ == "__main__":
    main(sys.argv[1:])
