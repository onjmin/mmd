"""顔の調整値：人がスライダーで決めた値をキャラごとに mmdgen/characters/face/<キー>.json に置き、作るときに読む。

調整ページ（python serve.py → http://localhost:8765/viewer/tune.html）で動かして保存する。
値はキャラのファイル（characters/*.py）で決めた形に対する倍率・ずらし量が中心で、何も書かなければ元のまま。
"""

import json
from pathlib import Path

DIR = Path(__file__).resolve().parent / "characters" / "face"

# (名前, 画面の見出し, 最小, 最大, 刻み, 既定値, 説明)
SPEC = [
    ("eye_size", "目の大きさ", 0.6, 1.5, 0.01, 1.0, "目の幅と高さをまとめて何倍にするか"),
    ("eye_aspect", "目の縦の長さ", 0.6, 1.5, 0.01, 1.0, "目の高さだけを何倍にするか"),
    ("eye_spread", "目の間隔", 0.85, 1.35, 0.01, 1.07, "大きいほど目が外へ離れる"),
    ("eye_height", "目の高さ", 0.32, 0.46, 0.005, 0.375, "あご 0〜頭頂 1 のどこに目の線を置くか"),
    ("eye_tilt", "目の傾き", -0.3, 0.3, 0.01, 0.0, "正でつり目、負でたれ目"),
    ("iris_w", "瞳の幅", 0.6, 1.4, 0.01, 1.0, "瞳の幅を何倍にするか"),
    ("iris_h", "瞳の高さ", 0.6, 1.4, 0.01, 1.0, "瞳の高さを何倍にするか"),
    ("iris_y", "瞳の上下", -0.3, 0.3, 0.01, 0.0, "正で瞳を上へ（目の高さの半分を 1 として）"),
    ("pupil", "瞳孔の大きさ", 0.4, 1.8, 0.01, 1.0, "瞳の中の暗い所の大きさ"),
    ("highlight", "瞳の光の大きさ", 0.0, 2.0, 0.01, 1.0, "左上の白い光（0 で消す）"),
    ("lash", "まつ毛の太さ", 0.3, 2.0, 0.01, 1.0, "上まつ毛の帯の太さ"),
    ("lash_light", "まつ毛の明るさ", 0.3, 2.5, 0.01, 1.0, "大きいほど薄い色、小さいほど黒に近い"),
    ("white_shade", "白目の影", 0.0, 2.0, 0.01, 1.0, "白目の上に落ちる色の濃さ（0 で真っ白）"),
    ("face_width", "顔の幅", 0.42, 0.64, 0.005, 0.52, "頭の横の大きさ"),
    ("cheek", "頬の張り", -0.4, 0.25, 0.01, -0.16, "負で目の下〜あごを広げて丸顔に、正で細く"),
]
DEFAULTS = {k: d for k, _, _, _, _, d, _ in SPEC}

_current = {}
EFFECTIVE = {}  # 最後に作ったキャラで実際に効いた値（瞳の色など。調整ページに見せる）


def path(key):
    return DIR / f"{key}.json"


def load(key):
    """保存した値（なければ空）。"""
    p = path(key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(key, values):
    DIR.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in values.items() if k in DEFAULTS or k == "eye_color"}
    path(key).write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def use(values):
    """これから作るキャラの調整値を決める（build.py と調整ページが呼ぶ）。"""
    global _current
    _current = dict(values or {})


def get(name, default=None):
    if name in _current:
        return _current[name]
    return DEFAULTS.get(name, default)
