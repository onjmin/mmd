"""書き出し済みの PMX に剛体とジョイントを付け直す（VRM から変換したモデルの揺れもの用）。

体は標準ボーンの位置から当たり判定を作り、swing に挙げた骨（とその子孫）を物理で揺らす。
単位は PMX のまま（1 = 8cm）。
"""

import struct

import numpy as np

from .inspect_pmx import read
from .pmx import _text

ALL = 0xFFFF
BODY, HAIR = 0, 1
SPHERE, BOX, CAPSULE = 0, 1, 2


def _idx(i, size):
    return struct.pack({1: "<b", 2: "<h", 4: "<i"}[size], i)


def _z_rot(d):
    """カプセル（Y 軸）を XY 平面内で d の向きにする Z 回転。"""
    return (0.0, 0.0, float(np.arctan2(-d[0], d[1])))


def add_physics(path, swing, hair_radius=0.25, lim=0.30):
    info = read(path)
    assert info["encoding"] == 0, "UTF-16 の PMX だけ扱う"
    bs = info["bone_index_size"]
    bones = info["bones"]
    index = {b[0]: i for i, b in enumerate(bones)}
    pos = {b[0]: np.array(b[1], float) for b in bones}
    parent = {b[0]: (bones[b[2]][0] if b[2] >= 0 else None) for b in bones}
    children = {}
    for n, p in parent.items():
        children.setdefault(p, []).append(n)

    rigids, joints = [], []
    rid = {}

    def rigid(name, bone, group, shape, size, at, rot=(0, 0, 0), mode=0, mass=1.0, damp=(0.5, 0.5), mask=None):
        mask = ALL & ~(1 << group) if mask is None else mask
        rid[name] = len(rigids)
        rigids.append((name, index[bone], group, mask, shape, size, tuple(at), tuple(rot), mass, damp, mode))
        return rid[name]

    # ---- 体（ボーン追従） ----
    def seg(name, a, b, r):
        if a in pos and b in pos:
            d = pos[b] - pos[a]
            rigid(name, a, BODY, CAPSULE, (r, max(0.0, np.linalg.norm(d) - 2 * r), 0), (pos[a] + pos[b]) / 2, _z_rot(d))

    if "頭" in pos:
        rigid("頭", "頭", BODY, SPHERE, (0.85, 0, 0), pos["頭"] + [0, 0.85, 0.05])
    seg("上半身", "上半身", "首", 0.75)
    if "下半身" in pos:
        rigid("下半身", "下半身", BODY, SPHERE, (0.95, 0, 0), pos["下半身"] + [0, -0.35, 0])
    for jp in "左右":
        seg(jp + "足", jp + "足", jp + "ひざ", 0.55)
        seg(jp + "ひざ", jp + "ひざ", jp + "足首", 0.42)
        seg(jp + "腕", jp + "腕", jp + "ひじ", 0.30)
        seg(jp + "ひじ", jp + "ひじ", jp + "手首", 0.25)

    # ---- 揺れもの ----
    moving = set()
    stack = [n for n in swing if n in pos]
    while stack:
        n = stack.pop()
        if n not in moving:
            moving.add(n)
            stack += children.get(n, [])
    order = [b[0] for b in bones if b[0] in moving]  # 親が先に来る
    for n in order:
        kids = [k for k in children.get(n, []) if k in moving]
        if kids:
            end = np.mean([pos[k] for k in kids], 0)
        elif parent[n] is not None:
            end = pos[n] + (pos[n] - pos[parent[n]]) * 0.5
        else:
            continue
        r = min(hair_radius, max(0.08, np.linalg.norm(end - pos[n]) * 0.35))
        me = rigid(n, n, HAIR, SPHERE, (r, 0, 0), (pos[n] + end) / 2, mode=1, mass=0.3, damp=(0.98, 0.998),
                   mask=1 << BODY)
        p = parent[n]
        if p not in rid:  # 根元の動かない骨には、髪とはぶつからない追従の剛体を置く
            rigid(p, p, HAIR, SPHERE, (0.05, 0, 0), pos[p], mask=0)
        joints.append((n, rid[p], me, pos[n], lim))

    data = open(path, "rb").read()[:info["rigid_offset"]]
    out = bytearray(data)
    out[16] = rs = 2  # 剛体の参照はジョイントにしか無いので、数が増えても入るよう 2 バイトにする
    out += struct.pack("<i", len(rigids))
    for name, bone, group, mask, shape, size, at, rot, mass, damp, mode in rigids:
        out += _text(name) + _text(name)
        out += _idx(bone, bs) + struct.pack("<BHB", group, mask, shape)
        out += struct.pack("<9f", *size, *at, *rot)
        out += struct.pack("<5fB", mass, damp[0], damp[1], 0.0, 0.5, mode)
    out += struct.pack("<i", len(joints))
    for name, a, b, at, l in joints:
        out += _text(name) + _text(name) + struct.pack("<B", 0) + _idx(a, rs) + _idx(b, rs)
        out += struct.pack("<6f", *at, 0, 0, 0)
        out += struct.pack("<12f", 0, 0, 0, 0, 0, 0, -l, -l * 0.3, -l, l, l * 0.3, l)
        out += struct.pack("<6f", 0, 0, 0, 0, 0, 0)
    open(path, "wb").write(bytes(out))
    return len(rigids), len(joints)
