"""揺れもの：髪やスカートに骨の鎖を通し、剛体とジョイントで物理演算させる。

剛体の mask は「ぶつかる相手」のビット（MMD・three.js と同じ）。
グループ 0 は体（ボーン追従）、1 は髪、2 はスカート。揺れもの同士はぶつけない。
"""

import numpy as np

from .body import SIDES, joints
from .builder import Joint, Rigid
from .parts import leg_at, torso_r

BODY, HAIR, SKIRT = 0, 1, 2
ALL = 0xFFFF
SPHERE, BOX, CAPSULE = 0, 1, 2


def _rigid(m, name, bone, group, shape, size, pos, mass=1.0, damp=(0.5, 0.5), mode=0, mask=None):
    mask = ALL & ~(1 << group) if mask is None else mask
    m.rigids.append(Rigid(name, m.bone[bone], group, mask, shape, tuple(size), tuple(np.asarray(pos, float)),
                          (0.0, 0.0, 0.0), mass, damp, mode))
    return len(m.rigids) - 1


def _joint(m, name, a, b, pos, lim):
    lim = np.broadcast_to(np.asarray(lim, float), (3,))
    m.joints.append(Joint(name, a, b, tuple(np.asarray(pos, float)), (0.0, 0.0, 0.0),
                          tuple(-lim), tuple(lim), (0.0, 0.0, 0.0)))


def body_rigids(c):
    """体の当たり判定。少し小さめにして、初めから揺れものに食い込まないようにする。"""
    m = c.m
    body = {}
    body["頭"] = _rigid(m, "頭", "頭", BODY, SPHERE, (0.074, 0, 0), c.head.c)
    for bone, y0, y1 in [("上半身2", 0.785, 0.700), ("上半身", 0.700, 0.600), ("下半身", 0.600, 0.500)]:
        ym = (y0 + y1) / 2
        rx, rz, cz = torso_r(ym)
        body[bone] = _rigid(m, bone, bone, BODY, BOX, (rx * 0.85, (y0 - y1) / 2, rz * 0.85), (0, ym, cz))
    for jp, en, s in SIDES:
        j = joints(s)
        for bone, a, b in [(jp + "足", j["足"], j["ひざ"]), (jp + "ひざ", j["ひざ"], j["足首"])]:
            r = leg_at((a[1] + b[1]) / 2, s)[0] * 0.85
            body[bone] = _rigid(m, bone, bone, BODY, CAPSULE, (r, max(0.0, a[1] - b[1] - 2 * r), 0), (a + b) / 2)
    c.body_rigid = body
    return body


def chain(c, name, pts, parent, radius, group=HAIR, mass=0.4, lim=0.35, prev_rigids=None):
    """pts（根元から）に沿って骨を並べ、各区間に球の剛体を付けて parent の剛体へつなぐ。"""
    m = c.m
    pts = np.asarray(pts, float)
    bones, rigids = [], []
    par = parent
    for i in range(len(pts) - 1):
        bn = f"{name}{i + 1}"
        m.add_bone(bn, bn, pts[i], par, tail=pts[i + 1] - pts[i])
        rid = _rigid(m, bn, bn, group, SPHERE, (radius * (1 - 0.3 * i / len(pts)), 0, 0), (pts[i] + pts[i + 1]) / 2,
                     mass=mass, damp=(0.95, 0.99), mode=1)
        a = rigids[-1] if rigids else c.body_rigid[parent]
        _joint(m, bn, a, rid, pts[i], [lim, lim * 0.3, lim])
        bones.append(bn)
        rigids.append(rid)
        par = bn
    return bones, rigids


def skin_chain(parent, bones, pts):
    """鎖に沿ったウェイト。根元より上は parent、骨の中ほどどうしを線形に混ぜる。"""
    pts = np.asarray(pts, float)
    names = [parent] + list(bones)
    knots = np.array([0.0] + [i + 0.5 for i in range(len(bones))])

    def f(p):
        best = np.full(len(p), np.inf)
        u = np.zeros(len(p))
        last = len(pts) - 2
        for i in range(last + 1):
            a, ab = pts[i], pts[i + 1] - pts[i]
            t = ((p - a) @ ab) / (ab @ ab)
            t = np.clip(t, -np.inf if i == 0 else 0, np.inf if i == last else 1)
            d = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
            hit = d < best
            best[hit], u[hit] = d[hit], (i + t)[hit]
        out = []
        for ui in u:
            k = int(np.searchsorted(knots, ui))
            if k == 0:
                out.append((parent, None, 1.0))
            elif k >= len(knots):
                out.append((names[-1], None, 1.0))
            else:
                w = (knots[k] - ui) / (knots[k] - knots[k - 1])
                out.append((names[k - 1], names[k], w))
        return out
    return f


def resample(path, y_from, count):
    """path のうち y_from より下を弧長で count 点に分ける（鎖の骨の位置）。"""
    path = np.asarray(path, float)
    i0 = int(np.argmax(path[:, 1] < y_from)) if np.any(path[:, 1] < y_from) else len(path) - 2
    sub = path[max(0, i0 - 1):]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(sub, axis=0), axis=1))])
    ts = np.linspace(0, s[-1], count)
    return np.stack([np.interp(ts, s, sub[:, k]) for k in range(3)], 1)


def skirt(c, y0, y1, r1_scale, cols=12, segs=3, flare_back=0.01, name="スカート"):
    """parts.skirt と同じ面の上に縦の鎖を cols 本並べ、横どうしもゆるくつなぐ。返り値はウェイト関数。"""
    m = c.m
    rx0, rz0, cz0 = torso_r(y0, 1.06)

    def surface(a, y):
        t = (y0 - y) / (y0 - y1)
        sx = rx0 + (rx0 * r1_scale - rx0) * t
        sz = rz0 + (rx0 * r1_scale * 0.85 - rz0) * t
        lr, lc = leg_at(y, 1.0)
        sx, sz = max(sx, lc[0] + lr + 0.010), max(sz, lr + 0.012)
        return np.array([sx * np.cos(a), y, cz0 + flare_back * t + sz * np.sin(a)])

    angles = np.arange(cols) / cols * 2 * np.pi
    ys = np.linspace(y0, y1, segs + 1)
    columns = []
    for k, a in enumerate(angles):
        pts = [surface(a, y) for y in ys]
        r = np.linalg.norm(pts[-1] - surface(a + 2 * np.pi / cols, ys[-1])) * 0.35
        columns.append(chain(c, f"{name}{k}_", pts, "下半身", r, group=SKIRT, mass=0.3, lim=0.6))
    for k in range(cols):
        for i in range(segs):
            a, b = columns[k][1][i], columns[(k + 1) % cols][1][i]
            mid = (np.asarray(m.rigids[a].pos) + np.asarray(m.rigids[b].pos)) / 2
            _joint(m, f"{name}横{k}_{i}", a, b, mid, 0.5)

    def f(p):
        ang = np.mod(np.arctan2(p[:, 2] - cz0, p[:, 0]), 2 * np.pi) / (2 * np.pi) * cols
        seg = np.clip(((y0 - p[:, 1]) / (y0 - y1) * segs).astype(int), 0, segs - 1)
        top = p[:, 1] > y0 - 0.004
        out = []
        for ai, si, tp in zip(ang, seg, top):
            k0 = int(np.floor(ai)) % cols
            k1 = (k0 + 1) % cols
            if tp:
                out.append(("下半身", None, 1.0))
            else:
                out.append((columns[k0][0][si], columns[k1][0][si], 1.0 - (ai - np.floor(ai))))
        return out
    return f
