"""揺れもの：髪やスカートに骨の鎖を通し、剛体とジョイントで物理演算させる。

剛体の mask は「ぶつかる相手」のビット（MMD・three.js と同じ）。
グループ 0 は体（ボーン追従）、1 は髪、2 はスカート。揺れもの同士はぶつけない。
"""

import numpy as np

from .body import SIDES, joints
from .builder import Joint, Rigid
from .parts import clear_legs, leg_at, torso_r

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


def chain(c, name, pts, parent, radius, group=HAIR, mass=0.4, lim=0.35, prev_rigids=None, root_lim=None):
    """pts（根元から）に沿って骨を並べ、各区間に球の剛体を付けて parent の剛体へつなぐ。
    root_lim は根元の関節だけの可動角（スカートが上げた太ももに押されて大きくめくれるように）。"""
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
        li = root_lim if (i == 0 and root_lim is not None) else lim
        _joint(m, bn, a, rid, pts[i], [li, lim * 0.3, li])
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


def cloth(c, name, surface, us, ys, parent, closed, mass=0.3, lim=0.6, root_lim=1.5, rscale=0.55):
    """布：surface(u, y) の上に縦の鎖を us の列ぶん並べ、横どうしもゆるくつなぐ。
    返り値は param(頂点)→(u, y) を受け取り、縦横 4 本の骨でなめらかに混ぜるウェイト関数を作る関数。"""
    m = c.m
    cols, segs = len(us), len(ys) - 1
    columns, rigids = [], []
    for k, u in enumerate(us):
        pts = [surface(u, y) for y in ys]
        nb = us[(k + 1) % cols] if (closed or k + 1 < cols) else us[k - 1]
        # 球は隣の列との間をほぼふさぐ大きさにして、脚が列の間を抜けないようにする
        r = np.linalg.norm(pts[-1] - surface(nb, ys[-1])) * rscale
        bones, rids = chain(c, f"{name}{k}_", pts, parent, r, group=SKIRT, mass=mass, lim=lim, root_lim=root_lim)
        columns.append(bones)
        rigids.append(rids)
    for k in range(cols if closed else cols - 1):
        for i in range(segs):
            a, b = rigids[k][i], rigids[(k + 1) % cols][i]
            mid = (np.asarray(m.rigids[a].pos) + np.asarray(m.rigids[b].pos)) / 2
            _joint(m, f"{name}横{k}_{i}", a, b, mid, 0.5)
    us = np.asarray(us, float)
    knots = np.concatenate([[ys[0]], (np.asarray(ys[:-1]) + np.asarray(ys[1:])) / 2])  # 根元、各骨の中ほど

    def weights(param):
        def f(p):
            u, y = param(p)
            out = []
            for ui, yi in zip(u, y):
                # 横：両隣の列（開いた布の端は端の列だけ）
                if closed:
                    span = 2 * np.pi
                    x = np.mod(ui - us[0], span) / span * cols
                    k0 = int(np.floor(x)) % cols
                    pairs = [(k0, 1 - (x - np.floor(x))), ((k0 + 1) % cols, x - np.floor(x))]
                else:
                    x = np.clip(np.interp(ui, us, np.arange(cols)), 0, cols - 1)
                    k0 = min(int(np.floor(x)), cols - 2)
                    pairs = [(k0, 1 - (x - k0)), (k0 + 1, x - k0)]
                # 縦：根元は親、そこから下は骨の中ほどどうしを線形に
                j = int(np.searchsorted(-knots, -yi))  # knots は下がっていく
                if j == 0:
                    vert = [(None, 1.0)]
                elif j >= len(knots):
                    vert = [(segs - 1, 1.0)]
                else:
                    w = (yi - knots[j]) / (knots[j - 1] - knots[j])
                    vert = [(j - 2 if j >= 2 else None, w), (j - 1, 1 - w)]
                acc = {}
                for kc, wc in pairs:
                    for sv, wv in vert:
                        b = parent if sv is None else columns[kc][sv]
                        acc[b] = acc.get(b, 0.0) + wc * wv
                out.append([(b, w) for b, w in acc.items() if w > 1e-4])
            return out
        return f
    return weights


def skirt(c, y0, y1, r1_scale, cols=12, segs=3, flare_back=0.01, name="スカート"):
    """parts.skirt と同じ面の上に縦の鎖を cols 本並べ、横どうしもゆるくつなぐ。返り値はウェイト関数。"""
    rx0, rz0, cz0 = torso_r(y0, 1.06)

    def surface(a, y):
        t = (y0 - y) / (y0 - y1)
        sx = rx0 + (rx0 * r1_scale - rx0) * t
        sz = rz0 + (rx0 * r1_scale * 0.85 - rz0) * t
        sx, sz = clear_legs(y, sx, sz, cz0 + flare_back * t)
        return np.array([sx * np.cos(a), y, cz0 + flare_back * t + sz * np.sin(a)])

    angles = np.arange(cols) / cols * 2 * np.pi
    w = cloth(c, name, surface, angles, np.linspace(y0, y1, segs + 1), "下半身", True)
    return w(lambda p: (np.arctan2(p[:, 2] - cz0, p[:, 0]), p[:, 1]))
