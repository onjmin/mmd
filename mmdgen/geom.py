"""形状の部品。座標は身長 = 1 の正規化単位、MMD と同じ左手系（+Y 上、正面は -Z、+X がキャラの左）。"""

import numpy as np

TAU = 2 * np.pi
UP = np.array([0.0, 1.0, 0.0])


def norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rot_axes(x, y, z):
    """ローカル x/y/z 軸を各列に持つ回転行列。"""
    return np.stack([norm(x), norm(y), norm(z)], axis=1)


def ring(c, u, v, ru, rv, n, a0=0.0, a1=TAU, closed=True, rmod=None):
    a = a0 + (a1 - a0) * (np.arange(n) / n if closed else np.linspace(0, 1, n))
    k = rmod(a) if rmod else 1.0
    return (np.asarray(c, float) + np.outer(np.cos(a) * ru * k, u) + np.outer(np.sin(a) * rv * k, v))


def loft(rings, closed=True, cap0=False, cap1=False, centers=None):
    """リングの列をつないで面にする。面は外向き（中心から離れる向き）に揃える。"""
    rings = [np.asarray(r, float) for r in rings]
    n = len(rings[0])
    if centers is None:
        centers = [r.mean(axis=0) for r in rings]
    verts = np.vstack(rings)
    faces = []
    span = n if closed else n - 1
    for i in range(len(rings) - 1):
        for j in range(span):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = (i + 1) * n + (j + 1) % n, (i + 1) * n + j
            faces += [(a, b, c), (a, c, d)]
    extra = []
    if cap0:
        ci = len(verts) + len(extra)
        extra.append(centers[0])
        faces += [(ci, (j + 1) % n, j) for j in range(span)]
    if cap1:
        ci = len(verts) + len(extra)
        extra.append(centers[-1])
        base = (len(rings) - 1) * n
        faces += [(ci, base + j, base + (j + 1) % n) for j in range(span)]
    if extra:
        verts = np.vstack([verts, extra])
    faces = np.array(faces, int)
    # 向きの判定：リング中心から外向きなら正
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ring_idx = np.minimum(faces[:, 0] // n, len(rings) - 1)
    ctr = np.asarray(centers)[ring_idx]
    if np.sum(np.einsum("ij,ij->i", fn, tri.mean(axis=1) - ctr)) < 0:
        faces = faces[:, ::-1]
    return verts, faces


def ellipsoid(c, r, R=None, nu=24, nv=14, keep=None, shape=None):
    """楕円体。keep(単位方向) が False の面を落とす。shape は局所座標の変形関数。"""
    c = np.asarray(c, float)
    R = np.eye(3) if R is None else R
    rings, dirs = [], []
    for i in range(nv + 1):
        phi = -np.pi / 2 + np.pi * i / nv
        a = np.arange(nu) / nu * TAU
        d = np.stack([np.cos(phi) * np.sin(a), np.full(nu, np.sin(phi)), -np.cos(phi) * np.cos(a)], 1)
        loc = d * r
        if shape is not None:
            loc = shape(loc)
        rings.append(c + loc @ R.T)
        dirs.append(d)
    verts, faces = loft(rings, centers=[c] * len(rings))
    if keep is not None:
        ok = keep(np.vstack(dirs))
        faces = faces[ok[faces].all(axis=1)]
    return verts, faces


def frames_along(path, up=None):
    """パスに沿った (接線, 横, 法線)。横は up と接線から作り、なければ平行移動で運ぶ。"""
    path = np.asarray(path, float)
    t = np.gradient(path, axis=0)
    t = norm(t)
    sides = []
    for i in range(len(path)):
        u = up[i] if up is not None and np.ndim(up) == 2 else (up if up is not None else None)
        if u is None:
            if i == 0:
                ref = UP if abs(t[0] @ UP) < 0.9 else np.array([1.0, 0, 0])
                s = norm(np.cross(ref, t[0]))
            else:
                s = sides[-1] - (sides[-1] @ t[i]) * t[i]
                s = norm(s)
        else:
            s = norm(np.asarray(u) - (np.asarray(u) @ t[i]) * t[i])
        sides.append(s)
    sides = np.array(sides)
    nrm = norm(np.cross(t, sides))
    return t, sides, nrm


def tube(path, ru, rv=None, n=12, side=None, cap0=False, cap1=False, rmod=None):
    """パスに沿う管。ru は横方向（side）、rv はそれと直交する方向の半径。"""
    path = np.asarray(path, float)
    ru = _resample(ru, len(path))
    rv = ru if rv is None else _resample(rv, len(path))
    _, s, nn = frames_along(path, side)
    rings = [ring(path[i], s[i], nn[i], ru[i], rv[i], n, rmod=rmod) for i in range(len(path))]
    return loft(rings, cap0=cap0, cap1=cap1, centers=list(path))


def _resample(r, n):
    """半径の指定が点数と違えば、パス全体に引き伸ばす。"""
    r = np.atleast_1d(np.asarray(r, float))
    if len(r) == 1:
        return np.full(n, r[0])
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r)), r)


def torus(c, axis, R, r, n=24, m=8):
    axis = norm(axis)
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 0, 1.0])
    u = norm(np.cross(axis, ref))
    v = np.cross(axis, u)
    a = np.arange(n + 1) / n * TAU
    path = np.asarray(c) + np.outer(np.cos(a), u) * R + np.outer(np.sin(a), v) * R
    verts, faces = tube(path, r, n=m, side=np.broadcast_to(axis, path.shape))
    return verts, faces


def catmull(points, samples=24):
    """制御点を滑らかにつないだ点列。"""
    p = np.asarray(points, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    segs = len(p) - 3
    for i in range(segs):
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        cnt = max(2, samples // segs)
        for t in np.linspace(0, 1, cnt, endpoint=(i == segs - 1)):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return norm(vn)
