"""服や小物でよく使う形。"""

import numpy as np

from .body import ARM, LEG, TORSO, arm_dir, joints, torso_profile, ARM_ANGLE
from .geom import ellipsoid, loft, ring, rot_axes, torus, tube, TAU


def torso_r(y, grow=1.0):
    rx, rz, cz = torso_profile(y, 1.0)
    return rx * grow, rz * grow, cz


def leg_at(y, s):
    r, x, z = (np.interp(-y, [-t[0] for t in LEG], [t[k] for t in LEG]) for k in (1, 2, 3))
    return r, np.array([s * x, y, z])


def band_around_torso(y0, y1, grow, a0=0.0, a1=TAU, steps=6, n=36, rmod=None, extra=None):
    """胴体を一回り大きく囲む帯。a0..a1 で前を開けられる（角度 -90° が正面）。"""
    closed = abs((a1 - a0) - TAU) < 1e-6
    rings, centers = [], []
    for y in np.linspace(y0, y1, steps):
        g = grow(y) if callable(grow) else grow
        rx, rz, cz = torso_r(y, g)
        if extra:
            rx, rz = rx + extra(y), rz + extra(y)
        c = np.array([0, y, cz])
        rings.append(ring(c, [1, 0, 0], [0, 0, 1], rx, rz, n, a0, a1, closed, rmod))
        centers.append(c)
    return loft(rings, closed=closed, centers=centers)


def clear_legs(y, sx, sz, cz, margin=0.012):
    """スカートの楕円 (sx, sz, 中心 z=cz) が太ももの断面を丸ごと含むまで広げる。
    外側だけ見ると、斜め前の太ももが楕円を突き抜けてスカートに穴があく。"""
    lr, lc = leg_at(y, 1.0)
    th = np.linspace(0, TAU, 48, endpoint=False)
    x = lc[0] + (lr + margin) * np.cos(th)
    z = lc[2] - cz + (lr + margin) * np.sin(th)
    k = np.sqrt(((x / sx) ** 2 + (z / sz) ** 2).max())
    return (sx * k, sz * k) if k > 1 else (sx, sz)


def skirt(y0, y1, r1_scale, pleats=12, depth=0.05, n=72, steps=6, flare_back=0.01):
    """プリーツスカート。腰回りから円錐状に広がる。"""
    rx0, rz0, cz0 = torso_r(y0, 1.06)
    rings, centers = [], []
    for k, y in enumerate(np.linspace(y0, y1, steps)):
        t = k / (steps - 1)
        sx = rx0 + (rx0 * r1_scale - rx0) * t
        sz = rz0 + (rx0 * r1_scale * 0.85 - rz0) * t
        # 太ももが突き抜けないよう、脚の外側より必ず外に出す
        c = np.array([0, y, cz0 + flare_back * t])
        sx, sz = clear_legs(y, sx, sz, c[2])
        dep = depth * t
        rings.append(ring(c, [1, 0, 0], [0, 0, 1], sx, sz, n,
                          rmod=lambda a, d=dep: 1 + d * np.abs(np.sin(a * pleats / 2))))
        centers.append(c)
    return loft(rings, centers=centers)


def sleeve(s, t0, t1, radii, n=20, rmod=None):
    """腕に沿う袖。radii は t0..t1 を等分した各点の半径。"""
    j = joints(s)
    d = arm_dir(s)
    u = np.array([0, 0, 1.0])
    v = np.array([s * np.sin(ARM_ANGLE), np.cos(ARM_ANGLE), 0])
    rings, centers = [], []
    for t, r in zip(np.linspace(t0, t1, len(radii)), radii):
        c = j["腕"] + d * t
        rings.append(ring(c, u, v, r, r, n, rmod=rmod))
        centers.append(c)
    return loft(rings, centers=centers)


def arm_point(s, t):
    return joints(s)["腕"] + arm_dir(s) * t


def arm_torus(s, t, R, r):
    return torus(arm_point(s, t), arm_dir(s), R, r, n=20, m=8)


def leg_tube(s, y0, y1, grow, steps=8, n=22, a0=0.0, a1=TAU, flare=0.0, jag=None, rmod=None):
    """脚を囲む筒（ブーツ・ズボン）。jag(角度) を渡すと裾をギザギザにする。"""
    closed = abs((a1 - a0) - TAU) < 1e-6
    rings, centers = [], []
    ys = np.linspace(y0, y1, steps)
    for k, y in enumerate(ys):
        t = k / (steps - 1)
        r, c = leg_at(y, s)
        rr = r * (grow(t) if callable(grow) else grow) + flare * t
        pts = ring(c, [1, 0, 0], [0, 0, 1], rr, rr, n, a0, a1, closed, rmod)
        if jag is not None and k == steps - 1:
            a = a0 + (a1 - a0) * (np.arange(n) / n if closed else np.linspace(0, 1, n))
            pts[:, 1] += jag(a)
        rings.append(pts)
        centers.append(c)
    return loft(rings, closed=closed, centers=centers)


def foot(s, r=(0.026, 0.026, 0.062), lift=0.0):
    j = joints(s)
    c = np.array([j["足首"][0], 0.026 + lift, -0.024])
    return ellipsoid(c, r, nu=20, nv=12)


def bow(center, span, size, tilt=0.35, facing=(0, 0, -1)):
    """リボン：左右の輪と結び目。facing はリボンの正面。"""
    center = np.asarray(center, float)
    f = np.asarray(facing, float) / np.linalg.norm(facing)
    side = np.cross([0, 1, 0], f)
    side /= np.linalg.norm(side)
    up = np.cross(f, side)
    meshes = []
    for sg in (1, -1):
        ax = side * sg * np.cos(tilt) + up * np.sin(tilt)
        R = rot_axes(ax, np.cross(f, ax), f)
        meshes.append(ellipsoid(center + ax * span, [span, size, size * 0.35], R, nu=40, nv=24))
    meshes.append(ellipsoid(center, [size * 0.55] * 3, nu=28, nv=18))
    return meshes
