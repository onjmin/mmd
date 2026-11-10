"""塗りの焼き込み：服や髪が落とす影と奥まった所の暗さを計算し、色付きの影としてテクスチャに入れる。

既製の MMD モデルはテクスチャに影やグラデーションが描き込んであり、単色の材質だけでは平板に見える。
ここでは頂点ごとに「まわりからどれだけ光が届くか」を求め、材質ごとの帯状のテクスチャ（影色→地色）の
どこを使うかを UV の v で指す。UV 展開がなくても頂点単位の塗り分けになる。

求め方：多数の向きから深度図を作り、各頂点がその向きから見えるかを調べる（空と正面上からの光を重く見る）。
顔の肌は、髪や帽子が落とす影だけを入れる（鼻や目のくぼみで顔が汚れないよう、肌どうしの影は入れない）。
"""

import numpy as np

from . import pmx

PIXEL = 0.0035  # 深度図の 1 画素（身長 1 に対して）
KEY = np.array([-0.25, 0.75, -0.60])  # 主な光：正面の少し上（顔は -Z 向き）


def _dirs(n=56):
    k = np.arange(n) + 0.5
    y = 1 - 2 * k / n
    r = np.sqrt(1 - y * y)
    a = k * np.pi * (3 - np.sqrt(5))
    d = np.stack([r * np.cos(a), y, r * np.sin(a)], 1)
    d = d[d[:, 1] > -0.35]  # 真下からの光はない
    key = KEY / np.linalg.norm(KEY)
    w = 0.30 + 0.25 * np.clip(d[:, 1], 0, 1) + 0.9 * np.clip(d @ key, 0, 1) ** 2
    return d, w


def _samples(P, F):
    """面の上の点。大きい三角形は画素より細かく割り、深度図に穴があかないようにする。"""
    out = [P]
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    edge = np.maximum(np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1)), np.linalg.norm(a - c, axis=1))
    n = np.clip(np.ceil(edge / (PIXEL * 0.8)), 1, 16).astype(int)
    for k in np.unique(n):
        if k < 2:
            continue
        sel = n == k
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        ok = (i + j <= k) & ~((i == 0) & (j == 0)) & ~((i == k) & (j == 0)) & ~((i == 0) & (j == k))
        u, v = i[ok] / k, j[ok] / k
        pts = a[sel][:, None] * (1 - u - v)[None, :, None] + b[sel][:, None] * u[None, :, None] + c[sel][:, None] * v[None, :, None]
        out.append(pts.reshape(-1, 3))
    return np.vstack(out)


def _visible(S_list, Q, d, bias):
    """向き d の光が点 Q に届くか。S_list は遮るものの点の集まり（複数の組を一度に調べる）。"""
    ref = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    qu, qv, qd = Q @ u, Q @ v, Q @ d
    u0, v0 = qu.min() - 2 * PIXEL, qv.min() - 2 * PIXEL
    W = int((qu.max() - u0) / PIXEL) + 3
    H = int((qv.max() - v0) / PIXEL) + 3
    qi = ((qv - v0) / PIXEL).astype(int) * W + ((qu - u0) / PIXEL).astype(int)
    res = []
    for S in S_list:
        su, sv, sd = S @ u, S @ v, S @ d
        ix, iy = ((su - u0) / PIXEL).astype(int), ((sv - v0) / PIXEL).astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        buf = np.full(W * H, -np.inf)
        np.maximum.at(buf, iy[ok] * W + ix[ok], sd[ok])
        res.append(qd >= buf[qi] - bias)
    return res


def _smooth(val, F, it=3):
    """となりの頂点とならして、画素のがたつきを消す。"""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    cnt = np.bincount(e[:, 0], minlength=len(val)).astype(float)
    for _ in range(it):
        s = np.bincount(e[:, 0], weights=val[e[:, 1]], minlength=len(val))
        val = np.where(cnt > 0, 0.5 * val + 0.5 * s / np.maximum(cnt, 1), val)
    return val


def shadow_color(base):
    """影の色：ただ暗くせず、色みを残して少し紫へ寄せる（明るい色ほど色を付け、暗い色は沈める）。"""
    base = np.asarray(base, float)[:3]
    lum = float(base @ [0.3, 0.59, 0.11])
    # 暖色（肌・茶・クリーム）は赤みへ、寒色や無彩色は青紫へ寄せる（暖色を紫に寄せると灰色に濁る）
    warm = float(np.clip((base[0] - base[2]) / 0.25, 0, 1))
    tint = np.array([0.74, 0.70, 0.86]) * (1 - warm) + np.array([0.90, 0.72, 0.66]) * warm
    sat = base - lum
    return np.clip((lum + sat * 1.25) * tint * (0.86 if lum > 0.25 else 0.70), 0, 1)


def ramp(base, shadow=None, n=128):
    """帯の色（n 行 × 3）。行 0 がいちばん深い影、最後の行が地色。"""
    base = np.asarray(base, float)[:3]
    shadow = shadow_color(base) if shadow is None else np.asarray(shadow, float)
    t = np.linspace(0, 1, n)
    k = np.clip((t - 0.30) / 0.45, 0, 1)
    k = k * k * (3 - 2 * k)
    deep = shadow * 0.82
    low = deep[None] + (shadow - deep)[None] * np.clip(t / 0.30, 0, 1)[:, None]
    col = low * (1 - k)[:, None] + base[None] * k[:, None]
    # 暗い色は影が見えにくいので、よく光の当たる所を少し持ち上げて立体を出す
    lum = float(base @ [0.3, 0.59, 0.11])
    if lum < 0.25:
        lift = np.clip((t - 0.80) / 0.20, 0, 1)[:, None] * np.array([0.07, 0.07, 0.10])
        col = col + lift
    return np.clip(col, 0, 1)


def bake(m, face_y=None):
    """m（builder.Model）の全頂点の明るさを求め、焼き込み対象の材質の UV を書き換える。
    材質の bake が "ramp" なら UV 全体、"v" なら v だけ（u は髪の筋に使う）を置き換える。"""
    P = np.asarray(m.positions, float)
    N = np.asarray(m.normals, float)
    uvs = np.asarray(m.uvs, float)
    owner = np.full(len(P), -1)
    occ_all, occ_cloth, faces_all = [], [], []
    for k, mt in enumerate(m.materials):
        F = np.asarray(mt.faces, int).reshape(-1, 3)
        if not len(F):
            continue
        owner[F.ravel()] = k
        faces_all.append(F)
        if mt.flags & pmx.MAT_SELF_SHADOW and mt.alpha > 0.5:
            s = _samples(P, F)
            occ_all.append(s)
            if mt.name_en not in ("skin", "ear_in"):
                occ_cloth.append(s)
    if not occ_all:
        return
    S_all = np.vstack(occ_all)
    S_cloth = np.vstack(occ_cloth) if occ_cloth else np.zeros((0, 3))
    F_all = np.vstack(faces_all)
    dirs, w = _dirs()
    got_all = np.zeros(len(P))
    got_cloth = np.zeros(len(P))
    free = np.zeros(len(P))
    for d, wd in zip(dirs, w):
        c = np.clip(N @ d, 0, 1) * wd
        va, vc = _visible([S_all, S_cloth], P, d, PIXEL * 2.2)
        got_all += c * va
        got_cloth += c * vc
        free += c
    ref = free.max()
    occl_all = got_all / np.maximum(free, 1e-6)
    occl_cloth = got_cloth / np.maximum(free, 1e-6)
    form = np.clip(free / ref, 0, 1)  # 下や奥を向いた面ほど暗い（塗りの「下は暗く」）
    shade = occl_all * (0.72 + 0.28 * form)
    if face_y is not None:
        skin = np.isin(owner, [k for k, mt in enumerate(m.materials) if mt.name_en in ("skin", "ear_in")])
        head = skin & (P[:, 1] > face_y)
        blend = np.clip((P[:, 1] - face_y) / 0.012, 0, 1)  # あごの下の影から顔へなめらかにつなぐ
        face_shade = 0.25 + 0.75 * occl_cloth  # 顔は髪の影だけ。薄めに入れる
        shade = np.where(head, shade * (1 - blend) + face_shade * blend, shade)
    shade = _smooth(shade, F_all)
    t = np.clip((shade - 0.18) / 0.62, 0.02, 0.98)
    # 肌は深い影まで落とさない（服の中で暗く濁って見えるのを防ぐ）
    is_skin = np.isin(owner, [k for k, mt in enumerate(m.materials) if mt.name_en in ("skin", "ear_in")])
    t = np.where(is_skin, np.maximum(t, 0.36), t)
    for k, mt in enumerate(m.materials):
        if not mt.bake:
            continue
        sel = owner == k
        if mt.bake == "ramp":
            uvs[sel, 0] = 0.5
        uvs[sel, 1] = t[sel]
    m.uvs = [(float(a), float(b)) for a, b in uvs]
