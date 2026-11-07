"""モデルの組み立て：材質・ボーン・ウェイト・モーフ・表示枠。"""

from dataclasses import dataclass, field

import numpy as np

from . import pmx
from .geom import norm, vertex_normals


@dataclass
class Material:
    name: str
    name_en: str
    diffuse: tuple
    ambient: tuple
    flags: int
    edge_size: float
    edge_color: tuple
    toon: int
    tex: str = None
    sphere: str = None
    sphere_mode: int = 0
    toon_tex: str = None
    alpha: float = 1.0
    specular: tuple = (0.0, 0.0, 0.0)
    power: float = 5.0
    faces: list = field(default_factory=list)


@dataclass
class Rigid:
    name: str
    bone: int
    group: int
    mask: int
    shape: int  # 0 球, 1 箱, 2 カプセル
    size: tuple
    pos: tuple
    rot: tuple
    mass: float
    damp: tuple
    mode: int  # 0 ボーン追従, 1 物理演算, 2 物理＋ボーン位置合わせ


@dataclass
class Joint:
    name: str
    a: int
    b: int
    pos: tuple
    rot: tuple
    rot_lo: tuple
    rot_hi: tuple
    spring: tuple


@dataclass
class Bone:
    name: str
    name_en: str
    pos: tuple
    parent: int
    flags: int
    tail: object
    inherit: tuple = None
    ik: tuple = None


@dataclass
class Morph:
    name: str
    name_en: str
    panel: int
    offsets: list


class Model:
    def __init__(self, name, name_en, comment, height):
        self.name, self.name_en = name, name_en
        self.comment, self.comment_en = comment, name_en
        self.scale = height / 8.0  # 身長(cm) → MMD 単位（1 = 8cm）
        self.materials = []
        self._mat = {}
        self.bones = []
        self.bone = {}
        self.seg = {}  # 自動ウェイト用の骨区間（正規化単位）
        self.positions, self.normals, self.uvs, self.weights = [], [], [], []
        self.morphs = []
        self._morph_acc = {}
        self.frames = []
        self.textures = []
        self.rigids, self.joints = [], []

    # ---- 材質 ----
    def material(self, key, name, color, double=False, edge=True, toon=1, edge_size=0.6, shade=0.4,
                 tex=None, sphere=None, sphere_mode=1, toon_tex=None, alpha=1.0, shadow=True):
        """tex を渡すと色はテクスチャから取り、color は輪郭線の色を決めるのに使う。"""
        if key in self._mat:
            return key
        flags = (pmx.MAT_GROUND_SHADOW | pmx.MAT_SHADOW_MAP | pmx.MAT_SELF_SHADOW) if shadow else 0
        if double:
            flags |= pmx.MAT_DOUBLE
        if edge:
            flags |= pmx.MAT_EDGE
        c = tuple(float(x) for x in color)
        ec = tuple(max(0.0, x * 0.35) for x in c)
        diffuse = (1.0, 1.0, 1.0) if tex else c
        mt = Material(name, key, diffuse, tuple(x * shade for x in diffuse), flags, edge_size, ec, toon,
                      tex=tex, sphere=sphere, sphere_mode=sphere_mode, toon_tex=toon_tex, alpha=alpha)
        for t in (tex, sphere, toon_tex):
            if t and t not in self.textures:
                self.textures.append(t)
        self._mat[key] = mt
        self.materials.append(mt)
        return key

    # ---- ボーン ----
    def add_bone(self, name, name_en, pos, parent=None, flags=None, tail=None, inherit=None, ik=None, seg_to=None):
        if flags is None:
            flags = pmx.BONE_ROTATE | pmx.BONE_VISIBLE | pmx.BONE_OPERABLE
        if isinstance(tail, str):
            tail_v = tail
            flags |= pmx.BONE_TAIL_INDEX
        else:
            tail_v = (0.0, 0.0, 0.0) if tail is None else tuple(tail)
        if inherit is not None:
            flags |= pmx.BONE_INHERIT_ROT
        b = Bone(name, name_en, tuple(np.asarray(pos, float)), -1 if parent is None else self.bone[parent], flags, tail_v, inherit, ik)
        self.bone[name] = len(self.bones)
        self.bones.append(b)
        if seg_to is not None:
            self.seg[name] = (np.asarray(pos, float), np.asarray(seg_to, float))
        return name

    def resolve_bones(self):
        for b in self.bones:
            if isinstance(b.tail, str):
                b.tail = self.bone[b.tail]
            if b.inherit is not None and isinstance(b.inherit[0], str):
                b.inherit = (self.bone[b.inherit[0]], b.inherit[1])
            if b.ik is not None:
                target, loop, limit, links = b.ik
                b.ik = (self.bone[target], loop, limit, [(self.bone[n], lim) for n, lim in links])

    # ---- ウェイト ----
    def skin_fixed(self, bone):
        def f(p):
            return [(bone, None, 1.0)] * len(p)
        return f

    def skin_auto(self, cands, bias=None, power=8.0):
        """候補骨の区間までの距離で 2 本に振り分ける。bias は距離の倍率（大きいほど効きにくい）。"""
        bias = bias or {}

        def f(p):
            ds = []
            for b in cands:
                a, e = self.seg[b]
                ab = e - a
                t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                d = np.linalg.norm(p - (a + t[:, None] * ab), axis=1) * bias.get(b, 1.0)
                ds.append(d)
            ds = np.stack(ds, 1)
            w = 1.0 / np.maximum(ds, 1e-4) ** power
            order = np.argsort(-w, axis=1)
            res = []
            for i in range(len(p)):
                i1, i2 = order[i, 0], order[i, 1] if len(cands) > 1 else order[i, 0]
                w1, w2 = w[i, i1], w[i, i2] if len(cands) > 1 else 0.0
                res.append((cands[i1], cands[i2] if len(cands) > 1 else None, w1 / (w1 + w2)))
            return res
        return f

    def skin_vertical(self, top, bottom, y_top, y_bot):
        """高さで top → bottom に移す（髪の毛先を体に付けるなど）。"""
        def f(p):
            t = np.clip((y_top - p[:, 1]) / (y_top - y_bot), 0, 1)
            return [(top, bottom, 1.0 - ti) for ti in t]
        return f

    # ---- 形状 ----
    def add(self, mat, mesh, skin, uv=None, normals=None):
        """uv を省くと mesh.uv を使う。normals(頂点) を渡すと法線をそれで置き換える（顔の影を整えるため）。"""
        verts, faces = mesh
        if uv is None:
            uv = getattr(mesh, "uv", None)
        if uv is None:
            uv = np.zeros((len(verts), 2))
        used = np.unique(faces)
        remap = -np.ones(len(verts), int)
        remap[used] = np.arange(len(used))
        verts = verts[used]
        uv = np.asarray(uv)[used]
        faces = remap[faces]
        base = len(self.positions)
        nrm = normals(verts) if normals else _merge_seams(verts, vertex_normals(verts, faces))
        for p, n, t, w in zip(verts, nrm, uv, skin(verts)):
            self.positions.append(p)
            self.normals.append(n)
            self.uvs.append((float(t[0]), float(t[1])))
            self.weights.append(w)
        self._mat[mat].faces.extend((faces + base).tolist())
        return base, len(verts), used

    def add_morph_part(self, morph, base_info, verts_new):
        """add() の返り値と、同じ頂点並びの変形後形状から差分を積む。"""
        base, count, used = base_info
        verts_new = np.asarray(verts_new)[used]
        acc = self._morph_acc.setdefault(morph, {})
        for k in range(count):
            d = verts_new[k] - self.positions[base + k]
            if np.linalg.norm(d) > 1e-7:
                acc[base + k] = acc.get(base + k, 0) + d

    def define_morph(self, name, name_en, panel):
        self.morphs.append(Morph(name, name_en, panel, []))
        self._morph_acc.setdefault(name, {})

    # ---- 書き出し ----
    def save(self, path):
        self.resolve_bones()
        s = self.scale
        out = _Out()
        out.name, out.name_en, out.comment, out.comment_en = self.name, self.name_en, self.comment, self.comment_en
        out.positions = [tuple(p * s) for p in self.positions]
        out.normals = [tuple(n) for n in self.normals]
        out.uvs = self.uvs
        out.textures = self.textures
        for rb in self.rigids:
            rb.size = tuple(np.asarray(rb.size) * s)
            rb.pos = tuple(np.asarray(rb.pos) * s)
        for j in self.joints:
            j.pos = tuple(np.asarray(j.pos) * s)
        out.rigids, out.joints = self.rigids, self.joints
        out.weights = [(self.bone[a], self.bone[b] if b else -1, w) for a, b, w in self.weights]
        out.materials = self.materials
        for b in self.bones:
            b.pos = tuple(np.asarray(b.pos) * s)
            if not isinstance(b.tail, int):
                b.tail = tuple(np.asarray(b.tail) * s)
        out.bones = self.bones
        for mo in self.morphs:
            mo.offsets = [(vi, tuple(d * s)) for vi, d in sorted(self._morph_acc[mo.name].items())]
        out.morphs = self.morphs
        out.frames = self.frames
        pmx.write_pmx(path, out)


class _Out:
    pass


def _merge_seams(verts, nrm):
    """UV の継ぎ目で複製した頂点の法線をそろえ、影に筋が出ないようにする。"""
    key = np.round(verts * 1e5).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((inv.max() + 1, 3))
    np.add.at(acc, inv, nrm)
    return norm(acc[inv])


def mirror(p):
    p = np.array(p, float)
    p[..., 0] *= -1
    return p
