"""PMX 2.0 の書き出し。テクスチャ・剛体・ジョイントは使わない最小構成。"""

import struct

BONE_TAIL_INDEX = 0x0001
BONE_ROTATE = 0x0002
BONE_MOVE = 0x0004
BONE_VISIBLE = 0x0008
BONE_OPERABLE = 0x0010
BONE_IK = 0x0020
BONE_INHERIT_ROT = 0x0100

MAT_DOUBLE = 0x01
MAT_GROUND_SHADOW = 0x02
MAT_SHADOW_MAP = 0x04
MAT_SELF_SHADOW = 0x08
MAT_EDGE = 0x10


def _text(s):
    b = s.encode("utf-16-le")
    return struct.pack("<i", len(b)) + b


def write_pmx(path, m):
    """m: Model（builder.py）を PMX として書き出す。座標は MMD 単位に変換済みのものを渡す。"""
    nv = len(m.positions)
    vfmt, vsize = ("<H", 2) if nv < 65536 else ("<i", 4)
    out = bytearray()
    out += b"PMX " + struct.pack("<f", 2.0)
    out += bytes([8, 0, 0, vsize, 1, 1, 2, 2, 2])
    out += _text(m.name) + _text(m.name_en) + _text(m.comment) + _text(m.comment_en)

    out += struct.pack("<i", nv)
    for p, n, uv, w in zip(m.positions, m.normals, m.uvs, m.weights):
        out += struct.pack("<8f", p[0], p[1], p[2], n[0], n[1], n[2], uv[0], uv[1])
        b1, b2, w1 = w
        if b2 < 0 or w1 >= 0.999:
            out += struct.pack("<Bh", 0, b1)
        else:
            out += struct.pack("<Bhhf", 1, b1, b2, w1)
        out += struct.pack("<f", 1.0)

    mats = [mt for mt in m.materials if mt.faces]
    out += struct.pack("<i", sum(len(mt.faces) * 3 for mt in mats))
    for mt in mats:
        for f in mt.faces:
            out += struct.pack(vfmt[0] + vfmt[1] * 3, *f)

    out += struct.pack("<i", len(m.textures))
    for t in m.textures:
        out += _text(t)
    tex_index = {t: i for i, t in enumerate(m.textures)}

    out += struct.pack("<i", len(mats))
    for mt in mats:
        out += _text(mt.name) + _text(mt.name_en)
        out += struct.pack("<4f", *mt.diffuse, mt.alpha)
        out += struct.pack("<3f", *mt.specular) + struct.pack("<f", mt.power)
        out += struct.pack("<3f", *mt.ambient)
        out += struct.pack("<B", mt.flags)
        out += struct.pack("<4f", *mt.edge_color, 1.0) + struct.pack("<f", mt.edge_size)
        out += struct.pack("<bbB", tex_index.get(mt.tex, -1), tex_index.get(mt.sphere, -1),
                           mt.sphere_mode if mt.sphere else 0)
        if mt.toon_tex:
            out += struct.pack("<Bb", 0, tex_index[mt.toon_tex])
        else:
            out += struct.pack("<BB", 1, mt.toon)
        out += _text("")
        out += struct.pack("<i", len(mt.faces) * 3)

    out += struct.pack("<i", len(m.bones))
    for b in m.bones:
        out += _text(b.name) + _text(b.name_en)
        out += struct.pack("<3f", *b.pos)
        out += struct.pack("<hiH", b.parent, 0, b.flags)
        if b.flags & BONE_TAIL_INDEX:
            out += struct.pack("<h", b.tail)
        else:
            out += struct.pack("<3f", *b.tail)
        if b.flags & BONE_INHERIT_ROT:
            out += struct.pack("<hf", b.inherit[0], b.inherit[1])
        if b.flags & BONE_IK:
            target, loop, limit, links = b.ik
            out += struct.pack("<hifi", target, loop, limit, len(links))
            for idx, lim in links:
                if lim is None:
                    out += struct.pack("<hB", idx, 0)
                else:
                    out += struct.pack("<hB6f", idx, 1, *lim[0], *lim[1])

    out += struct.pack("<i", len(m.morphs))
    for mo in m.morphs:
        out += _text(mo.name) + _text(mo.name_en)
        out += struct.pack("<BBi", mo.panel, 1, len(mo.offsets))
        for vi, d in mo.offsets:
            out += struct.pack(vfmt, vi) + struct.pack("<3f", *d)

    out += struct.pack("<i", len(m.frames))
    for name, name_en, special, elems in m.frames:
        out += _text(name) + _text(name_en) + struct.pack("<Bi", special, len(elems))
        for kind, idx in elems:
            out += struct.pack("<Bh", kind, idx)

    out += struct.pack("<i", len(m.rigids))
    for rb in m.rigids:
        out += _text(rb.name) + _text(rb.name)
        out += struct.pack("<hBHB", rb.bone, rb.group, rb.mask, rb.shape)
        out += struct.pack("<9f", *rb.size, *rb.pos, *rb.rot)
        out += struct.pack("<5fB", rb.mass, rb.damp[0], rb.damp[1], 0.0, 0.5, rb.mode)
    out += struct.pack("<i", len(m.joints))
    for j in m.joints:
        out += _text(j.name) + _text(j.name)
        out += struct.pack("<Bhh", 0, j.a, j.b)
        out += struct.pack("<6f", *j.pos, *j.rot)
        out += struct.pack("<6f", 0, 0, 0, 0, 0, 0)
        out += struct.pack("<6f", *j.rot_lo, *j.rot_hi)
        out += struct.pack("<6f", 0, 0, 0, *j.spring)

    with open(path, "wb") as f:
        f.write(out)
