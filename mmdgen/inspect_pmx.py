"""既存の PMX の構成（材質・ボーン・モーフ・物理）を要約する。python -m mmdgen.inspect_pmx <file.pmx>"""

import struct
import sys
from collections import Counter


class Reader:
    def __init__(self, data):
        self.d, self.o = data, 0

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.d, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def text(self, enc):
        n = self.take("i")
        b = self.d[self.o:self.o + n]
        self.o += n
        return b.decode("utf-16-le" if enc == 0 else "utf-8", "replace")

    def idx(self, size, unsigned=False):
        f = {1: "b", 2: "h", 4: "i"}[size]
        return self.take(f.upper() if unsigned and size < 4 else f)


def read(path):
    r = Reader(open(path, "rb").read())
    assert r.take("4s") == b"PMX "
    ver = r.take("f")
    g = r.take(f"{r.take('B')}B")
    enc, adduv, vs, ts, ms, bs, mos, rs = g[:8]
    info = {"version": ver, "name": r.text(enc)}
    r.text(enc), r.text(enc), r.text(enc)

    nv = r.take("i")
    wtypes = Counter()
    for _ in range(nv):
        r.take("8f")
        if adduv:
            r.take(f"{4 * adduv}f")
        w = r.take("B")
        wtypes[w] += 1
        if w == 0:
            r.idx(bs)
        elif w == 1:
            r.idx(bs), r.idx(bs), r.take("f")
        elif w in (2, 4):
            [r.idx(bs) for _ in range(4)], r.take("4f")
        elif w == 3:
            r.idx(bs), r.idx(bs), r.take("f"), r.take("9f")
        r.take("f")
    info["vertices"] = nv
    info["weight_types"] = dict(wtypes)
    nf = r.take("i")
    r.o += nf * vs
    info["faces"] = nf // 3

    textures = [r.text(enc) for _ in range(r.take("i"))]
    info["textures"] = textures
    mats = []
    for _ in range(r.take("i")):
        name = r.text(enc)
        r.text(enc)
        diffuse = r.take("4f")
        spec = r.take("3f")
        power = r.take("f")
        amb = r.take("3f")
        flags = r.take("B")
        edge = r.take("4f")
        edge_size = r.take("f")
        tex = r.idx(ts)
        sph = r.idx(ts)
        sph_mode = r.take("B")
        shared = r.take("B")
        toon = r.take("B") if shared else r.idx(ts)
        r.text(enc)
        cnt = r.take("i")
        mats.append(dict(name=name, faces=cnt // 3, diffuse=diffuse, ambient=amb, spec=spec, power=power,
                         flags=flags, edge=edge, edge_size=edge_size,
                         tex=textures[tex] if tex >= 0 else None,
                         sphere=(textures[sph] if sph >= 0 else None, sph_mode),
                         toon=("shared", toon) if shared else (textures[toon] if toon >= 0 else None)))
    info["materials"] = mats

    bones = []
    for _ in range(r.take("i")):
        name = r.text(enc)
        r.text(enc)
        pos = r.take("3f")
        parent = r.idx(bs)
        r.take("i")
        flags = r.take("H")
        if flags & 1:
            r.idx(bs)
        else:
            r.take("3f")
        if flags & 0x300:
            r.idx(bs), r.take("f")
        if flags & 0x400:
            r.take("3f")
        if flags & 0x800:
            r.take("6f")
        if flags & 0x2000:
            r.take("i")
        if flags & 0x20:
            r.idx(bs), r.take("i"), r.take("f")
            for _ in range(r.take("i")):
                r.idx(bs)
                if r.take("B"):
                    r.take("6f")
        bones.append((name, pos, parent, flags))
    info["bones"] = bones

    morphs = []
    for _ in range(r.take("i")):
        name = r.text(enc)
        r.text(enc)
        panel, kind, n = r.take("B"), r.take("B"), r.take("i")
        size = {0: bs and None, 1: vs + 12, 2: bs + 28, 3: vs + 16, 4: vs + 16, 5: vs + 16, 6: vs + 16,
                7: vs + 16, 8: ms + 113}
        if kind == 0:
            r.o += n * (mos + 4)
        elif kind == 1:
            r.o += n * (vs + 12)
        elif kind == 2:
            r.o += n * (bs + 28)
        elif 3 <= kind <= 7:
            r.o += n * (vs + 16)
        elif kind == 8:
            r.o += n * (ms + 1 + 16 + 12 + 4 + 12 + 16 + 4 + 16 + 16 + 16)
        morphs.append((name, panel, kind, n))
    info["morphs"] = morphs

    for _ in range(r.take("i")):
        r.text(enc), r.text(enc), r.take("B")
        for _ in range(r.take("i")):
            k = r.take("B")
            r.idx(bs if k == 0 else mos)
    rigids = []
    for _ in range(r.take("i")):
        name = r.text(enc)
        r.text(enc)
        bone = r.idx(bs)
        group, mask, shape = r.take("B"), r.take("H"), r.take("B")
        size, pos, rot = r.take("3f"), r.take("3f"), r.take("3f")
        mass, ld, ad, rest, fric = r.take("5f")
        mode = r.take("B")
        rigids.append(dict(name=name, bone=bones[bone][0] if bone >= 0 else None, group=group, shape=shape,
                           size=size, mass=mass, damp=(ld, ad), mode=mode))
    info["rigids"] = rigids
    joints = []
    for _ in range(r.take("i")):
        name = r.text(enc)
        r.text(enc)
        r.take("B")
        a, b = r.idx(rs), r.idx(rs)
        vals = r.take("24f")
        joints.append(dict(name=name, a=a, b=b, rot_lim=(vals[12:15], vals[15:18]), spring=(vals[18:21], vals[21:24])))
    info["joints"] = joints
    return info


def summary(path):
    i = read(path)
    print(f"== {i['name']}  (PMX {i['version']})")
    print(f"頂点 {i['vertices']}  面 {i['faces']}  ウェイト種別 {i['weight_types']}")
    print(f"テクスチャ {len(i['textures'])}: {i['textures']}")
    print(f"材質 {len(i['materials'])}")
    for m in i["materials"]:
        d = ",".join(f"{x:.2f}" for x in m["diffuse"])
        a = ",".join(f"{x:.2f}" for x in m["ambient"])
        print(f"  {m['name'][:18]:18s} 面{m['faces']:6d} 拡散({d}) 環境({a}) flag{m['flags']:#04x} "
              f"エッジ{m['edge_size']:.2f} tex={m['tex']} sph={m['sphere']} toon={m['toon']}")
    names = [b[0] for b in i["bones"]]
    print(f"ボーン {len(names)}")
    print("  " + " ".join(names))
    kinds = Counter(k for _, _, k, _ in i["morphs"])
    print(f"モーフ {len(i['morphs'])} 種別 {dict(kinds)}")
    print("  " + " ".join(f"{n}({p})" for n, p, k, c in i["morphs"]))
    print(f"剛体 {len(i['rigids'])}  ジョイント {len(i['joints'])}")
    for rb in i["rigids"][:200]:
        print(f"  {rb['name'][:14]:14s} bone={rb['bone']} g{rb['group']} shape{rb['shape']} "
              f"size={tuple(round(x, 2) for x in rb['size'])} mass={rb['mass']:.2f} damp={tuple(round(x, 2) for x in rb['damp'])} mode{rb['mode']}")
    for j in i["joints"][:6]:
        print(f"  joint {j['name']} rot={j['rot_lim']} spring={j['spring']}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summary(p)
