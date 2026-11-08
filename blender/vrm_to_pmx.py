"""VRM（Neural4D などで作ったもの）を MMD 向けの PMX にする。

blender -b --python blender/vrm_to_pmx.py -- 入力.vrm 出力.pmx "モデル名"
VRM アドオンと mmd_tools が入った Blender で動かす。
骨の名前を MMD の標準名にし、足 IK と表情モーフ（まばたき・あいうえお など）を作る。
"""

import math
import os
import sys

import bpy
from mathutils import Matrix

args = sys.argv[sys.argv.index("--") + 1:]
src, dst, name = args[0], args[1], args[2] if len(args) > 2 else "model"

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.import_scene.vrm(filepath=src)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
meshes = [o for o in bpy.data.objects if o.type == "MESH"]

# ---- 骨の名前 ----
HUMAN = {
    "spine": "上半身", "chest": "上半身2", "upperChest": "上半身3", "neck": "首", "head": "頭",
    "leftEye": "左目", "rightEye": "右目",
}
for jp, en in (("左", "left"), ("右", "right")):
    HUMAN.update({
        en + "Shoulder": jp + "肩", en + "UpperArm": jp + "腕", en + "LowerArm": jp + "ひじ", en + "Hand": jp + "手首",
        en + "UpperLeg": jp + "足", en + "LowerLeg": jp + "ひざ", en + "Foot": jp + "足首", en + "Toes": jp + "つま先",
    })
    for finger, mmd in (("Thumb", "親指"), ("Index", "人指"), ("Middle", "中指"), ("Ring", "薬指"), ("Little", "小指")):
        first = 0 if finger == "Thumb" else 1
        for k, part in enumerate(("Proximal", "Intermediate", "Distal")):
            HUMAN[en + finger + part] = f"{jp}{mmd}{'０１２３'[first + k]}"

human = {b.bone: b.node.bone_name for b in arm.data.vrm_addon_extension.vrm0.humanoid.human_bones if b.node.bone_name}
rename = {human[k]: v for k, v in HUMAN.items() if k in human}
hips = human["hips"]
# 揺れもの（VRM のスプリングボーン）の根元。名前は変えないので、書き出したあとの物理付けに使う
swing = [b.bone_name for g in arm.data.vrm_addon_extension.vrm0.secondary_animation.bone_groups for b in g.bones]

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = arm.data.edit_bones


def merge_weights(child, parent):
    for o in meshes:
        g = o.vertex_groups.get(child)
        if g is None:
            continue
        p = o.vertex_groups.get(parent) or o.vertex_groups.new(name=parent)
        for v in o.data.vertices:
            for ge in v.groups:
                if ge.group == g.index and ge.weight > 0:
                    p.add([v.index], ge.weight, "ADD")
        o.vertex_groups.remove(g)


# 腕や脚の途中にある補助の骨（.001 など）は親にまとめ、IK の鎖を素直にする
main = set(rename)
for b in list(eb):
    if b.name in main or b.name == hips or b.name.startswith("Hair"):
        continue
    par = b.parent
    if par is not None and par.name in main and "." in b.name and b.name.split(".")[0] == par.name:
        kids = list(b.children)
        merge_weights(b.name, par.name)
        for k in kids:
            k.use_connect = False  # つないだままだと子の根元が親の先へ吸い寄せられる
            k.parent = par
        eb.remove(b)
for b in eb:
    if b.name in main:
        kids = [k for k in b.children if k.name in main]
        if len(kids) == 1 and (kids[0].head - b.head).length > 1e-4:
            b.tail = kids[0].head.copy()
# ウェイトの名前も骨と一緒に変わるので、先に骨だけ直す
for old, new in rename.items():
    eb[old].name = new
for o in meshes:
    for g in o.vertex_groups:
        if g.name in rename:
            g.name = rename[g.name]

root = eb[hips]
root.name = "センター"
lower = eb.get("DEF-Hips")
if lower is not None:
    lower.name = "下半身"
    eb["上半身"].parent = root
mother = eb.new("全ての親")
mother.head, mother.tail = (0, 0, 0), (0, 0, 0.1)
root.parent = mother
for jp in "左右":
    ankle, toe = eb[jp + "足首"], eb.get(jp + "つま先")
    ik = eb.new(jp + "足ＩＫ")
    ik.head = ankle.head.copy()
    ik.tail = ankle.head.copy()
    ik.tail.y += 0.08
    ik.parent = mother
    if toe is not None:
        tik = eb.new(jp + "つま先ＩＫ")
        tik.head = toe.head.copy()
        tik.tail = toe.head.copy()
        tik.tail.z -= 0.06
        tik.parent = ik
# T ポーズを MMD の A ポーズ（腕を 37° 下げる）にし、それを基本の姿勢にする。
# 形状キーごとに変形後の形を焼き付けてから、ポーズを骨の基本姿勢として適用する
bpy.ops.object.mode_set(mode="POSE")
for jp, s in (("左", 1), ("右", -1)):
    pb = arm.pose.bones[jp + "腕"]
    h = pb.head.copy()
    pb.matrix = Matrix.Translation(h) @ Matrix.Rotation(math.radians(37) * s, 4, "Y") @ Matrix.Translation(-h) @ pb.matrix
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()


def deformed(o):
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    co = [v.co.copy() for v in me.vertices]
    ev.to_mesh_clear()
    return co


for o in meshes:
    keys = o.data.shape_keys.key_blocks if o.data.shape_keys else None
    if keys:
        o.show_only_shape_key = True
        baked = []
        for i in range(len(keys)):
            o.active_shape_key_index = i
            dg.update()
            baked.append(deformed(o))
        o.show_only_shape_key = False
        for k, co in zip(keys, baked):
            for v, c in zip(k.data, co):
                v.co = c
    else:
        for v, c in zip(o.data.vertices, deformed(o)):
            v.co = c
    if o.data.has_custom_normals:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
bpy.ops.pose.armature_apply(selected=False)
for jp in "左右":
    c = arm.pose.bones[jp + "ひざ"].constraints.new("IK")
    c.target, c.subtarget, c.chain_count, c.use_tail = arm, jp + "足ＩＫ", 2, True
    knee = arm.pose.bones[jp + "ひざ"]
    knee.use_ik_limit_x = True
    knee.ik_min_x, knee.ik_max_x = 0.0087, 3.1416  # ひざは前にしか曲げない
    knee.lock_ik_y = knee.lock_ik_z = True
    if (jp + "つま先ＩＫ") in arm.pose.bones:
        c = arm.pose.bones[jp + "足首"].constraints.new("IK")
        c.target, c.subtarget, c.chain_count, c.use_tail = arm, jp + "つま先ＩＫ", 1, True
bpy.ops.object.mode_set(mode="OBJECT")

# ---- 表情：VRM の BlendShape を MMD の名前の形状キーにまとめる ----
PRESET = {"blink": "まばたき", "blink_l": "ウィンク", "blink_r": "ウィンク右", "joy": "笑い", "a": "あ", "i": "い",
          "u": "う", "e": "え", "o": "お", "angry": "怒り", "sorrow": "困る", "fun": "にこり"}
groups = arm.data.vrm_addon_extension.vrm0.blend_shape_master.blend_shape_groups
for g in groups:
    jp = PRESET.get(g.preset_name)
    if not jp or not g.binds:
        continue
    by_mesh = {}
    for b in g.binds:
        by_mesh.setdefault(b.mesh.mesh_object_name, []).append((b.index, b.weight))
    for mesh_name, binds in by_mesh.items():
        o = bpy.data.objects.get(mesh_name)
        if o is None or o.data.shape_keys is None:
            continue
        kb = o.data.shape_keys.key_blocks
        for k in kb:
            k.value = 0.0
        for idx, w in binds:
            if idx in kb:
                kb[idx].value = min(1.0, w)
        o.shape_key_add(name=jp, from_mix=True)
        for k in kb:
            k.value = 0.0
# 元の形状キーはモーフ数が多すぎるので MMD 名のもの以外は消す
keep = set(PRESET.values()) | {"Basis"}
for o in meshes:
    if o.data.shape_keys is None:
        continue
    for k in list(o.data.shape_keys.key_blocks):
        if k.name not in keep:
            o.shape_key_remove(k)

# ---- MMD モデルにして書き出す ----
bpy.ops.object.select_all(action="DESELECT")
for o in [arm] + meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.mmd_tools.convert_to_mmd_model()
mmd_root = next(o for o in bpy.data.objects if o.mmd_type == "ROOT")
# 色は MToon の基本色から取り、テカりは消す。目や顔の細かい部品には輪郭線を付けない
NO_EDGE = ("eye", "brow", "nose", "teeth", "tongue", "face")
for m in bpy.data.materials:
    mm = m.mmd_material
    try:
        f = list(m.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_factor)[:3]
    except AttributeError:
        f = [1.0, 1.0, 1.0]
    mm.diffuse_color = f
    mm.ambient_color = [c * 0.5 for c in f]
    mm.specular_color = (0.0, 0.0, 0.0)
    mm.is_shared_toon_texture, mm.shared_toon_texture = True, 0  # toon01：影が薄いトゥーン
    mm.enabled_toon_edge = not any(k in m.name.lower() for k in NO_EDGE)
    mm.edge_weight = 0.5
    mm.edge_color = (0.15, 0.12, 0.14, 1.0)
mmd_root.mmd_root.name = name
mmd_root.mmd_root.name_e = name
bpy.context.view_layer.objects.active = mmd_root
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
bpy.ops.mmd_tools.export_pmx(filepath=dst, scale=12.5, copy_textures_mode="OVERWRITE", sort_materials=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmdgen.pmx_physics import add_physics  # noqa: E402

print("剛体・ジョイント", add_physics(dst, swing))
print("WROTE", dst)
