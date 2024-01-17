import os
import pickle

from vis import smpl_joints, smpl_offsets, smpl_parents
from mlib.pmx.pmx_part import (
    Material,
    DrawFlg,
    Bone,
    BoneFlg,
    Vertex,
    Bdef1,
    Face,
    DisplaySlotReference,
    DisplaySlot,
)
from mlib.pmx.pmx_reader import PmxReader
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_writer import PmxWriter
from mlib.vmd.vmd_part import VmdIkOnOff
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame, VmdShowIkFrame
from mlib.vmd.vmd_writer import VmdWriter
from mlib.core.math import MVector3D, MVector4D, MQuaternion

file_path = "/mnt/d/MMD/MikuMikuDance_v926x64/Work/2024/20240115_誕生日御礼/nc255131/test_nc255131.pkl"
# file_path = f"eval/motions/{file_id}/test_{file_id}.pkl"
file_name = os.path.basename(file_path)[:-4]
dir_path = os.path.dirname(file_path)

with open(file_path, "rb") as file:
    data = pickle.load(file)

smpl_poses = data["smpl_poses"]
smpl_trans = data["smpl_trans"]
full_pose = data["full_pose"]

print("smpl_poses:", smpl_poses.shape)
print("smpl_trans:", smpl_trans.shape)
print("full_pose:", full_pose.shape)

PMX_CONNECTIONS = {
    "root": "下半身",  # 0
    "root2": "下半身先",  # 0
    "lhip": "左足",  # 1
    "rhip": "右足",  # 2
    "belly": "上半身",  # 3
    "lknee": "左ひざ",  # 4
    "rknee": "右ひざ",  # 5
    "spine": "上半身2",  # 6
    "lankle": "左足首",  # 7
    "rankle": "右足首",  # 8
    "chest": "首",  # 9
    "ltoes": "左つま先",  # 10
    "rtoes": "右つま先",  # 11
    "neck": "頭",  # 12
    "linshoulder": "左肩",  # 13
    "rinshoulder": "右肩",  # 14
    "head": "頭先",  # 15
    "lshoulder": "左腕",  # 16
    "rshoulder": "右腕",  # 17
    "lelbow": "左ひじ",  # 18
    "relbow": "右ひじ",  # 19
    "lwrist": "左手首",  # 20
    "rwrist": "右手首",  # 21
    "lhand": "左中指１",  # 22
    "rhand": "右中指１",  # 23
}


VMD_CONNECTIONS = {
    "root": {
        "direction": ("root", "root2"),
        "up": ("lhip", "rhip"),
        "cancel": (),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "belly": {
        "direction": ("belly", "spine"),
        "up": ("lshoulder", "rshoulder"),
        "cancel": (),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "spine": {
        "direction": ("spine", "chest"),
        "up": ("lshoulder", "rshoulder"),
        "cancel": ("belly",),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "chest": {
        "direction": ("spine", "chest"),
        "up": ("neck", "head"),
        "cancel": (
            "belly",
            "spine",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "neck": {
        "direction": ("neck", "head"),
        "up": ("linshoulder", "rinshoulder"),
        "cancel": (
            "belly",
            "spine",
            "chest",
        ),
        "invert": {
            "before": MVector3D(30, 0, 0),
            "after": MVector3D(),
        },
    },
    "linshoulder": {
        "direction": ("linshoulder", "lshoulder"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
        ),
        "invert": {
            "before": MVector3D(0, 0, -20),
            "after": MVector3D(),
        },
    },
    "lshoulder": {
        "direction": ("lshoulder", "lelbow"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
            "linshoulder",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "lelbow": {
        "direction": ("lelbow", "lwrist"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
            "linshoulder",
            "lshoulder",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "lwrist": {
        "direction": ("lwrist", "lhand"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
            "linshoulder",
            "lshoulder",
            "lelbow",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "rinshoulder": {
        "direction": ("rinshoulder", "rshoulder"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
        ),
        "invert": {
            "before": MVector3D(0, 0, 20),
            "after": MVector3D(),
        },
    },
    "rshoulder": {
        "direction": ("rshoulder", "relbow"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
            "rinshoulder",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "relbow": {
        "direction": ("relbow", "rwrist"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
            "rinshoulder",
            "rshoulder",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "rwrist": {
        "direction": ("rwrist", "rhand"),
        "up": ("spine", "chest"),
        "cancel": (
            "belly",
            "spine",
            "rinshoulder",
            "rshoulder",
            "relbow",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "lhip": {
        "direction": ("lhip", "lknee"),
        "up": ("lhip", "rhip"),
        "cancel": ("root",),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "lknee": {
        "direction": ("lknee", "lankle"),
        "up": ("lhip", "rhip"),
        "cancel": (
            "root",
            "lhip",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "lankle": {
        "direction": ("lankle", "ltoes"),
        "up": ("lhip", "rhip"),
        "cancel": (
            "root",
            "lhip",
            "lknee",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "rhip": {
        "direction": ("rhip", "rknee"),
        "up": ("lhip", "rhip"),
        "cancel": ("root",),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "rknee": {
        "direction": ("rknee", "rankle"),
        "up": ("lhip", "rhip"),
        "cancel": (
            "root",
            "rhip",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
    "rankle": {
        "direction": ("rankle", "rtoes"),
        "up": ("lhip", "rhip"),
        "cancel": (
            "root",
            "rhip",
            "rknee",
        ),
        "invert": {
            "before": MVector3D(),
            "after": MVector3D(),
        },
    },
}

PMX_REVERSE_CONNECTIONS = dict(zip(PMX_CONNECTIONS.values(), PMX_CONNECTIONS.keys()))

# 身長158cmプラグインより
MIKU_CM = 0.1259496
WIDTH = 0.1
NORMAL_VEC = MVector3D(0, 1, 0)

move_base_name = f"{file_name}_move"
rot_base_name = f"{file_name}_rot"
mmd_base_name = f"{file_name}_mmd"
miku_base_name = f"{file_name}_miku"

# トレース用モデルを作成する
move_model = PmxModel(f"{dir_path}/{move_base_name}.pmx")
move_model.model_name = move_base_name
move_model.initialize_display_slots()

rot_model = PmxModel(f"{dir_path}/{rot_base_name}.pmx")
rot_model.model_name = rot_base_name
rot_model.initialize_display_slots()

# トレース用モデルを読み込む
mmd_model = PmxModel(f"{dir_path}/{mmd_base_name}.pmx")
mmd_model.model_name = mmd_base_name
mmd_model.initialize_display_slots()

miku_model = PmxReader().read_by_filepath("eval/motions/bone_初音ミクVer2 準標準 見せパン 3.pmx")

# 全ての親作成
move_root_bone = Bone(name="全ての親")
move_root_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
move_model.bones.append(move_root_bone)

rot_root_bone = Bone(name="全ての親")
rot_root_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
rot_model.bones.append(rot_root_bone)

mmd_root_bone = Bone(name="全ての親")
mmd_root_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
mmd_model.bones.append(mmd_root_bone)

# 材質作成
move_material = Material(name="材質1")
move_material.diffuse = MVector4D(1, 1, 1, 1)
move_material.ambient = MVector3D(1, 0, 0)
move_material.draw_flg = DrawFlg.DOUBLE_SIDED_DRAWING
move_model.materials.append(move_material)

rot_material = Material(name="材質1")
rot_material.diffuse = MVector4D(1, 1, 1, 1)
rot_material.ambient = MVector3D(1, 0, 0)
rot_material.draw_flg = DrawFlg.DOUBLE_SIDED_DRAWING
rot_model.materials.append(rot_material)

mmd_material = Material(name="材質1")
mmd_material.diffuse = MVector4D(1, 1, 1, 1)
mmd_material.ambient = MVector3D(1, 0, 0)
mmd_material.draw_flg = DrawFlg.DOUBLE_SIDED_DRAWING
mmd_model.materials.append(mmd_material)

# 表示枠作成
move_model.display_slots["Root"].references.append(
    DisplaySlotReference(display_index=0)
)
move_ds = DisplaySlot(name="Bone")

rot_model.display_slots["Root"].references.append(DisplaySlotReference(display_index=0))
rot_ds = DisplaySlot(name="Bone")

mmd_model.display_slots["Root"].references.append(DisplaySlotReference(display_index=0))
mmd_ds = DisplaySlot(name="Bone")

mmd_center_bone = Bone(name="センター")
mmd_center_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
mmd_center_bone.parent_index = 0
mmd_center_bone.position = MVector3D(0, 8, 0)
mmd_model.bones.append(mmd_center_bone)
mmd_ds.references.append(DisplaySlotReference(display_index=mmd_center_bone.index))

for n, (joint_name, joint_parent_index, offset) in enumerate(
    zip(smpl_joints, smpl_parents, smpl_offsets)
):
    move_bone = Bone(name=joint_name)
    move_bone.bone_flg = (
        BoneFlg.CAN_TRANSLATE
        | BoneFlg.CAN_ROTATE
        | BoneFlg.IS_VISIBLE
        | BoneFlg.CAN_MANIPULATE
    )
    move_bone.parent_index = 0
    move_model.bones.append(move_bone)
    move_ds.references.append(DisplaySlotReference(display_index=move_bone.index))

    rot_bone = Bone(name=joint_name)
    rot_bone.bone_flg = (
        BoneFlg.CAN_TRANSLATE
        | BoneFlg.CAN_ROTATE
        | BoneFlg.IS_VISIBLE
        | BoneFlg.CAN_MANIPULATE
    )
    parent_joint_position = MVector3D()
    if joint_parent_index >= 0:
        rot_bone.parent_index = rot_model.bones[smpl_joints[joint_parent_index]].index
        parent_joint_position = rot_model.bones[
            smpl_joints[joint_parent_index]
        ].position
    else:
        rot_bone.parent_index = 0
    rot_bone.position = (
        MVector3D(offset[0], offset[1], offset[2]) / MIKU_CM
    ) + parent_joint_position
    rot_model.bones.append(rot_bone)
    rot_ds.references.append(DisplaySlotReference(display_index=rot_bone.index))

    mmd_bone = Bone(name=PMX_CONNECTIONS[joint_name])
    mmd_bone.bone_flg = (
        BoneFlg.CAN_TRANSLATE
        | BoneFlg.CAN_ROTATE
        | BoneFlg.IS_VISIBLE
        | BoneFlg.CAN_MANIPULATE
    )
    parent_joint_position = MVector3D()
    if joint_parent_index >= 0:
        if mmd_bone.name == "上半身":
            mmd_bone.parent_index = mmd_model.bones["センター"].index
        elif "肩" in mmd_bone.name:
            mmd_bone.parent_index = mmd_model.bones["上半身2"].index
        else:
            mmd_bone.parent_index = mmd_model.bones[
                PMX_CONNECTIONS[smpl_joints[joint_parent_index]]
            ].index
        parent_joint_position = mmd_model.bones[
            PMX_CONNECTIONS[smpl_joints[joint_parent_index]]
        ].position
    else:
        mmd_bone.parent_index = 1
        parent_joint_position = MVector3D(0, 8, 0)
    mmd_bone.position = (
        MVector3D(-offset[0], offset[1], -offset[2]) / MIKU_CM
    ) + parent_joint_position
    mmd_model.bones.append(mmd_bone)
    mmd_ds.references.append(DisplaySlotReference(display_index=mmd_bone.index))

move_root2_bone = Bone(name="root2")
move_root2_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
move_root2_bone.parent_index = 0
move_model.bones.append(move_root2_bone)
move_ds.references.append(DisplaySlotReference(display_index=move_root2_bone.index))

rot_root2_bone = Bone(name="root2")
rot_root2_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
rot_root2_bone.parent_index = 0
rot_root2_bone.position = (
    rot_model.bones["lhip"].position + rot_model.bones["rhip"].position
) / 2
rot_model.bones.append(rot_root2_bone)
rot_ds.references.append(DisplaySlotReference(display_index=rot_root2_bone.index))

mmd_root2_bone = Bone(name=PMX_CONNECTIONS["root2"])
mmd_root2_bone.bone_flg = (
    BoneFlg.CAN_TRANSLATE
    | BoneFlg.CAN_ROTATE
    | BoneFlg.IS_VISIBLE
    | BoneFlg.CAN_MANIPULATE
)
mmd_root2_bone.parent_index = 0
mmd_root2_bone.position = (
    mmd_model.bones[PMX_CONNECTIONS["lhip"]].position
    + mmd_model.bones[PMX_CONNECTIONS["rhip"]].position
) / 2
mmd_model.bones.append(mmd_root2_bone)
mmd_ds.references.append(DisplaySlotReference(display_index=mmd_root2_bone.index))

move_model.display_slots.append(move_ds)
rot_model.display_slots.append(rot_ds)
mmd_model.display_slots.append(mmd_ds)

for model, is_mmd in ((move_model, False), (rot_model, False), (mmd_model, True)):
    for n, (joint_name, joint_parent_index, offset) in enumerate(
        zip(smpl_joints, smpl_parents, smpl_offsets)
    ):
        bone = (
            model.bones[PMX_CONNECTIONS[joint_name]]
            if is_mmd
            else model.bones[joint_name]
        )
        child_joint_indexes = [i for i, j in enumerate(smpl_parents) if j == n]
        if not child_joint_indexes:
            continue

        child_joint_index = child_joint_indexes[0]
        child_joint_name = (
            PMX_CONNECTIONS[smpl_joints[child_joint_index]]
            if is_mmd
            else smpl_joints[child_joint_index]
        )

        bone.bone_flg |= BoneFlg.TAIL_IS_BONE
        bone.tail_index = model.bones[child_joint_name].index

        from_pos = bone.position
        tail_pos = model.bones[child_joint_name].position

        # FROMからTOまで面を生成
        v1 = Vertex()
        v1.position = from_pos
        v1.normal = NORMAL_VEC
        v1.deform = Bdef1(bone.index)
        model.vertices.append(v1)

        v2 = Vertex()
        v2.position = tail_pos
        v2.normal = NORMAL_VEC
        v2.deform = Bdef1(bone.tail_index)
        model.vertices.append(v2)

        v3 = Vertex()
        v3.position = from_pos + MVector3D(WIDTH, 0, 0)
        v3.normal = NORMAL_VEC
        v3.deform = Bdef1(bone.index)
        model.vertices.append(v3)

        v4 = Vertex()
        v4.position = tail_pos + MVector3D(WIDTH, 0, 0)
        v4.normal = NORMAL_VEC
        v4.deform = Bdef1(bone.tail_index)
        model.vertices.append(v4)

        v5 = Vertex()
        v5.position = from_pos + MVector3D(WIDTH, WIDTH, 0)
        v5.normal = NORMAL_VEC
        v5.deform = Bdef1(bone.index)
        model.vertices.append(v5)

        v6 = Vertex()
        v6.position = tail_pos + MVector3D(WIDTH, WIDTH, 0)
        v6.normal = NORMAL_VEC
        v6.deform = Bdef1(bone.tail_index)
        model.vertices.append(v6)

        v7 = Vertex()
        v7.position = from_pos + MVector3D(0, 0, WIDTH)
        v7.normal = NORMAL_VEC
        v7.deform = Bdef1(bone.index)
        model.vertices.append(v7)

        v8 = Vertex()
        v8.position = tail_pos + MVector3D(0, 0, WIDTH)
        v8.normal = NORMAL_VEC
        v8.deform = Bdef1(bone.tail_index)
        model.vertices.append(v8)

        model.faces.append(
            Face(vertex_index0=v1.index, vertex_index1=v2.index, vertex_index2=v3.index)
        )
        model.faces.append(
            Face(vertex_index0=v3.index, vertex_index1=v2.index, vertex_index2=v4.index)
        )
        model.faces.append(
            Face(vertex_index0=v3.index, vertex_index1=v4.index, vertex_index2=v5.index)
        )
        model.faces.append(
            Face(vertex_index0=v5.index, vertex_index1=v4.index, vertex_index2=v6.index)
        )
        model.faces.append(
            Face(vertex_index0=v5.index, vertex_index1=v6.index, vertex_index2=v7.index)
        )
        model.faces.append(
            Face(vertex_index0=v7.index, vertex_index1=v6.index, vertex_index2=v8.index)
        )
        model.faces.append(
            Face(vertex_index0=v7.index, vertex_index1=v8.index, vertex_index2=v1.index)
        )
        model.faces.append(
            Face(vertex_index0=v1.index, vertex_index1=v8.index, vertex_index2=v2.index)
        )

        model.materials["材質1"].vertices_count += 24

PmxWriter(move_model, move_model.path).save()
# PmxWriter(rot_model, rot_model.path).save()
PmxWriter(mmd_model, mmd_model.path).save()

# ----------------------------------------------

move_motion = VmdMotion(f"{dir_path}/{move_base_name}.vmd")

for n in range(0, full_pose.shape[0]):
    pose = full_pose[n]
    for m, (joint_name, joint_parent_index, joint_pose) in enumerate(
        zip(smpl_joints, smpl_parents, pose)
    ):
        # if joint_name not in ["root", "lhip", "rhip", "lknee", "rknee"]:
        #     continue

        move_bf = VmdBoneFrame(index=n, name=joint_name, register=True)
        bone_position = move_model.bones[joint_name].position

        parent_bone_position = MVector3D()
        parent_joint_position = MVector3D()
        if joint_parent_index >= 0:
            while joint_parent_index >= 0:
                parent_bone_position += move_model.bones[
                    smpl_joints[joint_parent_index]
                ].position
                parent_joint_position += move_motion.bones[
                    smpl_joints[joint_parent_index]
                ][n].position
                joint_parent_index = smpl_parents[joint_parent_index]
        move_bf.position = (
            MVector3D(-joint_pose[0], joint_pose[2], -joint_pose[1]) / MIKU_CM
        )

        if n == 70:
            print(joint_name, joint_pose, parent_joint_position, move_bf.position)
            # print(joint_name, joint_pose)

        move_motion.append_bone_frame(move_bf)

        if joint_name == "rhip":
            left_move_bf = move_motion.bones["lhip"][n]

            root2_bf = VmdBoneFrame(index=n, name="root2", register=True)
            root2_bf.position = (left_move_bf.position + move_bf.position) / 2
            move_motion.append_bone_frame(root2_bf)

VmdWriter(move_motion, move_motion.path, move_model.name).save()

# ----------------------------------------------

miku_motion = VmdMotion(f"{dir_path}/{miku_base_name}.vmd")
# kf = VmdShowIkFrame(index=0, register=True, show=True)
# kf.iks.append(VmdIkOnOff(name="左足ＩＫ", onoff=False))
# kf.iks.append(VmdIkOnOff(name="右足ＩＫ", onoff=False))
# kf.iks.append(VmdIkOnOff(name="左つま先ＩＫ", onoff=False))
# kf.iks.append(VmdIkOnOff(name="右つま先ＩＫ", onoff=False))
# miku_motion.show_iks.append(kf)

root_bf = VmdBoneFrame(name="全ての親", index=0, register=True)
root_bf.position = MVector3D(0, -7, 0)
root_bf.rotation = MQuaternion.from_euler_degrees(MVector3D(0, 180, 0))
miku_motion.append_bone_frame(root_bf)

for target_bone_name, vmd_params in VMD_CONNECTIONS.items():
    if "direction" not in vmd_params:
        continue
    direction_from_name = vmd_params["direction"][0]
    direction_to_name = vmd_params["direction"][1]
    up_from_name = vmd_params["up"][0]
    up_to_name = vmd_params["up"][1]
    cross_from_name = (
        vmd_params["cross"][0] if "cross" in vmd_params else vmd_params["direction"][0]
    )
    cross_to_name = (
        vmd_params["cross"][1] if "cross" in vmd_params else vmd_params["direction"][1]
    )
    cancel_names = vmd_params["cancel"]
    invert_qq = MQuaternion.from_euler_degrees(vmd_params["invert"]["before"])

    for mov_bf in move_motion.bones[direction_from_name]:
        if (
            mov_bf.index not in move_motion.bones[direction_from_name]
            or mov_bf.index not in move_motion.bones[direction_to_name]
        ):
            # キーがない場合、スルーする
            continue

        bone_direction = (
            miku_model.bones[PMX_CONNECTIONS[direction_to_name]].position
            - miku_model.bones[PMX_CONNECTIONS[direction_from_name]].position
        ).normalized()

        bone_up = (
            miku_model.bones[PMX_CONNECTIONS[up_to_name]].position
            - miku_model.bones[PMX_CONNECTIONS[up_from_name]].position
        ).normalized()
        bone_cross = (
            miku_model.bones[PMX_CONNECTIONS[cross_to_name]].position
            - miku_model.bones[PMX_CONNECTIONS[cross_from_name]].position
        ).normalized()
        bone_cross_vec: MVector3D = bone_up.cross(bone_cross).normalized()

        initial_qq = MQuaternion.from_direction(bone_direction, bone_cross_vec)

        direction_from_abs_pos = move_motion.bones[direction_from_name][
            mov_bf.index
        ].position
        direction_to_abs_pos = move_motion.bones[direction_to_name][
            mov_bf.index
        ].position
        direction: MVector3D = (
            direction_to_abs_pos - direction_from_abs_pos
        ).normalized()

        up_from_abs_pos = move_motion.bones[up_from_name][mov_bf.index].position
        up_to_abs_pos = move_motion.bones[up_to_name][mov_bf.index].position
        up: MVector3D = (up_to_abs_pos - up_from_abs_pos).normalized()

        cross_from_abs_pos = move_motion.bones[cross_from_name][mov_bf.index].position
        cross_to_abs_pos = move_motion.bones[cross_to_name][mov_bf.index].position
        cross: MVector3D = (cross_to_abs_pos - cross_from_abs_pos).normalized()

        motion_cross_vec: MVector3D = up.cross(cross).normalized()
        motion_qq = MQuaternion.from_direction(direction, motion_cross_vec)

        miku_bone_name = PMX_CONNECTIONS[target_bone_name]

        cancel_qq = MQuaternion()
        for cancel_name in cancel_names:
            cancel_qq *= miku_motion.bones[PMX_CONNECTIONS[cancel_name]][
                mov_bf.index
            ].rotation

        bf = VmdBoneFrame(name=miku_bone_name, index=mov_bf.index, register=True)
        bf.rotation = cancel_qq.inverse() * motion_qq * initial_qq.inverse() * invert_qq
        miku_motion.append_bone_frame(bf)

        if target_bone_name == "root":
            center_bf = VmdBoneFrame(name="センター", index=mov_bf.index, register=True)
            center_bf.position = move_motion.bones[target_bone_name][
                mov_bf.index
            ].position - MVector3D(0, 8, 0)
            miku_motion.append_bone_frame(center_bf)

VmdWriter(miku_motion, miku_motion.path, "EdgeMotionModel").save()

# # ----------------------------------------------

# rot_motion = VmdMotion(f"{dir_path}/{rot_base_name}.vmd")

# for target_bone_name, vmd_params in VMD_CONNECTIONS.items():
#     if "direction" not in vmd_params:
#         continue
#     direction_from_name = vmd_params["direction"][0]
#     direction_to_name = vmd_params["direction"][1]
#     up_from_name = vmd_params["up"][0]
#     up_to_name = vmd_params["up"][1]
#     cross_from_name = vmd_params["cross"][0] if "cross" in vmd_params else vmd_params["direction"][0]
#     cross_to_name = vmd_params["cross"][1] if "cross" in vmd_params else vmd_params["direction"][1]
#     cancel_names = vmd_params["cancel"]
#     invert_qq = MQuaternion.from_euler_degrees(vmd_params["invert"]["before"])

#     for mov_bf in move_motion.bones[direction_from_name]:
#         if (
#             mov_bf.index not in move_motion.bones[direction_from_name]
#             or mov_bf.index not in move_motion.bones[direction_to_name]
#         ):
#             # キーがない場合、スルーする
#             continue

#         bone_direction = (
#             rot_model.bones[direction_to_name].position - rot_model.bones[direction_from_name].position
#         ).normalized()

#         bone_up = (rot_model.bones[up_to_name].position - rot_model.bones[up_from_name].position).normalized()
#         bone_cross = (rot_model.bones[cross_to_name].position - rot_model.bones[cross_from_name].position).normalized()
#         bone_cross_vec: MVector3D = bone_up.cross(bone_cross).normalized()

#         initial_qq = MQuaternion.from_direction(bone_direction, bone_cross_vec)

#         direction_from_abs_pos = move_motion.bones[direction_from_name][mov_bf.index].position
#         direction_to_abs_pos = move_motion.bones[direction_to_name][mov_bf.index].position
#         direction: MVector3D = (direction_to_abs_pos - direction_from_abs_pos).normalized()

#         up_from_abs_pos = move_motion.bones[up_from_name][mov_bf.index].position
#         up_to_abs_pos = move_motion.bones[up_to_name][mov_bf.index].position
#         up: MVector3D = (up_to_abs_pos - up_from_abs_pos).normalized()

#         cross_from_abs_pos = move_motion.bones[cross_from_name][mov_bf.index].position
#         cross_to_abs_pos = move_motion.bones[cross_to_name][mov_bf.index].position
#         cross: MVector3D = (cross_to_abs_pos - cross_from_abs_pos).normalized()

#         motion_cross_vec: MVector3D = up.cross(cross).normalized()
#         motion_qq = MQuaternion.from_direction(direction, motion_cross_vec)

#         cancel_qq = MQuaternion()
#         for cancel_name in cancel_names:
#             cancel_qq *= rot_motion.bones[cancel_name][mov_bf.index].rotation

#         bf = VmdBoneFrame(name=target_bone_name, index=mov_bf.index, register=True)
#         bf.rotation = cancel_qq.inverse() * motion_qq * initial_qq.inverse() * invert_qq

#         if target_bone_name == "root":
#             bf.position = move_motion.bones[target_bone_name][mov_bf.index].position

#         rot_motion.append_bone_frame(bf)


# VmdWriter(rot_motion, rot_motion.path, rot_model.name).save()

# # ----------------------------------------------

# mmd_motion = VmdMotion(f"{dir_path}/{mmd_base_name}.vmd")

# for target_bone_name, vmd_params in VMD_CONNECTIONS.items():
#     if "direction" not in vmd_params:
#         continue
#     direction_from_name = vmd_params["direction"][0]
#     direction_to_name = vmd_params["direction"][1]
#     up_from_name = vmd_params["up"][0]
#     up_to_name = vmd_params["up"][1]
#     cross_from_name = (
#         vmd_params["cross"][0] if "cross" in vmd_params else vmd_params["direction"][0]
#     )
#     cross_to_name = (
#         vmd_params["cross"][1] if "cross" in vmd_params else vmd_params["direction"][1]
#     )
#     cancel_names = vmd_params["cancel"]
#     invert_qq = MQuaternion.from_euler_degrees(vmd_params["invert"]["before"])

#     for mov_bf in move_motion.bones[direction_from_name]:
#         if (
#             mov_bf.index not in move_motion.bones[direction_from_name]
#             or mov_bf.index not in move_motion.bones[direction_to_name]
#         ):
#             # キーがない場合、スルーする
#             continue

#         bone_direction = (
#             mmd_model.bones[PMX_CONNECTIONS[direction_to_name]].position
#             - mmd_model.bones[PMX_CONNECTIONS[direction_from_name]].position
#         ).normalized()

#         bone_up = (
#             mmd_model.bones[PMX_CONNECTIONS[up_to_name]].position
#             - mmd_model.bones[PMX_CONNECTIONS[up_from_name]].position
#         ).normalized()
#         bone_cross = (
#             mmd_model.bones[PMX_CONNECTIONS[cross_to_name]].position
#             - mmd_model.bones[PMX_CONNECTIONS[cross_from_name]].position
#         ).normalized()
#         bone_cross_vec: MVector3D = bone_up.cross(bone_cross).normalized()

#         initial_qq = MQuaternion.from_direction(bone_direction, bone_cross_vec)

#         direction_from_abs_pos = move_motion.bones[direction_from_name][
#             mov_bf.index
#         ].position
#         direction_to_abs_pos = move_motion.bones[direction_to_name][
#             mov_bf.index
#         ].position
#         direction: MVector3D = (
#             direction_to_abs_pos - direction_from_abs_pos
#         ).normalized()

#         up_from_abs_pos = move_motion.bones[up_from_name][mov_bf.index].position
#         up_to_abs_pos = move_motion.bones[up_to_name][mov_bf.index].position
#         up: MVector3D = (up_to_abs_pos - up_from_abs_pos).normalized()

#         cross_from_abs_pos = move_motion.bones[cross_from_name][mov_bf.index].position
#         cross_to_abs_pos = move_motion.bones[cross_to_name][mov_bf.index].position
#         cross: MVector3D = (cross_to_abs_pos - cross_from_abs_pos).normalized()

#         motion_cross_vec: MVector3D = up.cross(cross).normalized()
#         motion_qq = MQuaternion.from_direction(direction, motion_cross_vec)

#         mmd_bone_name = PMX_CONNECTIONS[target_bone_name]

#         cancel_qq = MQuaternion()
#         for cancel_name in cancel_names:
#             cancel_qq *= mmd_motion.bones[PMX_CONNECTIONS[cancel_name]][
#                 mov_bf.index
#             ].rotation

#         bf = VmdBoneFrame(name=mmd_bone_name, index=mov_bf.index, register=True)
#         bf.rotation = cancel_qq.inverse() * motion_qq * initial_qq.inverse() * invert_qq
#         mmd_motion.append_bone_frame(bf)

#         if target_bone_name == "root":
#             center_bf = VmdBoneFrame(name="センター", index=mov_bf.index, register=True)
#             center_bf.position = move_motion.bones[target_bone_name][
#                 mov_bf.index
#             ].position - MVector3D(0, 8, 0)
#             mmd_motion.append_bone_frame(center_bf)

# for rot_bfs in rot_motion.bones:
#     if rot_bfs.name not in PMX_CONNECTIONS:
#         continue
#     for rot_bf in rot_bfs:
#         bone_name = PMX_CONNECTIONS[rot_bf.name]
#         bf = VmdBoneFrame(name=bone_name, index=rot_bf.index, register=True)

#         bf.rotation = rot_bf.rotation
#         bf.rotation.z *= -1
#         # bf.rotation.z *= -1
#         mmd_motion.append_bone_frame(bf)

#         if rot_bf.name == "root":
#             center_bf = VmdBoneFrame(name="センター", index=rot_bf.index, register=True)
#             center_bf.position = rot_bf.position
#             mmd_motion.append_bone_frame(center_bf)

# VmdWriter(mmd_motion, mmd_motion.path, "EdgeMotionModel").save()

# Beep音を鳴らす
os.system('echo -e "\a"')
