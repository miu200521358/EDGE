import os
import pickle
from vis import skeleton_render, smpl_joints, smpl_offsets, smpl_parents
from mlib.pmx.pmx_part import Material, DrawFlg, Bone, BoneFlg, Vertex, Bdef1, Face, DisplaySlotReference, DisplaySlot
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_writer import PmxWriter
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame, VmdBoneFrameTrees
from mlib.vmd.vmd_writer import VmdWriter
from mlib.core.math import MVector3D, MVector4D

file_path = "eval/motions/01/test_nc198287_1.pkl"
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

# 身長158cmプラグインより
MIKU_CM = 0.1259496
WIDTH = 0.1
NORMAL_VEC = MVector3D(0, 1, 0)

move_base_name = f"{file_name}_move"
rot_base_name = f"{file_name}_rot"

# トレース用モデルを作成する
move_model = PmxModel(f"{dir_path}/{move_base_name}.pmx")
move_model.model_name = move_base_name
move_model.initialize_display_slots()

rot_model = PmxModel(f"{dir_path}/{rot_base_name}.pmx")
rot_model.model_name = rot_base_name
rot_model.initialize_display_slots()

# 全ての親作成
move_root_bone = Bone(name="全ての親")
move_root_bone.bone_flg = BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_ROTATE | BoneFlg.IS_VISIBLE | BoneFlg.CAN_MANIPULATE
move_model.bones.append(move_root_bone)

rot_root_bone = Bone(name="全ての親")
rot_root_bone.bone_flg = BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_ROTATE | BoneFlg.IS_VISIBLE | BoneFlg.CAN_MANIPULATE
rot_model.bones.append(rot_root_bone)

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

# 表示枠作成
move_model.display_slots["Root"].references.append(
    DisplaySlotReference(display_index=0)
)
move_ds = DisplaySlot(name="Bone")

rot_model.display_slots["Root"].references.append(
    DisplaySlotReference(display_index=0)
)
rot_ds = DisplaySlot(name="Bone")

for n, (joint_name, joint_parent_index, offset) in enumerate(zip(smpl_joints, smpl_parents, smpl_offsets)):
    move_bone = Bone(name=joint_name)
    move_bone.bone_flg = BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_ROTATE | BoneFlg.IS_VISIBLE | BoneFlg.CAN_MANIPULATE
    if joint_parent_index >= 0:
        move_bone.parent_index = move_model.bones[smpl_joints[joint_parent_index]].index
    else:
        move_bone.parent_index = 0
    move_model.bones.append(move_bone)
    move_ds.references.append(DisplaySlotReference(display_index=move_bone.index))

    rot_bone = Bone(name=joint_name)
    rot_bone.bone_flg = BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_ROTATE | BoneFlg.IS_VISIBLE | BoneFlg.CAN_MANIPULATE
    parent_joint_position = MVector3D()
    if joint_parent_index >= 0:
        rot_bone.parent_index = rot_model.bones[smpl_joints[joint_parent_index]].index
        parent_joint_position = rot_model.bones[smpl_joints[joint_parent_index]].position
    else:
        rot_bone.parent_index = 0
    rot_bone.position = (MVector3D(offset[0], offset[1], -offset[2]) / MIKU_CM) + parent_joint_position
    rot_model.bones.append(rot_bone)
    rot_ds.references.append(DisplaySlotReference(display_index=rot_bone.index))

move_model.display_slots.append(move_ds)
rot_model.display_slots.append(rot_ds)

for model in (move_model, rot_model):
    for n, (joint_name, joint_parent_index, offset) in enumerate(zip(smpl_joints, smpl_parents, smpl_offsets)):
        bone = model.bones[joint_name]
        child_joint_indexes = [i for i, j in enumerate(smpl_parents) if j == n]
        if not child_joint_indexes:
            continue

        child_joint_index = child_joint_indexes[0]

        bone.bone_flg |= BoneFlg.TAIL_IS_BONE
        bone.tail_index = model.bones[smpl_joints[child_joint_index]].index

        from_pos = bone.position
        tail_pos = model.bones[smpl_joints[child_joint_index]].position

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
PmxWriter(rot_model, rot_model.path).save()

# ----------------------------------------------

move_motion = VmdMotion(f"{dir_path}/{move_base_name}.vmd")
rot_motion = VmdMotion(f"{dir_path}/{rot_base_name}.vmd")

for n in range(0, full_pose.shape[0]):
    pose = full_pose[n]
    for m, (joint_name, joint_parent_index, joint_pose) in enumerate(zip(smpl_joints, smpl_parents, pose)):
        # if joint_name not in ["root", "lhip", "rhip", "lknee", "rknee"]:
        #     continue

        bf = VmdBoneFrame(index=n, name=joint_name, register=True)
        bone_position = move_model.bones[joint_name].position

        parent_bone_position = MVector3D()
        parent_joint_position = MVector3D()
        if joint_parent_index >= 0:
            while joint_parent_index >= 0:
                parent_bone_position += move_model.bones[smpl_joints[joint_parent_index]].position
                parent_joint_position += move_motion.bones[smpl_joints[joint_parent_index]][n].position
                joint_parent_index = smpl_parents[joint_parent_index]
        bf.position = (MVector3D(joint_pose[0], joint_pose[2], -joint_pose[1]) / MIKU_CM) - parent_joint_position

        if n == 70:
            print(joint_name, joint_pose, parent_joint_position, bf.position)
            # print(joint_name, joint_pose)

        move_motion.append_bone_frame(bf)

VmdWriter(move_motion, move_motion.path, move_model.name).save()
