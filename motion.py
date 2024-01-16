import os
import pickle
from vis import skeleton_render, smpl_joints, smpl_offsets, smpl_parents
from mlib.pmx.pmx_part import Material, DrawFlg, Bone, BoneFlg, Vertex, Bdef1, Face, DisplaySlotReference, DisplaySlot
from mlib.pmx.pmx_collection import PmxModel
from mlib.pmx.pmx_writer import PmxWriter
from mlib.vmd.vmd_collection import VmdMotion, VmdBoneFrame
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

base_name = f"{file_name}_move"

# トレース用モデルを作成する
move_model = PmxModel(f"{dir_path}/{base_name}.pmx")
move_model.model_name = base_name
move_model.initialize_display_slots()

# 材質作成
material = Material(name="材質1")
material.diffuse = MVector4D(1, 1, 1, 1)
material.ambient = MVector3D(1, 0, 0)
material.draw_flg = DrawFlg.DOUBLE_SIDED_DRAWING
move_model.materials.append(material)

# 表示枠作成
move_model.display_slots["Root"].references.append(
    DisplaySlotReference(display_index=0)
)
ds = DisplaySlot(name="Bone")

for n, (joint_name, joint_parent_index, offset) in enumerate(zip(smpl_joints, smpl_parents, smpl_offsets)):
    bone = Bone(name=joint_name)
    bone.bone_flg = BoneFlg.CAN_TRANSLATE | BoneFlg.CAN_ROTATE | BoneFlg.IS_VISIBLE | BoneFlg.CAN_MANIPULATE
    parent_position = MVector3D()
    if joint_parent_index >= 0:
        bone.parent_index = move_model.bones[smpl_joints[joint_parent_index]].index
        parent_position = move_model.bones[smpl_joints[joint_parent_index]].position
    bone.position = (MVector3D(offset[0], offset[2], offset[1]) / MIKU_CM) + parent_position
    move_model.bones.append(bone)
    ds.references.append(DisplaySlotReference(display_index=bone.index))

move_model.display_slots.append(ds)

for n, (joint_name, joint_parent_index, offset) in enumerate(zip(smpl_joints, smpl_parents, smpl_offsets)):
    bone = move_model.bones[joint_name]
    child_joint_indexes = [i for i, j in enumerate(smpl_parents) if j == n]
    if not child_joint_indexes:
        continue

    bone.bone_flg |= BoneFlg.TAIL_IS_BONE
    bone.tail_index = move_model.bones[smpl_joints[child_joint_indexes[0]]].index

    from_pos = bone.position
    tail_pos = move_model.bones[smpl_joints[child_joint_indexes[0]]].position

    # FROMからTOまで面を生成
    v1 = Vertex()
    v1.position = from_pos
    v1.normal = NORMAL_VEC
    v1.deform = Bdef1(bone.index)
    move_model.vertices.append(v1)

    v2 = Vertex()
    v2.position = tail_pos
    v2.normal = NORMAL_VEC
    v2.deform = Bdef1(bone.index)
    move_model.vertices.append(v2)

    v3 = Vertex()
    v3.position = from_pos + MVector3D(WIDTH, 0, 0)
    v3.normal = NORMAL_VEC
    v3.deform = Bdef1(bone.index)
    move_model.vertices.append(v3)

    v4 = Vertex()
    v4.position = tail_pos + MVector3D(WIDTH, 0, 0)
    v4.normal = NORMAL_VEC
    v4.deform = Bdef1(bone.index)
    move_model.vertices.append(v4)

    v5 = Vertex()
    v5.position = from_pos + MVector3D(WIDTH, WIDTH, 0)
    v5.normal = NORMAL_VEC
    v5.deform = Bdef1(bone.index)
    move_model.vertices.append(v5)

    v6 = Vertex()
    v6.position = tail_pos + MVector3D(WIDTH, WIDTH, 0)
    v6.normal = NORMAL_VEC
    v6.deform = Bdef1(bone.index)
    move_model.vertices.append(v6)

    v7 = Vertex()
    v7.position = from_pos + MVector3D(0, 0, WIDTH)
    v7.normal = NORMAL_VEC
    v7.deform = Bdef1(bone.index)
    move_model.vertices.append(v7)

    v8 = Vertex()
    v8.position = tail_pos + MVector3D(0, 0, WIDTH)
    v8.normal = NORMAL_VEC
    v8.deform = Bdef1(bone.index)
    move_model.vertices.append(v8)

    move_model.faces.append(
        Face(vertex_index0=v1.index, vertex_index1=v2.index, vertex_index2=v3.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v3.index, vertex_index1=v2.index, vertex_index2=v4.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v3.index, vertex_index1=v4.index, vertex_index2=v5.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v5.index, vertex_index1=v4.index, vertex_index2=v6.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v5.index, vertex_index1=v6.index, vertex_index2=v7.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v7.index, vertex_index1=v6.index, vertex_index2=v8.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v7.index, vertex_index1=v8.index, vertex_index2=v1.index)
    )
    move_model.faces.append(
        Face(vertex_index0=v1.index, vertex_index1=v8.index, vertex_index2=v2.index)
    )

    material.vertices_count += 24

PmxWriter(move_model, move_model.path).save()

# ----------------------------------------------

move_motion = VmdMotion(f"{dir_path}/{base_name}.vmd")

for n in range(0, full_pose.shape[0]):
    pose = full_pose[n]
    for m, (joint_name, joint_parent_index, joint_pose) in enumerate(zip(smpl_joints, smpl_parents, pose)):
        # print(joint_name, joint_pose)
        bf = VmdBoneFrame(index=n, name=joint_name, register=True)

        parent_position = MVector3D()
        if joint_parent_index >= 0:
            parent_position = move_motion.bones[smpl_joints[joint_parent_index]][n].position

        bf.position = (MVector3D(joint_pose[0], joint_pose[2], joint_pose[1]) / MIKU_CM) + move_model.bones[joint_name].position
        move_motion.append_bone_frame(bf)

VmdWriter(move_motion, move_motion.path, move_model.name).save()
