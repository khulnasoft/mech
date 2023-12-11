"""Collection of tests for axis-angle pose functions"""
# global
import aikit
import aikit_mech
import numpy as np

# local
from aikit_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_euler_pose_to_axis_angle_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.euler_pose_to_axis_angle_pose(aikit.array(ptd.euler_pose)),
        ptd.axis_angle_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.euler_pose_to_axis_angle_pose(aikit.array(ptd.batched_euler_pose))[0],
        ptd.axis_angle_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_mat_pose_to_rot_vec_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.mat_pose_to_rot_vec_pose(aikit.array(ptd.matrix_pose)),
        ptd.rot_vec_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.mat_pose_to_rot_vec_pose(aikit.array(ptd.batched_matrix_pose))[0],
        ptd.rot_vec_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_quaternion_pose_to_rot_vec_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.quaternion_pose_to_rot_vec_pose(aikit.array(ptd.quaternion_pose)),
        ptd.rot_vec_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.quaternion_pose_to_rot_vec_pose(
            aikit.array(ptd.batched_quaternion_pose)
        )[0],
        ptd.rot_vec_pose,
        atol=1e-6,
    )
    aikit.previous_backend()
