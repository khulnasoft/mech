"""Collection of tests for matrix pose functions"""
# global
import aikit
import aikit_mech
import aikit.functional.backends.numpy as aikit_np
import numpy as np

# local
from aikit_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_axis_angle_pose_to_mat_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.axis_angle_pose_to_mat_pose(aikit.array(ptd.axis_angle_pose)),
        ptd.matrix_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.axis_angle_pose_to_mat_pose(aikit.array(ptd.batched_axis_angle_pose))[0],
        ptd.matrix_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_quaternion_pose_to_mat_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.quaternion_pose_to_mat_pose(aikit.array(ptd.quaternion_pose)),
        ptd.matrix_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.quaternion_pose_to_mat_pose(aikit.array(ptd.batched_quaternion_pose))[0],
        ptd.matrix_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_euler_pose_to_mat_pose(device, fw):
    with aikit_np.use:
        matrix_pose = aikit_mech.euler_pose_to_mat_pose(ptd.euler_pose)
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.euler_pose_to_mat_pose(aikit.array(ptd.euler_pose)),
        matrix_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.euler_pose_to_mat_pose(aikit.array(ptd.batched_euler_pose))[0],
        matrix_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_rot_vec_pose_to_mat_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.rot_vec_pose_to_mat_pose(aikit.array(ptd.rot_vec_pose)),
        ptd.matrix_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.rot_vec_pose_to_mat_pose(aikit.array(ptd.batched_rot_vec_pose))[0],
        ptd.matrix_pose,
        atol=1e-6,
    )
    aikit.previous_backend()
