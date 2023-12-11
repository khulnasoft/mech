"""Collection of tests for quaternion pose functions"""
# global
import aikit
import aikit_mech
import numpy as np

# local
from aikit_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_axis_angle_pose_to_quaternion_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.axis_angle_pose_to_quaternion_pose(aikit.array(ptd.axis_angle_pose)),
        ptd.quaternion_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.axis_angle_pose_to_quaternion_pose(
            aikit.array(ptd.batched_axis_angle_pose)
        )[0],
        ptd.quaternion_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_mat_pose_to_quaternion_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.mat_pose_to_quaternion_pose(aikit.array(ptd.matrix_pose)),
        ptd.quaternion_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.mat_pose_to_quaternion_pose(aikit.array(ptd.batched_matrix_pose))[0],
        ptd.quaternion_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_euler_pose_to_quaternion_pose(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.euler_pose_to_quaternion_pose(aikit.array(ptd.euler_pose)),
        ptd.quaternion_pose,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.euler_pose_to_quaternion_pose(aikit.array(ptd.batched_euler_pose))[0],
        ptd.quaternion_pose,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_increment_quaternion_pose_with_velocity(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.increment_quaternion_pose_with_velocity(
            aikit.array(ptd.quaternion_pose),
            aikit.array(ptd.velocity),
            aikit.array(ptd.control_dt),
        ),
        ptd.incremented_quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.increment_quaternion_pose_with_velocity(
            aikit.array(ptd.batched_quaternion_pose),
            aikit.array(ptd.batched_velocity),
            aikit.array(ptd.batched_control_dt),
        )[0],
        ptd.incremented_quaternion,
        atol=1e-6,
    )
    aikit.previous_backend()
