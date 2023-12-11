"""Collection of tests for euler pose functions"""
# global
import aikit
import aikit_mech
import aikit.functional.backends.numpy as aikit_np
import numpy as np

# local
from aikit_mech_tests.test_pose.pose_data import PoseTestData

ptd = PoseTestData()


def test_mat_pose_to_euler_pose(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            euler_pose = aikit_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        aikit.set_backend(fw)
        assert np.allclose(
            aikit_mech.mat_pose_to_euler_pose(aikit.array(ptd.matrix_pose), conv),
            euler_pose,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.mat_pose_to_euler_pose(aikit.array(ptd.batched_matrix_pose), conv)[
                0
            ],
            euler_pose,
            atol=1e-6,
        )
        aikit.previous_backend()


def test_quaternion_pose_to_euler_pose(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            euler_pose = aikit_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        aikit.set_backend(fw)
        assert np.allclose(
            aikit_mech.quaternion_pose_to_euler_pose(
                aikit.array(ptd.quaternion_pose), conv
            ),
            euler_pose,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.quaternion_pose_to_euler_pose(
                aikit.array(ptd.batched_quaternion_pose), conv
            )[0],
            euler_pose,
            atol=1e-6,
        )
        aikit.previous_backend()


def test_axis_angle_pose_to_euler_pose(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            euler_pose = aikit_mech.mat_pose_to_euler_pose(ptd.matrix_pose, conv)
        aikit.set_backend(fw)
        assert np.allclose(
            aikit_mech.axis_angle_pose_to_euler_pose(
                aikit.array(ptd.axis_angle_pose), conv
            ),
            euler_pose,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.axis_angle_pose_to_euler_pose(
                aikit.array(ptd.batched_axis_angle_pose), conv
            )[0],
            euler_pose,
            atol=1e-6,
        )
        aikit.previous_backend()
