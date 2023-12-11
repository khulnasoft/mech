"""Collection of tests for euler functions"""
# global
import aikit
import aikit_mech
import aikit.functional.backends.numpy as aikit_np
import numpy as np

# local
from aikit_mech_tests.test_orientation.orientation_data import OrientationTestData

otd = OrientationTestData()


def test_rot_mat_to_euler(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            euler_angles = aikit_mech.rot_mat_to_euler(otd.rotation_matrix, conv)
        assert np.allclose(
            aikit_mech.rot_mat_to_euler(aikit.array(otd.rotation_matrix), conv),
            euler_angles,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.rot_mat_to_euler(aikit.array(otd.batched_rotation_matrix), conv)[0],
            euler_angles,
            atol=1e-6,
        )


def test_quaternion_to_euler(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            euler_angles = aikit_mech.quaternion_to_euler(otd.quaternion, conv)
        assert np.allclose(
            aikit_mech.quaternion_to_euler(aikit.array(otd.quaternion), conv),
            euler_angles,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.quaternion_to_euler(aikit.array(otd.batched_quaternion), conv)[0],
            euler_angles,
            atol=1e-6,
        )


def test_axis_angle_to_euler(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            euler_angles = aikit_mech.quaternion_to_euler(otd.quaternion, conv)
        assert np.allclose(
            aikit_mech.axis_angle_to_euler(aikit.array(otd.axis_angle), conv),
            euler_angles,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.axis_angle_to_euler(aikit.array(otd.batched_axis_angle), conv)[0],
            euler_angles,
            atol=1e-6,
        )


def test_get_random_euler(device, fw):
    assert aikit_mech.get_random_euler().shape == (3,)
    assert aikit_mech.get_random_euler(batch_shape=(1, 1)).shape == (1, 1, 3)
