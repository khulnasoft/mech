"""Collection of tests for quaternion functions"""
# global
import aikit
import aikit_mech
import aikit.functional.backends.numpy as aikit_np
import numpy as np

# local
from aikit_mech_tests.test_orientation.quaternion_data import QuaternionTestData

qtd = QuaternionTestData()


# Representation Conversions #
# ---------------------------#


def test_vector_and_angle_to_quaternion(device, fw):
    assert np.allclose(
        aikit_mech.axis_angle_to_quaternion(aikit.array(qtd.axis_angle)),
        qtd.quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.axis_angle_to_quaternion(aikit.array(qtd.batched_axis_angle))[0],
        qtd.quaternion,
        atol=1e-6,
    )


def test_axis_angle_to_quaternion(device, fw):
    assert np.allclose(
        aikit_mech.polar_axis_angle_to_quaternion(aikit.array(qtd.polar_axis_angle)),
        qtd.quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.polar_axis_angle_to_quaternion(
            aikit.array(qtd.batched_polar_axis_angle)
        )[0],
        qtd.quaternion,
        atol=1e-6,
    )


def test_rot_mat_to_quaternion(device, fw):
    assert np.allclose(
        aikit_mech.rot_mat_to_quaternion(aikit.array(qtd.rotation_matrix)),
        qtd.quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.rot_mat_to_quaternion(aikit.array(qtd.batched_rotation_matrix))[0],
        qtd.quaternion,
        atol=1e-6,
    )


def test_euler_to_quaternion(device, fw):
    for conv in aikit_mech.VALID_EULER_CONVENTIONS:
        with aikit_np.use:
            quaternion = aikit_mech.euler_to_quaternion(qtd.euler_angles, conv)
        assert np.allclose(
            aikit_mech.euler_to_quaternion(aikit.array(qtd.euler_angles), conv),
            quaternion,
            atol=1e-6,
        )
        assert np.allclose(
            aikit_mech.euler_to_quaternion(aikit.array(qtd.batched_euler_angles), conv)[0],
            quaternion,
            atol=1e-6,
        )


def test_rotation_vector_to_quaternion(device, fw):
    assert np.allclose(
        aikit_mech.rotation_vector_to_quaternion(aikit.array(qtd.rotation_vector)),
        qtd.quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.rotation_vector_to_quaternion(aikit.array(qtd.batched_rotation_vector))[
            0
        ],
        qtd.quaternion,
        atol=1e-6,
    )


# Quaternion Operations #
# ----------------------#


def test_inverse_quaternion(device, fw):
    assert np.allclose(
        aikit_mech.inverse_quaternion(
            aikit_mech.inverse_quaternion(aikit.array(qtd.quaternion))
        ),
        qtd.quaternion,
        atol=1e-6,
    )


def test_get_random_quaternion(device, fw):
    assert aikit_mech.get_random_quaternion().shape == (4,)
    assert aikit_mech.get_random_quaternion(batch_shape=(1, 1)).shape == (1, 1, 4)


def test_scale_quaternion_theta(device, fw):
    assert np.allclose(
        aikit_mech.scale_quaternion_rotation_angle(
            aikit.array(qtd.quaternion), aikit.array(qtd.scale_factor)
        ),
        qtd.scaled_quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.scale_quaternion_rotation_angle(
            aikit.array(qtd.batched_quaternion), aikit.array(qtd.batched_scale_factor)
        )[0],
        qtd.scaled_quaternion,
        atol=1e-6,
    )


def test_hamilton_product(device, fw):
    assert np.allclose(
        aikit_mech.hamilton_product(aikit.array(qtd.quaternion), aikit.array(qtd.quaternion)),
        qtd.hp_quaternion,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.hamilton_product(
            aikit.array(qtd.batched_quaternion), aikit.array(qtd.batched_quaternion)
        )[0],
        qtd.hp_quaternion,
        atol=1e-6,
    )
