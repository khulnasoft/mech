"""Collection of tests for homogeneous co-ordinate functions"""
# global
import aikit
import aikit_mech
import numpy as np

# local
from aikit_mech_tests.test_position.position_data import PositionTestData

ptd = PositionTestData()


def test_make_coordinates_homogeneous(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.make_coordinates_homogeneous(aikit.array(ptd.cartesian_coords)),
        ptd.cartesian_coords_homo,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.make_coordinates_homogeneous(aikit.array(ptd.batched_cartesian_coords)),
        ptd.batched_cartesian_coords_homo,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_make_transformation_homogeneous(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.make_transformation_homogeneous(aikit.array(ptd.ext_mat)),
        ptd.ext_mat_homo,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.make_transformation_homogeneous(aikit.array(ptd.batched_ext_mat)),
        ptd.batched_ext_mat_homo,
        atol=1e-6,
    )
    aikit.previous_backend()
