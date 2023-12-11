"""Collection of tests for co-ordinate conversion functions"""
# global
import aikit
import aikit_mech
import numpy as np

# local
from aikit_mech_tests.test_position.position_data import PositionTestData

ptd = PositionTestData()


def test_polar_to_cartesian_coords(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.polar_to_cartesian_coords(aikit.array(ptd.polar_coords)),
        ptd.cartesian_coords,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.polar_to_cartesian_coords(aikit.array(ptd.batched_polar_coords))[0],
        ptd.cartesian_coords,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_cartesian_to_polar_coords(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.cartesian_to_polar_coords(aikit.array(ptd.cartesian_coords)),
        ptd.polar_coords,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.cartesian_to_polar_coords(aikit.array(ptd.batched_cartesian_coords))[0],
        ptd.polar_coords,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_cartesian_to_polar_coords_and_back(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.polar_to_cartesian_coords(
            aikit_mech.cartesian_to_polar_coords(aikit.array(ptd.cartesian_coords))
        ),
        ptd.cartesian_coords,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.polar_to_cartesian_coords(
            aikit_mech.cartesian_to_polar_coords(aikit.array(ptd.batched_cartesian_coords))
        )[0],
        ptd.cartesian_coords,
        atol=1e-6,
    )
    aikit.previous_backend()


def test_polar_to_cartesian_coords_and_back(device, fw):
    aikit.set_backend(fw)
    assert np.allclose(
        aikit_mech.cartesian_to_polar_coords(
            aikit_mech.polar_to_cartesian_coords(aikit.array(ptd.polar_coords))
        ),
        ptd.polar_coords,
        atol=1e-6,
    )
    assert np.allclose(
        aikit_mech.cartesian_to_polar_coords(
            aikit_mech.polar_to_cartesian_coords(aikit.array(ptd.batched_polar_coords))
        )[0],
        ptd.polar_coords,
        atol=1e-6,
    )
    aikit.previous_backend()
