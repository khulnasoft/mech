"""Collection of Position Co-ordinate Conversion Functions"""
# global
import aikit

MIN_DENOMINATOR = 1e-12


def polar_to_cartesian_coords(polar_coords):
    r"""Convert spherical polar co-ordinates
    :math:`\mathbf{x}_p = [r, α, β]` to cartesian co-ordinates
    :math:`\mathbf{x}_c = [x, y, z]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_ # noqa

    Parameters
    ----------
    polar_coords
        Spherical polar co-ordinates *[batch_shape,3]*

    Returns
    -------
    ret
        Cartesian co-ordinates *[batch_shape,3]*

    """
    # BS x 1
    phi = polar_coords[..., 0:1]
    theta = polar_coords[..., 1:2]
    r = polar_coords[..., 2:3]

    x = r * aikit.sin(theta) * aikit.cos(phi)
    y = r * aikit.sin(theta) * aikit.sin(phi)
    z = r * aikit.cos(theta)

    # BS x 3
    return aikit.concat([x, y, z], axis=-1)


def cartesian_to_polar_coords(cartesian_coords):
    r"""Convert cartesian co-ordinates
    :math:`\mathbf{x}_c = [x, y, z]` to spherical polar co-ordinates
    :math:`\mathbf{x}_p = [r, α, β]`.\n
    `[reference] <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>`_ # noqa

    Parameters
    ----------
    cartesian_coords
        Cartesian co-ordinates *[batch_shape,3]*

    Returns
    -------
    ret
        Spherical polar co-ordinates *[batch_shape,3]*

    """
    # BS x 1
    x = cartesian_coords[..., 0:1]
    y = cartesian_coords[..., 1:2]
    z = cartesian_coords[..., 2:3]

    r = (x**2 + y**2 + z**2) ** 0.5
    phi = aikit.atan2(y, x)
    theta = aikit.acos(z / (r + MIN_DENOMINATOR))

    # BS x 3
    return aikit.concat([phi, theta, r], axis=-1)
