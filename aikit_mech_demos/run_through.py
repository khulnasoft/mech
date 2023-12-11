# global
import aikit
import argparse
import aikit_mech


def main(fw=None):
    # Framework Setup #
    # ----------------#

    # choose random framework

    fw = aikit.choose_random_backend() if fw is None else fw
    aikit.set_backend(fw)

    # Orientation #
    # ------------#

    # rotation representations

    # 3
    rot_vec = aikit.array([0.0, 1.0, 0.0])

    # 3 x 3
    rot_mat = aikit_mech.rot_vec_to_rot_mat(rot_vec)

    # 3
    euler_angles = aikit_mech.rot_mat_to_euler(rot_mat, "zyx")

    # 4
    quat = aikit_mech.euler_to_quaternion(euler_angles)

    # 4
    aikit_mech.quaternion_to_axis_angle(quat)

    # Pose #
    # -----#

    # pose representations

    # 3
    position = aikit.ones_like(rot_vec)

    # 6
    rot_vec_pose = aikit.concat([position, rot_vec], axis=0)

    # 3 x 4
    mat_pose = aikit_mech.rot_vec_pose_to_mat_pose(rot_vec_pose)

    # 6
    euler_pose = aikit_mech.mat_pose_to_euler_pose(mat_pose)

    # 7
    quat_pose = aikit_mech.euler_pose_to_quaternion_pose(euler_pose)

    # 6
    aikit_mech.quaternion_pose_to_rot_vec_pose(quat_pose)

    # Position #
    # ---------#

    # conversions of positional representation

    # 3
    cartesian_coord = aikit.random_uniform(low=0.0, high=1.0, shape=(3,))

    # 3
    polar_coord = aikit_mech.cartesian_to_polar_coords(cartesian_coord)

    # 3
    aikit_mech.polar_to_cartesian_coords(polar_coord)

    # cartesian co-ordinate frame-of-reference transformations

    # 3 x 4
    trans_mat = aikit.random_uniform(low=0.0, high=1.0, shape=(3, 4))

    # 4
    aikit_mech.make_coordinates_homogeneous(cartesian_coord)

    # 4 x 4
    aikit_mech.make_transformation_homogeneous(trans_mat)

    # message
    print("End of Run Through Demo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="which backend to use. Chooses a random backend if unspecified.",
    )
    parsed_args = parser.parse_args()
    fw = parsed_args.backend()
    main(fw)
