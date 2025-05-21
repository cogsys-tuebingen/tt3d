"""
Script to put the infered 3D pose from MotionBert which are 
in camera frame into the world frame
"""

import argparse
from scipy.interpolate import interp1d
from pathlib import Path
from tt3d.calibration.utils import read_camera_info, get_K
from tt3d.rally.geometry import get_transform
from tt3d.pose.data import read_3d_pose, read_2d_pose, filter_by_idx
from tt3d.pose.plot_feet_position import get_low_vel_acc_indices
from tt3d.pose.plots import get_img_from_fig, plot_3d_povs
from scipy.optimize import minimize
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm


def project_pts(pts, K):
    """
    Project 3D points P to 2D using the intrinsic matrix.

    Args:
        P (np.ndarray): 3D points of shape (N, 3).
        intrinsic_matrix (np.ndarray): Intrinsic matrix of the camera (3x3).

    Returns:
        np.ndarray: Projected 2D points of shape (N, 2).
    """
    P_2d_homogeneous = (K @ pts.T).T
    # Convert back from homogeneous to 2D
    P_2d = P_2d_homogeneous[:, :2] / P_2d_homogeneous[:, 2].reshape(-1, 1)
    return P_2d


def ReLU(x):
    return x * (x > 0)


def reprojection_error(params, P_3d, P_2d, K, T_table):
    """
    Calculate the reprojection error between 3D points and corresponding 2D points.

    Args:
        params (np.ndarray): The parameters for the rotation (R) and translation (t).
        P_3d (np.ndarray): 3D points of shape (N, 3).
        P_2d (np.ndarray): Corresponding 2D points of shape (N, 2).
        intrinsic_matrix (np.ndarray): Intrinsic matrix of the camera (3x3).

    Returns:
        float: The total reprojection error.
    """
    # Extract the rotation and translation from params
    # r_vec = params[:3]  # Rotation vector
    # r_vec = np.array(params[:3])
    t = np.array(params[:3])  # Translation vector
    scale = params[3]

    # Apply the transformation to the 3D points
    # n = P_3d.shape[0]
    # for i in range(int(n / 17)):
    #     P_3d[17 * i : 17 * (i + 1)] = scale_points(
    #         P_3d[17 * i : 17 * (i + 1)], P_3d[17 * i], 1
    #     )
    P_3d_table = scale * (P_3d + t)
    P_3d_camera = (P_3d_table @ T_table[:3, :3].T) + T_table[:3, 3]
    # ax = plot_skeleton(P_3d_table)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()
    # print(P_3d_table)
    # print(P_3d_camera)

    # Project the 3D points to 2D
    # print(P_3d_camera)
    P_2d_projected = project_pts(P_3d_camera, K)

    # Calculate the reprojection error
    error = np.linalg.norm(P_2d - P_2d_projected, axis=1)

    # print(np.mean(error))
    # reg_error = ReLU(-P_3d_table[6, 2] - 0.66) + ReLU(-P_3d_table[3, 2] - 0.66)
    # print(reg_error)
    reg_error = np.sum(np.abs(P_3d_table[6::17, 2] + 0.66)) + np.sum(
        np.abs(P_3d_table[3::17, 2] + 0.66)
    )
    # reg_error = 0
    return np.mean(error) + 10 * reg_error  # Sum of reprojection errors for all points


def optimize_reprojection(P_3d, P_2d, intrinsic_matrix, T_table, initial_params=None):
    """
    Optimize the reprojection error by adjusting the transformation matrix.

    Args:
        P_3d (np.ndarray): 3D points of shape (N, 3).
        P_2d (np.ndarray): Corresponding 2D points of shape (N, 2).
        intrinsic_matrix (np.ndarray): Intrinsic matrix of the camera (3x3).
        initial_params (np.ndarray): Initial guess for rotation and translation (optional).

    Returns:
        np.ndarray: Optimized parameters for rotation and translation.
    """
    if initial_params is None:
        # Default initial parameters: 0 rotation, 0 translation
        # initial_params = np.zeros(7)
        # initial_params[6] = 1  # Scale
        initial_params = np.zeros(4)
        initial_params[3] = 1

    # Get the idx where the ground is touched by the feet
    # floor_idx_right = get_low_vel_acc_indices(P_3d[3::17, 2], 1 / 30, 0.1) * 17 + 3
    # floor_idx_left = get_low_vel_acc_indices(P_3d[6::17, 2], 1 / 30, 0.1) * 17 + 6

    # Minimize the reprojection error
    result = minimize(
        reprojection_error,
        initial_params,
        args=(P_3d, P_2d, intrinsic_matrix, T_table),
        method="BFGS",
    )

    print(result)
    return result.x


def find_opt_T(pt_2d, pt_3d, K, T_table, init_guess=None):
    optimized_params = optimize_reprojection(
        pt_3d, pt_2d, K, T_table, initial_params=init_guess
    )
    tvec = np.array(optimized_params[:3])
    scale = optimized_params[3]
    return tvec, scale


def smooth_tvecs(tvecs, window_size=5):
    """
    Smooths a list of n x 3 translation vectors using a moving average.

    Parameters:
        tvecs (array-like): An n x 3 array of translation vectors.
        window_size (int): The size of the moving average window. Must be an odd positive integer.

    Returns:
        np.ndarray: An n x 3 array of smoothed translation vectors.
    """
    tvecs = np.asarray(tvecs)
    if tvecs.ndim != 2 or tvecs.shape[1] != 3:
        raise ValueError("Input must be an n x 3 array of translation vectors.")

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")

    # Padding mode 'edge' to avoid boundary effects
    pad_size = window_size // 2
    padded_tvecs = np.pad(tvecs, ((pad_size, pad_size), (0, 0)), mode="edge")

    # Apply moving average along each column separately
    smoothed_tvecs = np.array(
        [
            np.convolve(
                padded_tvecs[:, i], np.ones(window_size) / window_size, mode="valid"
            )
            for i in range(3)
        ]
    ).T

    return smoothed_tvecs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align the 3D pose with the world frame"
    )
    parser.add_argument("input_dir", type=str, help="Path to the input dir")
    args = parser.parse_args()

    # Index of the players
    p1 = 0
    p2 = 1
    root_dir = Path(args.input_dir)
    calib_path = root_dir / "camera.yaml"
    video_path = root_dir / "rally.mp4"

    rvec, tvec, f, h, w = read_camera_info(calib_path)
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    T_table = get_transform(rvec, tvec)
    T_table_inv = np.linalg.inv(T_table)
    K = get_K(f, h, w)

    pose_2d_path = root_dir / "mb_input.json"
    p1_pose_3d_path = root_dir / "player_0.npy"
    p2_pose_3d_path = root_dir / "player_1.npy"

    #  Reading the 3d pose data from motionbert
    p1_pose_3d = read_3d_pose(p1_pose_3d_path)
    p2_pose_3d = read_3d_pose(p2_pose_3d_path)

    # print(p1_pose_3d[0])
    # Convert to world frame orientation (No translation)
    p1_pose_3d = p1_pose_3d @ T_table_inv[:3, :3].T
    # print(p1_pose_3d[0])
    p2_pose_3d = p2_pose_3d @ T_table_inv[:3, :3].T

    all_pose_2d = read_2d_pose(pose_2d_path)

    # Extract the corresponding player
    p1_pose_2d = filter_by_idx(all_pose_2d, p1)
    p2_pose_2d = filter_by_idx(all_pose_2d, p2)

    # Flatten points to get global reprojection error
    pt_2d = p1_pose_2d.reshape(-1, 2)
    pt_3d = p1_pose_3d.reshape(-1, 3)
    T1, s1 = find_opt_T(pt_2d, pt_3d, K, T_table)
    pt_2d = p2_pose_2d.reshape(-1, 2)
    pt_3d = p2_pose_3d.reshape(-1, 3)
    T2, s2 = find_opt_T(pt_2d, pt_3d, K, T_table)

    # Calculate when to calculate the transform
    # Player 1
    right_floor_idx = get_low_vel_acc_indices(p1_pose_3d[:, 3, 2], 1 / 25, 0.1)
    left_floor_idx = get_low_vel_acc_indices(p1_pose_3d[:, 6, 2], 1 / 25, 0.1)
    floor_idx = np.unique(np.concatenate([right_floor_idx, left_floor_idx]))
    # floor_idx = np.arange(len(p1_pose_3d))
    # print(right_floor_idx)
    # print(left_floor_idx)
    # print(floor_idx)
    T1 = []
    S1 = []
    for idx in floor_idx:
        t1, s1 = find_opt_T(p1_pose_2d[idx], p1_pose_3d[idx], K, T_table)
        T1.append(t1)
        S1.append(s1)

    T1 = np.array(T1)
    s1 = np.mean(np.array(S1))
    interp_func = interp1d(
        floor_idx,
        T1,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    tvec1 = interp_func(np.arange(len(p1_pose_3d)))
    tvec1 = smooth_tvecs(tvec1)

    # Player 2
    right_floor_idx = get_low_vel_acc_indices(p2_pose_3d[:, 3, 2], 1 / 25, 0.1)
    left_floor_idx = get_low_vel_acc_indices(p2_pose_3d[:, 6, 2], 1 / 25, 0.1)
    floor_idx = np.unique(np.concatenate([right_floor_idx, left_floor_idx]))
    # floor_idx = np.arange(len(p1_pose_3d))
    # print(right_floor_idx)
    # print(left_floor_idx)
    # print(floor_idx)
    T2 = []
    S2 = []
    for idx in floor_idx:
        t2, s2 = find_opt_T(p2_pose_2d[idx], p2_pose_3d[idx], K, T_table)
        T2.append(t2)
        S2.append(s2)

    s2 = np.mean(np.array(S2))
    T2 = np.array(T2)
    interp_func = interp1d(
        floor_idx,
        T2,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    tvec2 = interp_func(np.arange(len(p2_pose_3d)))
    tvec2 = smooth_tvecs(tvec2)
    # print(tvec1)
    # plt.plot(tvec1)
    # plt.show()
    # Apply optimized transformation

    videowriter = imageio.get_writer(root_dir / "aligned_3d_pose.mp4", fps=25)
    cap = cv2.VideoCapture(str(video_path))
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # print(s1)
    # print(s2)
    save_p1_3d = []
    save_p2_3d = []
    for i in tqdm(range(len(p1_pose_2d))):
        pose_3d_p1 = p1_pose_3d[i]
        # T1, s1 = find_opt_T(p1_pose_2d[i], pose_3d_p1, K, T_table)
        pose_3d_p1 = s1 * (pose_3d_p1 + tvec1[i])
        # pose_3d_p1 = s1 * pose_3d_p1 + T1
        save_p1_3d.append(pose_3d_p1)
        # pose_3d_p2 = scale_points(p2_pose_3d[i], np.zeros(3), s2)
        pose_3d_p2 = p2_pose_3d[i]
        # T2, s2 = find_opt_T(p2_pose_2d[i], pose_3d_p2, K, T_table)
        pose_3d_p2 = s2 * (pose_3d_p2 + tvec2[i])
        # pose_3d_p2 = s2 * pose_3d_p2 + T2
        save_p2_3d.append(pose_3d_p2)
        # pose_3d_p2 = pose_3d_p2 @ T_table[:3, :3].T + T_table[:3, 3]
        # pose_3d_p1 = pose_3d_p1 @ T_table[:3, :3].T + T_table[:3, 3]
        fig = plot_3d_povs(pose_3d_p1, pose_3d_p2)
        frame = get_img_from_fig(fig)
        videowriter.append_data(frame)
        plt.close("all")
    videowriter.close()

    save_p1_3d = np.array(save_p1_3d)
    save_p2_3d = np.array(save_p2_3d)
    np.save(str(root_dir / "p0_3d.npy"), save_p1_3d)
    np.save(str(root_dir / "p1_3d.npy"), save_p2_3d)
