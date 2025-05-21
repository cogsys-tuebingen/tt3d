"""
Projects the estimated 3D pose on the video
"""

import argparse
from pathlib import Path
from tt3d.calibration.utils import read_camera_info, get_K
from tt3d.rally.geometry import get_transform
from tt3d.pose.data import read_3d_pose, read_2d_pose, filter_by_idx
from tt3d.pose.plots import (
    get_img_from_fig,
    plot_3d_povs,
    draw_skeleton_on_frame,
    draw_table_on_frame,
)
from scipy.optimize import minimize
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from tt3d.pose.align import find_opt_T, project_pts

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

    # Convert the camera frame pose to world frame orientation (No translation)
    p1_pose_3d = p1_pose_3d @ T_table_inv[:3, :3].T
    p2_pose_3d = p2_pose_3d @ T_table_inv[:3, :3].T

    all_pose_2d = read_2d_pose(pose_2d_path)

    # Extract the corresponding player
    p1_pose_2d = filter_by_idx(all_pose_2d, p1)
    p2_pose_2d = filter_by_idx(all_pose_2d, p2)

    # Flatten points to get global reprojection error
    pt_2d = p1_pose_2d.reshape(-1, 2)
    pt_3d = p1_pose_3d.reshape(-1, 3)
    T1, s1 = find_opt_T(
        pt_2d, pt_3d, K, T_table, init_guess=np.array([0, 0, 0, 0, 0, 0, 1.1])
    )
    pt_2d = p2_pose_2d.reshape(-1, 2)
    pt_3d = p2_pose_3d.reshape(-1, 3)
    T2, s2 = find_opt_T(
        pt_2d, pt_3d, K, T_table, init_guess=np.array([0, 0, 0, 0, 0, 0, 1.1])
    )

    # Apply optimized transformation

    videowriter = imageio.get_writer(root_dir / "3d_render.mp4", fps=30)
    cap = cv2.VideoCapture(str(video_path))
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    for i in tqdm(range(len(p1_pose_2d))):
        pose_3d_p1 = p1_pose_3d[i]
        pose_3d_p1 = pose_3d_p1 + T1
        # pose_3d_p2 = scale_points(p2_pose_3d[i], np.zeros(3), s2)
        pose_3d_p2 = p2_pose_3d[i] + T2
        pose_3d_p2 = pose_3d_p2
        p1_2d = project_pts(pose_3d_p1 @ T_table[:3, :3].T + T_table[:3, 3], K)
        p2_2d = project_pts(pose_3d_p2 @ T_table[:3, :3].T + T_table[:3, 3], K)
        ret, frame = cap.read()
        frame = draw_table_on_frame(frame, T_table, K)
        frame = draw_skeleton_on_frame(frame, p2_2d)
        frame = draw_skeleton_on_frame(frame, p2_pose_2d[i], gt=True)
        frame = draw_skeleton_on_frame(frame, p1_2d)
        frame = draw_skeleton_on_frame(frame, p1_pose_2d[i], gt=True)

        videowriter.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.tight_layout()
        # plt.close("all")
    videowriter.close()
