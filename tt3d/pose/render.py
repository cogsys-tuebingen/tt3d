"""
Fuse all the calculated information into a nice visualization
"""
import argparse
import pandas as pd
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


def resize_and_concat(img1, img2, height=720):
    """
    Resizes two images to the same height and concatenates them horizontally.

    Parameters:
    - img1 (numpy array): First image.
    - img2 (numpy array): Second image.
    - height (int): Desired height for resizing (default: 500).

    Returns:
    - concatenated_img (numpy array): The concatenated image.
    """
    # Compute aspect ratios and new widths while keeping the aspect ratio
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    new_w1 = int((w1 / h1) * height)
    new_w2 = int((w2 / h2) * height)

    # Resize images
    img1_resized = cv2.resize(img1, (new_w1, height))
    img2_resized = cv2.resize(img2, (new_w2, height))

    # Concatenate images horizontally
    concatenated_img = np.hstack((img1_resized, img2_resized))

    return concatenated_img


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
    p1_pose_3d_path = root_dir / "p0_3d.npy"
    p2_pose_3d_path = root_dir / "p1_3d.npy"

    #  Reading the 3d pose data from motionbert
    p1_pose_3d = read_3d_pose(p1_pose_3d_path)
    p2_pose_3d = read_3d_pose(p2_pose_3d_path)
    ball_3d = pd.read_csv(root_dir / "ball_traj_3D.csv")
    videowriter = imageio.get_writer(root_dir / "3d_render.mp4", fps=25)
    cap = cv2.VideoCapture(str(video_path))
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    for i in tqdm(range(len(p1_pose_3d))):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_3d_p1 = p1_pose_3d[i]
        pose_3d_p2 = p2_pose_3d[i]
        if i in ball_3d["idx"].values:
            ball_pos = ball_3d.loc[ball_3d["idx"] == i, ["x", "y", "z"]].iloc[0]
            fig = plot_3d_povs(pose_3d_p1, pose_3d_p2, ball_pos.values)
        else:
            fig = plot_3d_povs(pose_3d_p1, pose_3d_p2)
        fig_img = get_img_from_fig(fig)
        cv2.imwrite(f"pov_{i:04d}.png", cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"frame_{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        f_img = resize_and_concat(fig_img, frame)
        videowriter.append_data(f_img)
        plt.close("all")
    videowriter.close()
