"""
Check the feet position
"""
import cv2
from scipy.interpolate import interp1d
import imageio
import numpy as np
import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from tt3d.calibration.utils import read_camera_info, get_K
from tt3d.rally.geometry import get_transform
from tt3d.pose.data import read_3d_pose, read_2d_pose, filter_by_idx
from tt3d.pose.plots import get_img_from_fig, plot_3d_povs


# H36M, 0: 'root',
#                             1: 'rhip',
#                             2: 'rkne',
#                             3: 'rank',
#                             4: 'lhip',
#                             5: 'lkne',
#                             6: 'lank',
#                             7: 'belly',
#                             8: 'neck',
#                             9: 'nose',
#                             10: 'head',
#                             11: 'lsho',
#                             12: 'lelb',
#                             13: 'lwri',
#                             14: 'rsho',
#                             15: 'relb',
#                             16: 'rwri'
def low_pass_filter(data, alpha):
    """
    Apply a low-pass filter to a 1D array or list using an exponential moving average.

    Parameters:
    - data (array-like): The input data as a list or 1D numpy array.
    - alpha (float): The smoothing factor between 0 and 1.
                     Higher values give more weight to recent values (less smoothing),
                     while lower values give more weight to older values (more smoothing).

    Returns:
    - filtered_data (numpy array): The low-pass filtered data.
    """
    data = np.asarray(data)  # Ensure input is a numpy array
    if not (0 < alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    # Initialize the output array
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]  # Set the first value as the initial condition

    # Apply the low-pass filter
    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]

    return filtered_data


def get_low_vel_acc_indices(pos, dt, threshold):
    vel = np.diff(pos) / dt
    acc = np.diff(vel) / dt

    # Find indices where absolute velocity is below the threshold
    low_velocity_indices = np.where(np.abs(vel) < threshold)[0]
    low_acc_indices = np.where(np.abs(acc) < threshold / dt)[0]
    return np.intersect1d(low_velocity_indices, low_acc_indices)


def get_floor(pos, dt):
    floor = np.zeros_like(pos)
    vel = np.diff(pos) / dt
    idxs = [0]
    idxs = get_low_vel_acc_indices(pos, dt, 0.1)
    low_speed_pos = pos[idxs]
    interp_func = interp1d(
        idxs,
        low_speed_pos,
        kind="linear",
        fill_value="extrapolate",
    )
    floor = interp_func(np.arange(len(pos)))
    return floor


if __name__ == "__main__":
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }

    plt.rcParams.update(tex_fonts)
    root_dir = Path("/home/gossard/Git/tt3d/data/demo_video/satsuki_002")
    calib_path = root_dir / "camera.yaml"
    rvec, tvec, f, h, w = read_camera_info(calib_path)
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    T_table = get_transform(rvec, tvec)
    T_table_inv = np.linalg.inv(T_table)
    p1_pose_3d_path = root_dir / "player_0.npy"
    p2_pose_3d_path = root_dir / "player_1.npy"

    #  Reading the 3d pose data from motionbert
    p1_pose_3d = read_3d_pose(p1_pose_3d_path)
    p2_pose_3d = read_3d_pose(p2_pose_3d_path)
    print(p1_pose_3d.shape)
    p1_pose_3d = p1_pose_3d @ T_table_inv[:3, :3].T
    p2_pose_3d = p2_pose_3d @ T_table_inv[:3, :3].T

    # Choose the player to study
    pose_3d = p2_pose_3d

    n = pose_3d.shape[0]
    t = np.arange(n) / 25

    floor_right = get_floor(pose_3d[:, 3, 2], 1 / 25)
    floor_left = get_floor(pose_3d[:, 6, 2], 1 / 25)
    # Create subplots that share the x-axis
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    # Plot positions
    axs[0].plot(t, pose_3d[:, 3, 2], label="Right Ankle", color="b")
    axs[0].plot(t, pose_3d[:, 6, 2], label="Left Ankle", color="g")
    axs[0].plot(t, floor_right, label="Est. Right Floor", linestyle="--", color="c")
    axs[0].plot(t, floor_left, label="Est. Left Floor", linestyle="--", color="m")
    # axs[0].set_title("Positions of Right and Left Ankles")
    axs[0].set_ylabel("Z [m]")
    axs[0].legend()
    axs[0].grid()

    # Calculate velocity
    velocity_right = np.diff(pose_3d[:, 3, 2]) * 25  # Scale by frame rate
    velocity_left = np.diff(pose_3d[:, 6, 2]) * 25

    # Adjust time array for velocity
    t_velocity = t[:-1]

    # Plot velocity
    axs[1].plot(t_velocity, velocity_right, label="Velocity Right Ankle", color="b")
    axs[1].plot(t_velocity, velocity_left, label="Velocity Left Ankle", color="g")
    axs[1].axhline(y=0.1, color="r", linestyle="-")
    axs[1].axhline(y=-0.1, color="r", linestyle="-")
    # axs[1].set_title("Velocity of Right and Left Ankles")
    axs[1].set_ylabel("V_Z [m/s]")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlabel("Time [s]")

    # Calculate acceleration
    acceleration_right = np.diff(velocity_right) * 25  # Scale by frame rate
    acceleration_left = np.diff(velocity_left) * 25

    # Adjust time array for acceleration
    t_acceleration = t_velocity[:-1]

    # Plot acceleration
    # axs[2].plot(
    #     t_acceleration, acceleration_right, label="Acceleration Right Ankle", color="b"
    # )
    # axs[2].plot(
    #     t_acceleration, acceleration_left, label="Acceleration Left Ankle", color="g"
    # )
    # axs[2].set_title("Acceleration of Right and Left Ankles")
    # axs[2].set_ylabel("Acceleration (Z-axis)")
    # axs[2].legend()
    # axs[2].grid()

    # # Set common x-axis label
    # axs[2].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.show()
