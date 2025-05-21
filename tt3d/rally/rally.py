"""
Script to reconstruct the whole rally
"""

import argparse
from tt3d.calibration.utils import read_camera_info
from tt3d.traj_seg.utils import read_traj
from pathlib import Path
from scipy.optimize import curve_fit
from tt3d.traj_seg.segmenter import basic_segmenter, classify_q
from tt3d.rally.geometry import intersect_ray_plane, get_transform
from tt3d.traj_seg.segmenter import (
    basic_segmenter,
    get_accurate_bouncing_pose,
    polynomial_model,
)
import matplotlib.pyplot as plt
from tt3d.rally.casadi_reconstruction import solve_trajectory, rebuild
from tt3d.rally.plots import *
import numpy as np
import csv


def save_3d_traj(x, y, filename="output.csv"):
    """
    Saves x and y values to a CSV file.

    Parameters:
    - x (array-like): Array of x values.
    - y (array-like): Corresponding y values.
    - filename (str): Name of the output CSV file.
    """
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "x", "y", "z"])  # Write header
        for x_val, y_val in zip(x, y):
            writer.writerow([x_val, y_val[0], y_val[1], y_val[2]])

    print(f"Saved to {filename}")


def fuse_duplicates(x, y):
    """
    Fuses y values for duplicate x values using a specified fusion function.

    Parameters:
    - x (array-like): Array of x values.
    - y (array-like): Corresponding y values.
    - fusion_fn (function): Function to fuse duplicate y values (default is np.mean).

    Returns:
    - unique_x (numpy array): Array of unique x values.
    - fused_y (numpy array): Corresponding fused y values.
    """
    unique_x = []
    fused_y = []

    x = np.array(x)
    y = np.array(y)

    # print(np.unique(x.round(decimals=3)))
    # print(y)
    for val in np.unique(x):
        indices = np.where(abs(x - val) < 1e-3)[0]
        unique_x.append(val)
        print(y[indices])
        fused_y.append(np.mean(y[indices], axis=0))

    return np.array(unique_x), np.array(fused_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    # Read the trajectory
    csv_path = root_dir / "ball_traj_2D.csv"
    fps = 25  # HACK: Change
    traj = read_traj(csv_path)
    serve_visible = True

    # print(traj)

    t = traj[:, 0] / fps
    idxs = traj[:, 0]
    traj = traj[:, 1:]

    # Filter out the miss detections
    traj[:3, 2] = 0
    # traj[100:, 2] = 0
    t = t[traj[:, 2] != 0]
    traj = traj[traj[:, 2] != 0]

    # print(traj)
    camcal_path = root_dir / "camera.yaml"
    rvec, tvec, f, h, w = read_camera_info(camcal_path)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    q_sol = basic_segmenter(t, traj, use_blur=True, L=200)
    print("q_sol:", q_sol)
    if serve_visible:
        q_serve = q_sol[1]
        q_rally = q_sol[2:]
    else:
        q_rally = q_sol[1:]
    q_racket, q_table = classify_q(q_rally, t, traj)
    # q_table= q_table[1:]
    print("q_racket", q_racket)
    print("q_table", q_table)

    # Plot the velocity
    dt = np.diff(t)
    vx = np.diff(traj[:, 0]) / dt
    vy = np.diff(traj[:, 1]) / dt

    f, axs = plt.subplots(2, figsize=(10, 10), sharex=True)
    for x in q_table:
        axs[0].axvline(x=t[x], c="r", zorder=0)
        axs[1].axvline(x=t[x], c="r", zorder=0)
    for x in q_racket:
        axs[0].axvline(x=t[x], c="b", zorder=0)
        axs[1].axvline(x=t[x], c="b", zorder=0)

    axs[0].scatter(t, traj[:, 0], s=20)
    axs[1].scatter(t, traj[:, 1], s=20)
    # axs[2].scatter(t[:-1], vx)
    # axs[3].scatter(t[:-1], vy)
    axs[0].set_ylabel("u")
    axs[1].set_ylabel("v")
    axs[1].set_xlabel("Time step")
    # plt.show()

    # Plot the fitted polynomes
    ts_exact = [t[0]]
    x_bounces = []
    y_bounces = []
    for i in range(1, len(q_sol) - 1):
        t_bounce, (x_bounce, y_bounce) = get_accurate_bouncing_pose(
            t[q_sol[i - 1] : q_sol[i]],
            traj[q_sol[i - 1] : q_sol[i]],
            t[q_sol[i] : q_sol[i + 1]],
            traj[q_sol[i] : q_sol[i + 1]],
        )
        axs[0].scatter(t_bounce, x_bounce, s=50, marker="x", c="black", zorder=3)
        axs[1].scatter(t_bounce, y_bounce, s=50, marker="x", c="black", zorder=3)
        ts_exact.append(t_bounce)
        x_bounces.append(x_bounce)
        y_bounces.append(y_bounce)

    # For the last one
    t_bounce, (x_bounce, y_bounce) = get_accurate_bouncing_pose(
        t[q_sol[-2] : q_sol[-1]],
        traj[q_sol[-2] : q_sol[-1]],
        t[q_sol[-1] : -1],
        traj[q_sol[-1] : -1],
    )
    x_bounces.append(x_bounce)
    y_bounces.append(y_bounce)
    axs[0].scatter(t_bounce, x_bounce, s=50, marker="x", c="black", zorder=3)
    axs[1].scatter(t_bounce, y_bounce, s=50, marker="x", c="black", zorder=3)
    ts_exact.append(t_bounce)

    ts_exact = np.array(ts_exact).astype(np.float32)
    print("t_exact", ts_exact)

    for i in range(len(q_sol) - 1):
        t_temp = np.linspace(ts_exact[i], ts_exact[i + 1], 10)
        t_rebuild = np.arange(t[q_sol[i]], t[q_sol[i + 1]], 1 / fps)
        # For the x coord
        popt_X, _, infodict_X, *_ = curve_fit(
            polynomial_model,
            t[q_sol[i] : q_sol[i + 1]],
            traj[q_sol[i] : q_sol[i + 1], 0],
            full_output=True,
        )
        axs[0].plot(
            t_temp,
            polynomial_model(t_temp, *popt_X),
        )
        popt_Y, _, infodict_Y, *_ = curve_fit(
            polynomial_model,
            t[q_sol[i] : q_sol[i + 1]],
            traj[q_sol[i] : q_sol[i + 1], 1],
            full_output=True,
        )
        axs[1].plot(
            t_temp,
            polynomial_model(t_temp, *popt_Y),
        )
    # t_temp = np.linspace(ts_exact[-1], t[-1], 10)
    # # For the x coord
    # popt_X, _, infodict_X, *_ = curve_fit(
    #     polynomial_model,
    #     t[q_sol[-1] : -1],
    #     traj[q_sol[-1] : -1, 0],
    #     full_output=True,
    # )
    # axs[0].plot(
    #     t_temp,
    #     polynomial_model(t_temp, *popt_X),
    # )
    # popt_Y, _, infodict_Y, *_ = curve_fit(
    #     polynomial_model,
    #     t[q_sol[-1] : -1],
    #     traj[q_sol[-1] : -1, 1],
    #     full_output=True,
    # )
    # axs[1].plot(
    #     t_temp,
    #     polynomial_model(t_temp, *popt_Y),
    # )

    # for x in q_sol:
    #     axs[0].axvline(x=t[x - 1], c="r")
    #     axs[1].axvline(x=t[x - 1], c="r")
    for i in range(2):
        axs[i].grid(True)

    # q_table = [12]
    # q_racket = [20]
    # before_bounce = traj[: q_table[0], :2]
    # t_before = t[: q_table[0]]
    # after_bounce = traj[q_table[0] : q_racket[0]]
    # t_after = t[q_table[0] : q_racket[0]]

    plt.tight_layout()
    plt.show()
    plt.close("all")

    # Iterate over all the bounce points
    traj = traj[:, :2]
    t_3d = []
    traj_3d = []
    all_idx = []
    all_traj_3d = []
    # HACK: removed last
    for i, q in enumerate(q_sol[:-1]):
        if not q in q_table:
            continue
        print("Processing: ", i, q)
        x_bounce, y_bounce = x_bounces[i - 1], y_bounces[i - 1]
        t_bounce = ts_exact[i]
        t_before = t[q_sol[i - 1] : q_sol[i]]
        traj_before = traj[q_sol[i - 1] : q_sol[i]]
        t_after = t[q_sol[i] : q_sol[i + 1]]
        traj_after = traj[q_sol[i] : q_sol[i + 1]]
        t_cut = np.concatenate([t_before, t_after]) - t_bounce
        traj_cut = np.concatenate((traj_before, traj_after))
        print(t_cut)
        # print(traj_cut)

        bounce_point_cam = intersect_ray_plane(rvec, tvec, (x_bounce, y_bounce), K)
        T_table = get_transform(rvec, tvec)
        T_table_inv = np.linalg.inv(T_table)
        bounce_point_table = T_table_inv[:3, :3] @ bounce_point_cam + T_table_inv[:3, 3]
        # bounce_point_table[2] = 0.02
        # print(bounce_poinit_table)
        if np.mean(np.diff(traj[:, 0])) > 0:
            vy_ini = -5
        else:
            vy_ini = 5
        valid_idx = np.arange(len(t))

        pred_v_bounce, pred_w, err = solve_trajectory(
            bounce_point_table,
            traj_cut,
            t_cut,
            K,
            rvec,
            tvec,
            spin=True,
            init_params=np.array([0, vy_ini, -1, 0, 0, 0]),
        )
        print("V:", pred_v_bounce)
        print("w:", pred_w)
        t_rebuild = np.arange(t[q_sol[i - 1]], t[q_sol[i + 1]], 1 / fps)
        # print(t_rebuild - t_bounce)
        pred_p = rebuild(
            t_rebuild - t_bounce, bounce_point_table, pred_v_bounce, pred_w
        )
        all_idx.append(t_rebuild)
        print(t_rebuild)
        all_traj_3d.append(pred_p)
        ax = plt.figure().add_subplot(projection="3d", computed_zorder=False)
        ax.scatter(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], color="red", zorder=2)
        draw_tt_table(ax)
        set_axes_equal(ax)
        plt.show()
        t_3d.append(t_cut + t_bounce)
        traj_3d.append(pred_p)

    t_3d = np.hstack(t_3d)
    # print(t_3d)
    traj_3d = np.vstack(traj_3d)
    # print(traj_3d)
    # animate_3d_positions(traj_3d, t_3d, fps=25)
    all_idx = np.concatenate(all_idx)
    all_traj_3d = np.concatenate(all_traj_3d)
    print(all_idx)
    all_idx, all_traj_3d = fuse_duplicates(all_idx, all_traj_3d)

    save_3d_traj(
        (all_idx * 25).astype(np.uint8), all_traj_3d, str(root_dir / "ball_traj_3D.csv")
    )
