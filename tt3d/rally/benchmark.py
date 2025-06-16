"""
Script to evaluate the accuracy of the reconstruction
"""
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
import math
import cv2
import pandas as pd
import numpy as np
import yaml
from scipy.optimize import curve_fit
from tqdm import tqdm
import csv
import glob
from scipy.interpolate import interp1d
import sys
from tt3d.calibration.utils import read_camera_info
from tt3d.traj_seg.segmenter import (
    basic_segmenter,
    get_accurate_bouncing_pose,
    polynomial_model,
)
from tt3d.traj_seg.utils import wrap_angles
from tt3d.rally.geometry import get_transform, intersect_ray_plane
from tt3d.rally.casadi_reconstruction import solve_trajectory
from tt3d.rally.casadi_dae import rebuild


if __name__ == "__main__":
    np.random.seed(0)
    # Resolve the path relative to the script location
    script_dir = Path(__file__).resolve().parent

    csv_dir = script_dir / "../../data/evaluation/side_no_noise/"
    cam_cal_path = script_dir / "../../data/evaluation/side.yaml"
    rvec, tvec, f, h, w = read_camera_info(cam_cal_path)

    K = np.array(
        [
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1],
        ]
    )
    list_csv = [str(x) for x in csv_dir.glob("*.csv")]
    list_csv = sorted(list_csv)[10:]
    errors = []
    proj_errors = []
    failures = 0
    for k, csv_path in enumerate(list_csv[:]):
        print(csv_path)
        data = pd.read_csv(csv_path)
        t = data["Timestamp"].to_numpy()
        traj = np.vstack(
            [
                data["u"],
                data["v"],
                [1] * len(data["u"]),
                data["l"],
                wrap_angles(np.degrees(data["theta"])),
            ]
        ).T

        qs = basic_segmenter(t, traj, use_blur=True, L=200)
        if len(qs) < 2:
            # fig, axs = plt.subplots(4)
            # for x in qs:
            #     axs[0].axvline(x=t[x], c="r", zorder=0)
            #     axs[1].axvline(x=t[x], c="r", zorder=0)
            # # axs[0].scatter(t_bounce, x_bounce, marker="x")
            # # axs[1].scatter(t_bounce, y_bounce, marker="x")
            # axs[0].scatter(t, traj[:, 0])
            # axs[1].scatter(t, traj[:, 1])
            # axs[2].scatter(t, traj[:, 3])
            # axs[2].axhline(y=3, c="black", linestyle="--")
            # axs[3].scatter(t, traj[:, 4])
            # plt.show()
            failures += 1
            continue
        q = qs[1]
        t_before = t[:q]
        t_after = t[q:]
        t_bounce, (x_bounce, y_bounce) = get_accurate_bouncing_pose(
            t_before,
            traj[:q],
            t_after,
            traj[q:],
        )
        if t_bounce == -1:
            # fig, axs = plt.subplots(2)
            # for x in qs:
            #     axs[0].axvline(x=t[x], c="r", zorder=0)
            #     axs[1].axvline(x=t[x], c="r", zorder=0)
            # # axs[0].scatter(t_bounce, x_bounce, marker="x")
            # # axs[1].scatter(t_bounce, y_bounce, marker="x")
            # axs[0].scatter(t, traj[:, 0])
            # axs[1].scatter(t, traj[:, 1])
            # plt.show()
            failures += 1
            continue

        #     # print(t_bounce)
        # fig, axs = plt.subplots(4, figsize=(10, 8))
        # for x in qs:
        #     axs[0].axvline(x=t[x], c="r", zorder=0)
        #     axs[1].axvline(x=t[x], c="r", zorder=0)
        # axs[0].scatter(t_bounce, x_bounce, marker="x")
        # axs[1].scatter(t_bounce, y_bounce, marker="x")
        # axs[0].scatter(t, traj[:, 0])
        # axs[1].scatter(t, traj[:, 1])
        # axs[2].scatter(t, traj[:, 3])
        # axs[2].axhline(y=3, c="black", linestyle="--")
        # axs[3].scatter(t, traj[:, 4])
        # # plot polynomes before bounce
        # # print(t)
        # t_temp = np.linspace(0, t_bounce, 10)
        # # for the x coord
        # popt_x, _, infodict_x, *_ = curve_fit(
        #     polynomial_model,
        #     t[0:q],
        #     traj[0:q, 0],
        #     full_output=True,
        # )
        # axs[0].plot(
        #     t_temp,
        #     polynomial_model(t_temp, *popt_x),
        # )
        # popt_y, _, infodict_y, *_ = curve_fit(
        #     polynomial_model,
        #     t[0:q],
        #     traj[0:q, 1],
        #     full_output=True,
        # )
        # axs[1].plot(
        #     t_temp,
        #     polynomial_model(t_temp, *popt_y),
        # )
        # # plot polynome after bounce
        # t_temp = np.linspace(t_bounce, t[-1], 10)
        # popt_x, _, infodict_x, *_ = curve_fit(
        #     polynomial_model,
        #     t[q:],
        #     traj[q:, 0],
        #     full_output=True,
        # )
        # axs[0].plot(
        #     t_temp,
        #     polynomial_model(t_temp, *popt_x),
        # )
        # popt_y, _, infodict_y, *_ = curve_fit(
        #     polynomial_model,
        #     t[q:],
        #     traj[q:, 1],
        #     full_output=True,
        # )
        # axs[1].plot(
        #     t_temp,
        #     polynomial_model(t_temp, *popt_y),
        # )
        # plt.show()
        # x_der = np.polyval(np.polyder(popt_x[::-1]), t[q_sol[i] : q_sol[i + 1]])
        # y_der = np.polyval(np.polyder(popt_y[::-1]), t[q_sol[i] : q_sol[i + 1]])
        # pred_angle = np.degrees(np.arctan2(y_der, x_der))
        # pred_angle = wrap_angles(pred_angle)
        # cost_blur = np.abs(pred_angle - traj[q_sol[i] : q_sol[i + 1], 4])
        # cost_blur = np.abs(wrap_angles(cost_blur))
        # axs[4].scatter(t[q_sol[i] : q_sol[i + 1]], cost_blur)
        # axs[3].plot(t[q_sol[i] : q_sol[i + 1]], pred_angle)

        # # print(before_bounce)
        t = np.concatenate([t_before, t_after]) - t_bounce

        # Cut the after observations
        # traj_cut = traj[: q + 2]
        # ls_cut = traj[: q + 2, 3]
        # thetas_cut = traj[: q + 2, 4]

        # t = np.append(t, 0)
        # before_bounce
        # print("t", t)
        # print("pts 2d:", pts_2d)
        bounce_point_cam = intersect_ray_plane(
            rvec, tvec, (x_bounce, y_bounce), K, offset=np.array([0, 0, 0.02])
        )
        T_table = get_transform(rvec, tvec)
        T_table_inv = np.linalg.inv(T_table)
        bounce_point_table = T_table_inv[:3, :3] @ bounce_point_cam + T_table_inv[:3, 3]
        # bounce_point_table[2] = 0.02
        # print(bounce_point_table)
        if np.mean(np.diff(traj[:, 0])) > 0:
            vy_ini = -5
        else:
            vy_ini = 5
        valid_idx = np.arange(len(t))
        pred_v_bounce, pred_w, err = solve_trajectory(
            bounce_point_table,
            traj[:, :2],
            t,
            K,
            rvec,
            tvec,
            spin=True,
            init_params=np.array([0, vy_ini, -1, 0, 0, 0]),
            verbose=False,
        )
        print(err)
        # print("Running exact ode")
        # init_params = np.concatenate([pred_v_bounce, pred_w])
        # pred_v_bounce, pred_w, err = solve_trajectory_ode(
        #     bounce_point_table,
        #     traj[::],
        #     t[::],
        #     K,
        #     rvec,
        #     tvec,
        #     spin=True,
        #     # init_params=np.array([0, vy_ini, -1, 0, 0, 0]),
        #     init_params=init_params,
        # )
        print(pred_v_bounce)
        print(pred_w)
        # print(err)
        # if err > 130:
        #     failures += 1
        #     continue
        # pred_v_bounce, pred_w, err = solve_trajectory_blur(
        #     bounce_point_table,
        #     traj[::],
        #     ls[::],
        #     thetas[::],
        #     t[::],
        #     K,
        #     rvec,
        #     tvec,
        #     spin=True,
        #     init_params=np.array([0.01, 0, vy_ini, -1, 0, 0, 0]),
        # )
        # pred_v_bounce, pred_w, err = solve_trajectory_blur(
        #     bounce_point_table,
        #     traj_cut,
        #     ls_cut,
        #     thetas_cut,
        #     t_cut,
        #     K,
        #     rvec,
        #     tvec,
        #     spin=True,
        #     init_params=np.array([0.01, 0, vy_ini, -1, 0, 0, 0]),
        # )
        # if err > 300:
        #     failures += 1
        #     continue
        proj_errors.append(err)
        # print(pred_v_bounce)
        # print(err)
        # x0 = np.hstack((bounce_point_table, pred_v_bounce))
        # print("x0: ", x0)
        # print("W", pred_w)

        pred_p = rebuild(t, bounce_point_table, pred_v_bounce, pred_w).toarray()
        # print(pred_p)
        traj_3d = np.vstack([data["X"], data["Y"], data["Z"]]).T
        # print(traj_3d)
        # print(pred_p)
        if len(pred_p) != len(traj_3d):
            failures += 1
            continue
        if np.mean(np.linalg.norm(pred_p - traj_3d, axis=1)) > 1:
            failures += 1
            continue
        print("MAE 3D", np.mean(np.linalg.norm(pred_p - traj_3d, axis=1)))
        errors.append(np.mean(np.linalg.norm(pred_p - traj_3d, axis=1)))
        print(np.mean(np.array(errors)))
        print()
        # ax = plt.figure().add_subplot(projection="3d")
        # ax.scatter(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], zorder=1)
        # ax.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], zorder=1)
        # draw_table_tennis_table(ax)
        # set_axes_equal(ax)
        # plt.show()
    print()
    print("MAE 3D", np.mean(np.array(errors)))
    print("STD MAE 3D", np.std(np.array(errors)))
    print("Reproj", np.mean(np.array(proj_errors)))
    print("Reproj std", np.std(np.array(proj_errors)))
    print("Failure rate", failures / len(list_csv))
