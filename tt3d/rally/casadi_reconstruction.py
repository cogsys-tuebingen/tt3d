"""
Table tennis reconstuction using the CASADI
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import casadi as ca
from casadi_dae import *
import cv2


def casadi_projection(points_3d, rvec, tvec, K):
    """
    CasADi implementation of 3D to 2D projection using a rotation vector, translation vector, and camera matrix.

    Parameters:
        points_3d (ca.MX): Symbolic (n,3) array of 3D points in world coordinates.
        rvec (ca.MX): 3x1 rotation vector in Rodrigues form.
        tvec (ca.MX): 3x1 translation vector.
        camera_matrix (ca.MX): 3x3 intrinsic camera matrix.

    Returns:
        ca.MX: Symbolic (n,2) array of projected 2D points.
    """

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # transformed_points = rotation_matrix @ points_3d.T + tvec.reshape(
    #     -1, 1
    # )  # Shape (3, n)
    # print(transformed_points)
    # time.sleep(1)

    P = K @ np.hstack([rotation_matrix, tvec.reshape((3, 1))])
    # Convert to homogeneous coordinates
    # print(points_3d.shape)
    homogeneous_points = ca.vertcat(
        points_3d.T, ca.DM.ones(1, points_3d.shape[0])
    )  # Shape (4, n)
    # print(homogeneous_points.shape)

    # Project using the camera matrix
    projected_points = P @ homogeneous_points

    # Convert to 2D by dividing by z-coordinate
    projected_points_2d = projected_points[:2, :] / ca.repmat(
        projected_points[2, :], 2, 1
    )

    return projected_points_2d.T  # Shape (n, 2)


# Assuming pts_2d_before, proj_points_bef, pts_2d_after, and proj_points_aft are CasADi MX symbolic variables
def reprojection_error(pts_2d, proj_points):
    reproj_error = 0
    for i in range(len(pts_2d)):
        # Compute the squared differences for each point
        diff = pts_2d[i, :] - proj_points[i, :].T

        # Compute the norms (Euclidean distance)
        norm = ca.norm_2(diff)  # CasADi's norm_2 for Euclidean distance
        reproj_error += norm

    return reproj_error / len(pts_2d)


def solve_trajectory(
    p_bounce, pts_2d, t, K, rvec, tvec, spin=False, init_params=None, verbose=False
):
    v0, w0, traj = rebuild_diff(t, p_bounce)
    proj_traj = casadi_projection(traj, rvec, tvec, K)

    reproj_error = reprojection_error(pts_2d, proj_traj)

    # J = reproj_bef
    prob = {"f": reproj_error, "x": ca.vertcat(v0, w0)}
    if verbose:
        opts = {"ipopt": {"max_iter": 15}}
    else:
        opts = {"ipopt": {"max_iter": 15, "print_level": 0, "sb": "yes"}}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)
    # solver.print_options()
    lbx = [-4, -20, -8, -900, -900, -900]
    ubx = [4, 20, -0.5, 900, 900, 900]
    # v0 = [0.55, -6.36, -2.9]
    # w0 = [280, 170, -297]
    # sol = solver(x0=[0.55, -6.35, -2.9, 280, 170, -297], lbx=lbx, ubx=ubx)
    if not init_params is None:
        x0 = init_params
    else:
        init_params = [0, -5, -1, 0, 0, 0]
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)
    v_sol = sol["x"][:3]
    w_sol = sol["x"][3:]
    # print(sol)
    # print(v_sol)
    # print(w_sol)
    error = sol["f"]

    # Check sanity
    # traj = rebuild(t, p_bounce, v_sol, w_sol)
    # plt.plot(t, traj)
    # plt.show()

    return v_sol, w_sol, error


def solve_serve(
    p_bounce_1,
    p_bounce_2,
    pts_2d,
    t,
    K,
    rvec,
    tvec,
    spin=False,
    init_params=None,
    verbose=False,
):
    v0, w0, traj = rebuild_diff(t, p_bounce_2)
    proj_traj = casadi_projection(traj, rvec, tvec, K)

    reproj_error = reprojection_error(pts_2d, proj_traj)

    # J = reproj_bef
    prob = {"f": reproj_error, "x": ca.vertcat(v0, w0)}
    if verbose:
        opts = {"ipopt": {"max_iter": 15}}
    else:
        opts = {"ipopt": {"max_iter": 15, "print_level": 0, "sb": "yes"}}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)
    # solver.print_options()
    lbx = [-4, -20, -8, -900, -900, -900]
    ubx = [4, 20, -0.5, 900, 900, 900]
    # v0 = [0.55, -6.36, -2.9]
    # w0 = [280, 170, -297]
    # sol = solver(x0=[0.55, -6.35, -2.9, 280, 170, -297], lbx=lbx, ubx=ubx)
    if not init_params is None:
        x0 = init_params
    else:
        init_params = [0, -5, -1, 0, 0, 0]
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)
    v_sol = sol["x"][:3]
    w_sol = sol["x"][3:]
    # print(sol)
    # print(v_sol)
    # print(w_sol)
    error = sol["f"]

    # Check sanity
    # traj = rebuild(t, p_bounce, v_sol, w_sol)
    # plt.plot(t, traj)
    # plt.show()

    return v_sol, w_sol, error
