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
    """Projects 3D points into 2D using camera intrinsics and extrinsics (symbolically with CasADi)."""
    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec.reshape((3, 1))])
    pts_hom = ca.vertcat(points_3d.T, ca.DM.ones(1, points_3d.shape[0]))
    proj = P @ pts_hom
    return (proj[:2, :] / ca.repmat(proj[2, :], 2, 1)).T


# Assuming pts_2d_before, proj_points_bef, pts_2d_after, and proj_points_aft are CasADi MX symbolic variables
def reprojection_error(pts_2d, proj_points):
    diff = pts_2d - proj_points
    return ca.sum1(ca.sum2(diff**2)) / pts_2d.shape[0]


def make_solver(obj_fun, var, verbose=False):
    opts = {
        "ipopt": {
            "max_iter": 30,
            # "tol": 1e-6,
            # "constr_viol_tol": 1e-6,
            "print_level": 0 if not verbose else 5,
            "sb": "yes",  # Small banner - Supresses casadi small presentation
        },
        "print_time": False,
    }
    problem = {"f": obj_fun, "x": var}
    return ca.nlpsol("solver", "ipopt", problem, opts)


def solve_trajectory(
    p_bounce,
    pts_2d,
    t,
    K,
    rvec,
    tvec,
    spin=False,
    init_params=None,
    verbose=False,
    reg_spin=False,
):
    v0, w0, traj = rebuild_diff(t, p_bounce)
    proj_traj = casadi_projection(traj, rvec, tvec, K)

    # Spin regularization term
    reg_term = 1e-3 * ca.sumsqr(w0)
    obj = reprojection_error(pts_2d, proj_traj)
    if reg_spin:
        obj += reg_term
    x = ca.vertcat(v0, w0)

    solver = make_solver(obj, x, verbose)

    # solver.print_options()
    lbx = [-5, -20, -10, -900, -900, -900]
    ubx = [5, 20, -0.5, 900, 900, 900]
    if not init_params is None:
        x0 = init_params
    else:
        init_params = [0, -5, -1, 0, 0, 0]
    sol = solver(x0=x0, lbx=lbx, ubx=ubx)
    v_sol = sol["x"][:3]
    w_sol = sol["x"][3:]
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

    prob = {"f": reproj_error, "x": ca.vertcat(v0, w0)}
    if verbose:
        opts = {"ipopt": {"max_iter": 15}}
    else:
        opts = {"ipopt": {"max_iter": 15, "print_level": 0, "sb": "yes"}}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)
    # solver.print_options()
    lbx = [-4, -20, -8, -900, -900, -900]
    ubx = [4, 20, -0.5, 900, 900, 900]
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
