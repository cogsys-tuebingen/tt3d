"""
Script to segment the 2D ball trajectory

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9618954
https://www.cs.toronto.edu/~jepson/papers/MannJepsonElMaraghiICPR2002.pdf

"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
from tt3d.traj_seg.utils import read_traj, wrap_angles
import matplotlib.pyplot as plt


def polynomial_model(t, p0, p1, p2):
    return p0 + p1 * t + p2 * t**2


def fit_polynomial(t, traj):
    """Fits a second-degree polynomial to the given trajectory."""
    coeffs = np.polyfit(t, traj, deg=2)
    return coeffs


def find_intersection(coeffs1, coeffs2):
    """Finds the intersection of two second-degree polynomials."""
    # The polynomials are of the form: a1*t^2 + b1*t + c1 = a2*t^2 + b2*t + c2
    # This reduces to: (a1 - a2)*t^2 + (b1 - b2)*t + (c1 - c2) = 0
    diff_coeffs = coeffs1 - coeffs2
    roots = np.roots(diff_coeffs)  # Solve the quadratic equation
    # print(roots)
    return roots


def get_accurate_bouncing_pose(t_before, traj_before, t_after, traj_after):
    """
    Fit second-degree polynomials to the trajectories before and after the bounce,
    and compute the intersection point as the bounce location.

    Args:
        t_before: Time points before the bounce.
        traj_before: Trajectory points (x, y) before the bounce.
        t_after: Time points after the bounce.
        traj_after: Trajectory points (x, y) after the bounce.

    Returns:
        t_bounce: Time of the bounce.
        (x_bounce, y_bounce): Position of the bounce.
    """

    def filter_real_roots(roots):
        """Filter out complex roots."""
        return np.real(roots[np.isreal(roots)])

    def get_valid_times(t_bounce_list, t_min, t_max):
        """Get valid bounce times within the time range."""
        return [t for t in t_bounce_list if t_min <= t <= t_max]

    # Plot
    # plt.plot(t_before, traj_before)
    # plt.plot(t_after, traj_after)
    # plt.show()

    # Fit polynomials for x and y coordinates before and after the bounce
    coeffs_x_before = fit_polynomial(t_before, traj_before[:, 0])
    coeffs_y_before = fit_polynomial(t_before, traj_before[:, 1])
    coeffs_x_after = fit_polynomial(t_after, traj_after[:, 0])
    coeffs_y_after = fit_polynomial(t_after, traj_after[:, 1])

    # Find the time of intersection (bounce time) for x and y coordinates
    t_bounce_x = filter_real_roots(find_intersection(coeffs_x_before, coeffs_x_after))
    t_bounce_y = filter_real_roots(find_intersection(coeffs_y_before, coeffs_y_after))

    # Time range constraint
    t_min, t_max = t_before.max() - 0.05, t_after.min() + 0.05
    valid_t_x = get_valid_times(t_bounce_x, t_min, t_max)
    valid_t_y = get_valid_times(t_bounce_y, t_min, t_max)

    # If no valid intersections, return -1
    if not valid_t_x and not valid_t_y:
        return (t_min + t_max) / 2, (
            (traj_before[-1, 0] + traj_after[0, 0]) / 2,
            (traj_before[-1, 1] + traj_after[0, 1]) / 2,
        )

    # Handle cases with two intersections for both x and y
    if len(t_bounce_x) == 2 and len(t_bounce_y) == 2:
        # Check how parallel the polynomes are at the intersection point
        # print(t_bounce_x)
        # print(valid_t_x)
        # print(t_bounce_y)
        # print(valid_t_y)
        der_x_bef = np.polyval(np.polyder(coeffs_x_before[::-1]), (t_min + t_max) / 2)
        der_x_aft = np.polyval(np.polyder(coeffs_x_after[::-1]), (t_min + t_max) / 2)
        delta_x = abs(der_x_bef - der_x_aft)
        # print(delta_x)
        der_y_bef = np.polyval(np.polyder(coeffs_y_before[::-1]), (t_min + t_max) / 2)
        der_y_aft = np.polyval(np.polyder(coeffs_y_after[::-1]), (t_min + t_max) / 2)
        delta_y = abs(der_y_bef - der_y_aft)
        # print(delta_y)

        if delta_x > 2 * delta_y:
            # print("x")
            t_bounce = valid_t_x[0]
        elif 2 * delta_x < delta_y:
            # print("y")
            t_bounce = valid_t_y[0]
        else:
            # print("mean")
            t_bounce = np.mean(valid_t_x + valid_t_y)
    else:
        t_bounce = np.mean(valid_t_x + valid_t_y)

    # Compute bounce position
    x_bounce = np.polyval(coeffs_x_before, t_bounce)
    y_bounce = np.polyval(coeffs_y_before, t_bounce)

    return t_bounce, (x_bounce, y_bounce)


def basic_segmenter(t, traj, deg=2, L=200, use_blur=False):
    """
    Segments trajectory data based on polynomial fitting with penalties.

    Parameters:
        t (array): Time or independent variable.
        traj (array): Trajectory data with shape (n, 2) for X and Y coordinates.
        deg (int): Polynomial degree for each segment.
        L (float): Penalty for each segment.

    Returns:
        np.ndarray: Indices of segment boundaries.
    """
    n = len(t)
    # Initialize dynamic programming tables
    phi = [L if x > 0 else 0 for x in range(deg + 1)]
    q = [[x - 1] if (x > 0) else [] for x in range(deg + 1)]
    # print(q)

    for v in range(deg, n):
        # print("v", v)
        min_cost = float("inf")
        best_u = None

        # Precompute segment costs for all valid u
        for u in range(v - deg + 1):
            # print("u", u, "v", v)
            try:
                # Fit X and Y trajectories separately
                popt_X, _, infodict_X, *_ = curve_fit(
                    polynomial_model, t[u : v + 1], traj[u : v + 1, 0], full_output=True
                )
                popt_Y, _, infodict_Y, *_ = curve_fit(
                    polynomial_model, t[u : v + 1], traj[u : v + 1, 1], full_output=True
                )

                if use_blur:
                    x_der = np.polyval(np.polyder(popt_X[::-1]), t[u : v + 1])
                    y_der = np.polyval(np.polyder(popt_Y[::-1]), t[u : v + 1])
                    pred_angle = np.degrees(np.arctan2(y_der, x_der))
                    # Wrap able between -90 and 90
                    pred_angle = wrap_angles(pred_angle)

                    cost_blur = np.abs(pred_angle - traj[u : v + 1, 4])
                    cost_blur = np.abs(wrap_angles(cost_blur))
                    # Remove errors where the blur is too short
                    cost_blur = np.sum(cost_blur[traj[u : v + 1, 3] >= 3])
                    # print(t[u : v + 1])
                    # print(x_der)
                    # print(y_der)
                    # print(pred_angle)
                    # print(traj[u : v + 1, 4])
                    # print()

                # Residuals as cost
                res_X = np.sum(infodict_X["fvec"] ** 2)
                res_Y = np.sum(infodict_Y["fvec"] ** 2)
                # print(res_X + res_Y)
                if use_blur:
                    total_cost = phi[u] + res_X + res_Y + 2 * cost_blur
                else:
                    total_cost = phi[u] + res_X + res_Y

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_u = u
            except RuntimeError:
                # Handle cases where curve fitting fails
                continue

        if best_u is not None:
            # print(best_u)
            phi.append(min_cost + L)
            q.append([*q[best_u], best_u])
        else:
            # In case no valid fitting was found
            phi.append(float("inf"))
            q.append(q[-1])

    # Extract segment boundaries
    q_sol = q[-1]
    # print(q_sol)
    # print(phi[-1])
    return np.array(q_sol)


def classify_q(qs, t, traj):
    q_racket = []
    q_table = []
    pad = 3
    # print(traj)
    for i in range(len(qs)):
        u_vel_before = np.mean(
            np.diff(traj[qs[i] - pad : qs[i], 0], axis=0)
            / np.diff(t[qs[i] - pad : qs[i]])[:, None]
        )
        u_vel_after = np.mean(
            np.diff(traj[qs[i] : qs[i] + pad, 0], axis=0)
            / np.diff(t[qs[i] : qs[i] + pad])[:, None]
        )
        if u_vel_before * u_vel_after > 0:
            q_table.append(qs[i])
        else:
            q_racket.append(qs[i])
    return q_racket, q_table


if __name__ == "__main__":
    # csv_path = Path("/home/gossard/Git/tt3d/data/demo_video/traj/satsuki_001.csv")
    # fps = 30
    csv_path = Path("/home/gossard/Git/tt3d/data//demo_video/traj/ma_lebrun_001.csv")
    fps = 25

    traj = read_traj(csv_path)
    traj = traj[2:]

    # Generate time step axis
    n = traj.shape[0]
    t = traj[:, 0] / fps
    idxs = traj[:, 0]
    traj = traj[:, 1:]

    # Filter out time steps where the ball is not visible
    t = t[traj[:, 1] != 0]
    traj = traj[traj[:, 1] != 0]

    # Show the trajectory
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].scatter(t, traj[:, 0])
    axs[0].scatter(t, traj[:, 1])
    axs[1].scatter(t, traj[:, 3])
    axs[2].scatter(t, traj[:, 4])
    plt.show()

    ## Plotting the trajectory segmentation

    q_sol = basic_segmenter(t, traj, use_blur=True)
    # q_sol = q_sol[1:]
    print("q_sol:", q_sol)
    q_racket, q_table = classify_q(q_sol, t, traj)
    print("Racket: ", q_racket)
    print("Table: ", q_table)

    # Plot the velocity
    dt = np.diff(t)
    vx = np.diff(traj[:, 0]) / dt
    vy = np.diff(traj[:, 1]) / dt

    f, axs = plt.subplots(4, figsize=(10, 8), sharex=True)
    for x in q_table:
        for i in range(4):
            axs[i].axvline(x=t[x], c="r", zorder=0)
    for x in q_racket:
        for i in range(4):
            axs[i].axvline(x=t[x], c="b", zorder=0)
    axs[2].axhline(y=3, c="black", linestyle="--")

    axs[0].scatter(t, traj[:, 0], s=20)
    axs[1].scatter(t, traj[:, 1], s=20)
    axs[2].scatter(t, traj[:, 3], s=20)
    l_thres = traj[:, 3] >= 3
    axs[3].scatter(t[l_thres], traj[l_thres, 4], s=20, label="Used")
    axs[3].scatter(t[~l_thres], traj[~l_thres, 4], s=20, marker="x", label="Not used")
    axs[0].set_ylabel("u")
    axs[1].set_ylabel("v")
    axs[2].set_ylabel("l")
    axs[3].set_ylabel(r"$\theta$")
    axs[3].set_xlabel("Time step")
    axs[3].legend()

    # Plot the fitted polynomes
    ts_exact = [0]
    for i in range(1, len(q_sol) - 1):
        t_bounce, (x_bounce, y_bounce) = get_accurate_bouncing_pose(
            t[q_sol[i - 1] : q_sol[i]],
            traj[q_sol[i - 1] : q_sol[i]],
            t[q_sol[i] : q_sol[i + 1]],
            traj[q_sol[i] : q_sol[i + 1]],
        )
        if t_bounce != -1:
            axs[0].scatter(t_bounce, x_bounce, s=50, marker="x", c="black", zorder=3)
            axs[1].scatter(t_bounce, y_bounce, s=50, marker="x", c="black", zorder=3)
        ts_exact.append(t_bounce)
    # For the last one
    t_bounce, (x_bounce, y_bounce) = get_accurate_bouncing_pose(
        t[q_sol[-2] : q_sol[-1]],
        traj[q_sol[-2] : q_sol[-1]],
        t[q_sol[-1] : -1],
        traj[q_sol[-1] : -1],
    )
    axs[0].scatter(t_bounce, x_bounce, s=50, marker="x", c="black", zorder=3)
    axs[1].scatter(t_bounce, y_bounce, s=50, marker="x", c="black", zorder=3)
    ts_exact.append(t_bounce)

    ts_exact = np.array(ts_exact).astype(np.float32)
    print("t_exact", ts_exact)

    for i in range(len(q_sol) - 1):
        t_temp = np.linspace(ts_exact[i], ts_exact[i + 1], 10)
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
        x_der = np.polyval(np.polyder(popt_X[::-1]), t[q_sol[i] : q_sol[i + 1]])
        y_der = np.polyval(np.polyder(popt_Y[::-1]), t[q_sol[i] : q_sol[i + 1]])
        pred_angle = np.degrees(np.arctan2(y_der, x_der))
        pred_angle = wrap_angles(pred_angle)
        cost_blur = np.abs(pred_angle - traj[q_sol[i] : q_sol[i + 1], 4])
        cost_blur = np.abs(wrap_angles(cost_blur))
        # axs[4].scatter(t[q_sol[i] : q_sol[i + 1]], cost_blur)
        axs[3].plot(t[q_sol[i] : q_sol[i + 1]], pred_angle)

    t_temp = np.linspace(ts_exact[-1], t[-1], 10)
    # For the x coord
    popt_X, _, infodict_X, *_ = curve_fit(
        polynomial_model,
        t[q_sol[-1] : -1],
        traj[q_sol[-1] : -1, 0],
        full_output=True,
    )
    axs[0].plot(
        t_temp,
        polynomial_model(t_temp, *popt_X),
    )
    popt_Y, _, infodict_Y, *_ = curve_fit(
        polynomial_model,
        t[q_sol[-1] : -1],
        traj[q_sol[-1] : -1, 1],
        full_output=True,
    )
    axs[1].plot(
        t_temp,
        polynomial_model(t_temp, *popt_Y),
    )
    x_der = np.polyval(np.polyder(popt_X[::-1]), t_temp)
    y_der = np.polyval(np.polyder(popt_Y[::-1]), t_temp)
    pred_angle = np.degrees(np.arctan2(y_der, x_der))
    pred_angle = wrap_angles(pred_angle)
    # Remove errors where the blur is too short
    cost_blur = np.abs(pred_angle - traj[q_sol[-1], -1])
    cost_blur = np.abs(wrap_angles(cost_blur))
    # axs[4].scatter(t_temp, cost_blur)
    axs[3].plot(t_temp, pred_angle)

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
