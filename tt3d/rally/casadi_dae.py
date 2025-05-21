"""
Casadi implementation of the dynamics
The dynamics are modeled as a Differential Algebraic Equation (DAE)
"""
import casadi as ca
from constants import KD, KM, M, R, MU, COR


G = ca.DM([0, 0, -9.81])  # Gravity acceleration (m/s^2)
ode_method = "collocation"


def rebuild_diff(t, p_bounce):
    """
    Differentiable version of the ball trajectory reconstruction
    Returns the variables involved in the dae
    Input:
        t: np.array timestamps
    Output:
        v0: bounce velocity
        w0: bounce spin
        traj: 3xN ball trajectory
    """
    # Calculate trajectory before bounce
    t_before = t[t < 0]
    v0 = ca.MX.sym("v0", 3)
    w0 = ca.MX.sym("w0", 3)
    p0 = ca.DM(p_bounce)

    # Defining the dynamics
    p = ca.MX.sym("p", 3)  # Ball position
    v = ca.MX.sym("v", 3)  # Ball velocity
    w = ca.MX.sym("w", 3)
    v, a = ball_dynamics(p, v, w)
    xdot = ca.vertcat(v, a)  # dx/dt
    x = ca.vertcat(p, v)
    dae = {"x": x, "ode": xdot, "p": w}  # Defjine system equations

    # Defining the ODE for backward in time
    bk_I = ca.integrator("bk_I", ode_method, dae, 0, t_before[::-1])
    bk_res = bk_I(x0=ca.vertcat(p0, v0), p=w0)  # Start at [0.2, 0.3] with u=0.4

    bk_p = bk_res["xf"].T[::-1, :3]

    # Calculate trajectory for after
    t_after = t[t > 0]

    # Use the bounce model to calculate after bounce velocity and spin
    v1, w1 = bounce_model(v0, w0)
    fw_I = ca.integrator(
        "fw_I", ode_method, dae, 0, t_after
    )  # Integrate over T seconds
    fw_res = fw_I(x0=ca.vertcat(p0, v1), p=w1)  # Start at [0.2, 0.3] with u=0.4
    fw_p = fw_res["xf"].T[:, :3]

    traj = ca.vertcat(bk_p, fw_p)
    return v0, w0, traj


def rebuild(t, p_bounce, v0, w0):
    """
    Rebuilds the ball's trajectory given the bounce state and the required timestamps
    """
    # Calculate trajectory   error for before
    t_before = t[t < 0]
    v0 = ca.DM(v0)
    w0 = ca.DM(w0)
    p0 = ca.DM(p_bounce)

    p = ca.MX.sym("p", 3)  # Ball position
    v = ca.MX.sym("v", 3)  # Ball velocity
    w = ca.MX.sym("w", 3)
    v, a = ball_dynamics(p, v, w)

    xdot = ca.vertcat(v, a)  # dx/dt
    x = ca.vertcat(p, v)
    # Defining the ODE for backward in time
    bk_dae = {"x": x, "ode": xdot, "p": w}  # Defjine system equations
    bk_I = ca.integrator("bk_I", ode_method, bk_dae, 0, t_before[::-1])
    bk_res = bk_I(x0=ca.vertcat(p0, v0), p=w0)  # Start at [0.2, 0.3] with u=0.4

    bk_p = bk_res["xf"].T[::-1, :3]

    # Calculate reprojection for after
    t_after = t[t > 0]

    # Use the bounce model to calculate after bounce velocity and spin
    v1, w1 = bounce_model(v0, w0)
    fw_dae = {"x": x, "ode": xdot, "p": w}  # Define system equations
    fw_I = ca.integrator(
        "fw_I", ode_method, fw_dae, 0, t_after
    )  # Integrate over T seconds
    fw_res = fw_I(x0=ca.vertcat(p0, v1), p=w1)  # Start at [0.2, 0.3] with u=0.4
    fw_p = fw_res["xf"].T[:, :3]

    traj = ca.vertcat(bk_p, fw_p)
    return traj


def ball_dynamics(p, v, w):
    """Compute ball dynamics with drag and Magnus effect."""
    speed = ca.norm_2(v)
    drag_force = -KD * speed * v
    magnus_force = KM * ca.cross(w, v)
    a = G + drag_force / M + magnus_force / M
    return v, a


def get_surface_v(v, w):
    """Calculate the surface velocity for symbolic vectors v and w."""
    # CasADi compatible surface velocity calculation
    return ca.sqrt((v[0] - R * w[1]) ** 2 + (v[1] + R * w[0]) ** 2)


def get_alpha(v, w):
    """Calculate the alpha value for symbolic vectors v and w."""
    # Compute the surface velocity (use CasADi expressions)
    surf_v = get_surface_v(v, w)

    # CasADi-compatible alpha calculation
    alpha = MU * (1 + COR) * ca.fabs(v[2]) / surf_v
    return alpha


def bounce_model(v, w):
    """CasADi-compatible bounce dynamics for symbolic vectors v and w."""
    assert v.shape == (3, 1) and w.shape == (3, 1)

    alpha = get_alpha(v, w)

    # Rolling matrices
    A_roll = ca.DM([[0.6, 0, 0], [0, 0.6, 0], [0, 0, -COR]])
    B_roll = ca.DM([[0, 0.4 * R, 0], [-0.4 * R, 0, 0], [0, 0, 0]])
    C_roll = ca.DM([[0, -0.6 / R, 0], [0.6 / R, 0, 0], [0, 0, 0]])
    D_roll = ca.DM([[0.4, 0, 0], [0, 0.4, 0], [0, 0, 1]])

    # Sliding matrices
    A_slide = ca.vertcat(
        ca.horzcat(1 - alpha, 0, 0),
        ca.horzcat(0, 1 - alpha, 0),
        ca.horzcat(0, 0, -COR),
    )
    B_slide = ca.vertcat(
        ca.horzcat(0, alpha * R, 0),
        ca.horzcat(-alpha * R, 0, 0),
        ca.horzcat(0, 0, 0),
    )
    C_slide = ca.vertcat(
        ca.horzcat(0, -3 / 2 * alpha / R, 0),
        ca.horzcat(3 / 2 * alpha / R, 0, 0),
        ca.horzcat(0, 0, 0),
    )
    D_slide = ca.vertcat(
        ca.horzcat(1 - 3 / 2 * alpha, 0, 0),
        ca.horzcat(0, 1 - 3 / 2 * alpha, 0),
        ca.horzcat(0, 0, 1),
    )

    # Select rolling or sliding matrices
    A = ca.if_else(alpha > 0.4, A_roll, A_slide)
    B = ca.if_else(alpha > 0.4, B_roll, B_slide)
    C = ca.if_else(alpha > 0.4, C_roll, C_slide)
    D = ca.if_else(alpha > 0.4, D_roll, D_slide)

    # Compute va and wa using matrix multiplication
    va = ca.mtimes(A, v) + ca.mtimes(B, w)
    wa = ca.mtimes(C, v) + ca.mtimes(D, w)

    return va, wa
