"""
Utility plotting functions
"""
import matplotlib.pyplot as plt
import io
import matplotlib.animation as animation
from tqdm import tqdm
import cv2
import numpy as np
from tt3d.rally.plots import draw_tt_table


def draw_table_on_frame(frame, T, K):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    L = 1.370
    l = 0.7625
    axis = np.array([[0, 0, 0], [l, 0, 0], [0, L, 0], [0, 0, 0.5]])
    pts, _ = cv2.projectPoints(axis, rvec, tvec, K, np.zeros(5))
    # print(pts)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(1, len(pts)):
        cv2.arrowedLine(
            frame,
            (int(pts[0][0][0]), int(pts[0][0][1])),
            (int(pts[i][0][0]), int(pts[i][0][1])),
            color=colors[i - 1],
            thickness=2,
        )
    return frame


def plot_3D_skeleton(j3d, fig, ax):
    joint_pairs = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [8, 11],
        [8, 14],
        [9, 10],
        [11, 12],
        [12, 13],
        [14, 15],
        [15, 16],
    ]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    plt.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    xs, ys, zs = [np.array([j3d[3, j], j3d[3, j]]) for j in range(3)]
    ax.plot(
        xs,
        ys,
        zs,
        color="red",
        lw=3,
        marker="o",
        markerfacecolor="w",
        markersize=5,
        markeredgewidth=2,
    )  # axis transformation for visualization
    for i in range(len(joint_pairs)):
        limb = joint_pairs[i]
        xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
        if joint_pairs[i] in joint_pairs_left:
            ax.plot(
                xs,
                ys,
                zs,
                color=color_left,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization
        elif joint_pairs[i] in joint_pairs_right:
            ax.plot(
                xs,
                ys,
                zs,
                color=color_right,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization
        else:
            ax.plot(
                xs,
                ys,
                zs,
                color=color_mid,
                lw=3,
                marker="o",
                markerfacecolor="w",
                markersize=3,
                markeredgewidth=2,
            )  # axis transformation for visualization


def draw_skeleton_on_frame(frame, keypoints, gt=False):
    """
    Draws the 2D keypoints and skeleton on a video frame.

    Args:
        frame (np.ndarray): The video frame (image) on which to draw.
        keypoints (np.ndarray): A NumPy array of shape (1, 17, 2) representing the keypoints.
    """
    # Define joint pairs for skeleton
    joint_pairs = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 4],
        [4, 5],
        [5, 6],
        [0, 7],
        [7, 8],
        [8, 9],
        [8, 11],
        [8, 14],
        [9, 10],
        [11, 12],
        [12, 13],
        [14, 15],
        [15, 16],
    ]

    # Define specific joint pairs for left and right body parts
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]

    # Define colors for different body parts in BGR format
    if gt:
        color_mid = (126, 71, 0)  # Example color for middle body parts
        color_left = (94, 49, 2)  # Example color for left body parts
        color_right = (175, 112, 47)  # Example color for right body parts
    else:
        color_mid = (0, 71, 126)  # Example color for middle body parts
        color_left = (2, 49, 94)  # Example color for left body parts
        color_right = (47, 112, 175)  # Example color for right body parts

    # Reshape the keypoints array to ensure correct shape
    keypoints = keypoints.reshape(-1, 2)

    # Draw the skeleton
    for i, limb in enumerate(joint_pairs):
        pt1, pt2 = keypoints[limb[0]], keypoints[limb[1]]
        if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
            continue  # Skip if keypoints are missing

        # Choose color based on the limb
        if limb in joint_pairs_left:
            color = color_left
        elif limb in joint_pairs_right:
            color = color_right
        else:
            color = color_mid

        # Draw line between keypoints
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        cv2.line(frame, pt1, pt2, color, thickness=3)
        cv2.circle(frame, pt1, radius=5, color=(255, 255, 255), thickness=-1)
        cv2.circle(frame, pt2, radius=5, color=(255, 255, 255), thickness=-1)

    return frame


def get_img_from_fig(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_3d_povs(p1, p2, ball=None):
    # Creating the figure
    fig = plt.figure(0, figsize=(10, 10))
    ax_side = fig.add_subplot(2, 2, 1, projection="3d")
    ax_back = fig.add_subplot(2, 2, 2, projection="3d")
    ax_obl = fig.add_subplot(2, 2, 3, projection="3d")
    ax_top = fig.add_subplot(2, 2, 4, projection="3d")
    x_lim = [3, -3]
    y_lim = [-3, 3]
    z_lim = [0, 3]

    draw_tt_table(ax_side)
    plot_3D_skeleton(p1, fig, ax_side)
    plot_3D_skeleton(p2, fig, ax_side)
    ax_side.view_init(elev=10, azim=180, roll=0)
    # ax_side.view_init(elev=10, azim=-110, roll=0)
    ax_side.set_xlim(x_lim)
    ax_side.set_ylim(y_lim)
    ax_side.set_zlim(z_lim)  # Adding a bit of space above the table
    ax_side.set_box_aspect(
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )  # Aspect ratio is (width, length, height)
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Y")
    ax_side.set_zlabel("Z")

    draw_tt_table(ax_obl)
    plot_3D_skeleton(p1, fig, ax_obl)
    plot_3D_skeleton(p2, fig, ax_obl)
    ax_obl.view_init(elev=10, azim=45, roll=0)
    ax_obl.set_xlim(x_lim)
    ax_obl.set_ylim(y_lim)
    ax_obl.set_zlim(z_lim)  # Adding a bit of space above the table
    ax_obl.set_box_aspect(
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )  # Aspect ratio is (width, length, height)
    ax_obl.set_xlabel("X")
    ax_obl.set_ylabel("Y")
    ax_obl.set_zlabel("Z")

    draw_tt_table(ax_top)
    plot_3D_skeleton(p1, fig, ax_top)
    plot_3D_skeleton(p2, fig, ax_top)
    ax_top.view_init(elev=90, azim=180, roll=0)
    # ax_top.view_init(elev=10, azim=-110, roll=0)
    ax_top.set_xlim(x_lim)
    ax_top.set_ylim(y_lim)
    ax_top.set_zlim(z_lim)  # Adding a bit of space above the table
    ax_top.set_box_aspect(
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )  # Aspect ratio is (width, length, height)
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_zlabel("Z")

    draw_tt_table(ax_back)
    plot_3D_skeleton(p1, fig, ax_back)
    plot_3D_skeleton(p2, fig, ax_back)
    ax_back.view_init(elev=10, azim=-90, roll=0)
    # ax_back.view_init(elev=10, azim=-110, roll=0)
    ax_back.set_xlim(x_lim)
    ax_back.set_ylim(y_lim)
    ax_back.set_zlim(z_lim)  # Adding a bit of space above the table
    ax_back.set_box_aspect(
        [x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]]
    )  # Aspect ratio is (width, length, height)
    ax_back.set_xlabel("X")
    ax_back.set_ylabel("Y")
    ax_back.set_zlabel("Z")
    if ball is not None:
        ax_side.scatter(*ball, color="red", s=50, zorder=3)
        ax_top.scatter(*ball, color="red", s=50, zorder=3)
        ax_obl.scatter(*ball, color="red", s=50, zorder=3)
        ax_back.scatter(*ball, color="red", s=50, zorder=3)
    plt.tight_layout()
    return fig
