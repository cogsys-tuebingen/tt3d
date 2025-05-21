import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_tt_table(ax):
    # Table dimensions
    length = 2.74  # meters (length along y-axis)
    width = 1.525  # meters (width along x-axis)
    height = 0.0  # 0.76  # meters
    net_height = 0.1525  # meters

    # Create table surface
    x = np.array([-width / 2, width / 2])
    y = np.array([-length / 2, length / 2])
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Draw the top surface of the table
    ax.plot_surface(X, Y, Z + height, color="green", alpha=0.8, zorder=0)

    # Create net surface
    net_x = np.array([-width / 2, width / 2])
    net_z = np.array([height, height + net_height])
    net_X, net_Z = np.meshgrid(net_x, net_z)
    net_Y = np.zeros_like(net_X)

    # Draw the top surface of the table
    ax.plot_surface(net_X, net_Y, net_Z, color="gray", alpha=0.8, zorder=1)

    # Add table edges
    edges_x = [-width / 2, width / 2, width / 2, -width / 2, -width / 2]
    edges_y = [-length / 2, -length / 2, length / 2, length / 2, -length / 2]
    edges_z = [height, height, height, height, height]

    ax.plot(edges_x, edges_y, edges_z, color="black", zorder=0)


def draw_tennis_court(ax):
    # Tennis court dimensions (in meters)
    court_length = 23.77  # total length (y-axis)
    court_width = 10.97  # total width (x-axis, doubles court)
    singles_width = 8.23  # singles court width
    service_line_distance = 6.40  # distance from net to service line
    net_height = 0.914  # net height at center
    court_height = 0.0  # court surface height

    # Create court surface
    x = np.array([-court_width / 2, court_width / 2])
    y = np.array([-court_length / 2, court_length / 2])
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X) + court_height

    ax.plot_surface(X, Y, Z, color="green", alpha=0.8, zorder=0)

    # Create net surface
    net_x = np.array([-court_width / 2, court_width / 2])
    net_z = np.array([court_height, court_height + net_height])
    net_X, net_Z = np.meshgrid(net_x, net_z)
    net_Y = np.zeros_like(net_X)

    ax.plot_surface(net_X, net_Y, net_Z, color="gray", alpha=0.8, zorder=0.5)

    # Define important court lines
    lines = [
        # Outer boundary (doubles court)
        [[-court_width / 2, -court_width / 2], [-court_length / 2, court_length / 2]],
        [[court_width / 2, court_width / 2], [-court_length / 2, court_length / 2]],
        [[-court_width / 2, court_width / 2], [-court_length / 2, -court_length / 2]],
        [[-court_width / 2, court_width / 2], [court_length / 2, court_length / 2]],
        # Singles sidelines
        [
            [-singles_width / 2, -singles_width / 2],
            [-court_length / 2, court_length / 2],
        ],
        [[singles_width / 2, singles_width / 2], [-court_length / 2, court_length / 2]],
        # Service lines
        [
            [-singles_width / 2, singles_width / 2],
            [-service_line_distance, -service_line_distance],
        ],
        [
            [-singles_width / 2, singles_width / 2],
            [service_line_distance, service_line_distance],
        ],
        # Center service line
        [[0, 0], [-service_line_distance, service_line_distance]],
        # Net line
        [[-court_width / 2, court_width / 2], [0, 0]],
    ]

    # Draw court lines
    for line in lines:
        ax.plot(
            line[0],
            line[1],
            [court_height, court_height],
            color="white",
            linewidth=2,
            zorder=1,
        )

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("Tennis Court")


def animate_3d_positions(
    positions, timestamps, output_file="trajectory.mp4", fps=30, field=None
):
    """
    Generates an animation of 3D positions and writes it to a video file.

    Parameters:
    - positions: (N, 3) NumPy array of 3D positions (x, y, z).
    - timestamps: (N,) NumPy array of timestamps.
    - output_file: Name of the output video file (default: "trajectory.mp4").
    - fps: Frames per second for the video (default: 30).
    """
    assert positions.shape[1] == 3, "Positions must be a (N, 3) array."
    assert len(timestamps) == len(
        positions
    ), "Timestamps must match positions in length."

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory Animation")

    # Set axis limits based on data range
    # ax.set_xlim(np.min(positions[:, 0]), np.max(positions[:, 0]))
    # ax.set_ylim(np.min(positions[:, 1]), np.max(positions[:, 1]))
    # ax.set_zlim(np.min(positions[:, 2]), np.max(positions[:, 2]))

    # Line plot and point (for animation)
    (line,) = ax.plot([], [], [], "b-", lw=2, zorder=3)
    (point,) = ax.plot([], [], [], "ro", zorder=3)
    # if field == "tennis":
    # draw_tennis_court(ax)
    draw_table_tennis_table(ax)
    set_axes_equal(ax)
    ax.view_init(elev=30, azim=180)

    def update(frame):
        """Update function for the animation"""
        line.set_data(positions[:frame, 0], positions[:frame, 1])
        line.set_3d_properties(positions[:frame, 2])

        point.set_data(positions[frame, 0], positions[frame, 1])
        point.set_3d_properties(positions[frame, 2])
        return line, point

    ani = animation.FuncAnimation(
        fig, update, frames=len(timestamps), interval=1000 / fps, blit=False
    )

    # Save animation to file
    ani.save(output_file, writer="ffmpeg", fps=fps)
    plt.close(fig)

    print(f"Animation saved as {output_file}")
