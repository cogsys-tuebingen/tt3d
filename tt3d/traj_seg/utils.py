import pandas as pd
import numpy as np


def wrap_angles(angles):
    angles = np.asarray(angles)  # Ensure input is a NumPy array
    angles = np.where(angles > 90, angles - 180, angles)
    angles = np.where(angles < -90, angles + 180, angles)
    return angles


def read_traj(csv_path, frame_dir=None):
    df = pd.read_csv(csv_path)
    fids, visis, xs, ys, ls, thetas = (
        df["Frame"].tolist(),
        df["Visibility"].tolist(),
        df["X"].tolist(),
        df["Y"].tolist(),
        df["L"].tolist(),
        df["Theta"].tolist(),
    )
    xyvs = []
    for fid, visi, x, y, l, theta in zip(fids, visis, xs, ys, ls, thetas):
        if int(fid) in [item[0] for item in xyvs]:
            raise KeyError("fid {} already exists".format(fid))
        xyvs.append(
            [
                int(fid),  # Frame idx aka time*fps
                float(x),  # x-coordinate
                float(y),  # y-coordinate
                int(visi),  # is_visible
                float(l),  # Blur length
                float(theta),  # Blur angle
            ]
        )
    return np.array(xyvs)
