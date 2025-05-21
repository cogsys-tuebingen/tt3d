"""
Class to find the camera's focal length via optimization
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


class ReprojOpt:
    def __init__(
        self,
        h,
        w,
        object_points,
        f_ini=1500,
        dist=np.zeros(5),
    ):
        self.object_points = np.array(object_points, dtype=np.float32)
        self.f_ini = f_ini
        self.dist = dist
        self.pnp_method = cv2.SOLVEPNP_ITERATIVE

        self.K = np.array(
            [
                [f_ini, 0, w / 2],
                [0, f_ini, h / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def update_K(self, f):
        self.K[0, 0] = f
        self.K[1, 1] = f

    def pose_est(self, f, img_pts):
        self.update_K(f)
        img_pts = np.array(img_pts, dtype=np.float32)

        _, rvec, tvec = cv2.solvePnP(
            self.object_points,
            img_pts,
            self.K,
            self.dist,
            None,
            None,
            useExtrinsicGuess=False,
            flags=self.pnp_method,
        )
        return rvec, tvec

    def reprojection_error(self, f, img_pts):
        self.update_K(f)
        img_pts = np.array(img_pts, dtype=np.float32)
        _, rvec, tvec = cv2.solvePnP(
            self.object_points,
            img_pts,
            self.K,
            self.dist,
            None,
            None,
            useExtrinsicGuess=False,
            flags=self.pnp_method,
        )
        projected_points, _ = cv2.projectPoints(
            self.object_points,
            rvec,
            tvec,
            self.K,
            self.dist,
        )
        projected_points = projected_points.reshape(-1, 2)

        error = img_pts - projected_points
        error = np.mean(np.linalg.norm(error, axis=1))
        return error

    def optimize_f(self, img_pts):
        img_pts = np.array(img_pts, dtype=np.float32)
        result = least_squares(
            self.reprojection_error,
            self.f_ini,
            method="trf",
            bounds=(500, 20000),
            args=(img_pts,),
        )
        # print(result)
        error = result.cost

        optimized_f = result.x[0]
        rvec, tvec = self.pose_est(optimized_f, img_pts)

        return optimized_f, rvec, tvec, error

    def map_error(self, img_pts, low=500, high=7000):
        img_pts = np.array(img_pts, dtype=np.float32)
        fs = np.linspace(low, high, 50)
        errors = []
        for f in fs:
            errors.append(self.reprojection_error(f, img_pts))
        plt.plot(fs, errors)
        plt.xlabel("f")
        plt.ylabel("Reprojection error")
        plt.show()


if __name__ == "__main__":
    # U = np.array(
    #     [
    #         [320.0, 240.0],
    #         [475.3294, 290.2849],
    #         [275.81876, 388.31577],
    #         [348.14145, 230.89165],
    #         [425.5374, 442.7193],
    #         [310.43387, 355.6944],
    #     ]
    # )
    # X = np.array(
    #     [
    #         [0.0, 0.0, 0.0],
    #         [1.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0],
    #         [1, 1, 0],
    #         [0, 1, 1],
    #     ]
    # )
    # Test values
    # U = np.array(
    #     [
    #         [745.07, 216.77],
    #         # [639.31, 217.3, 1],
    #         [533.8, 216.1],
    #         [517.61, 370.99],
    #         # [640.81, 371, 1],
    #         [763.01, 371.09],
    #     ]
    # )
    # Real values
    U = np.array(
        [
            [482, 396],
            [457, 476],
            [836, 481],
            [807, 401],
        ]
    )

    X = np.array(
        [
            [0.7625, 1.37, 0],
            # [0, 1.37, 0],
            [-0.7625, 1.37, 0],
            [-0.7625, -1.37, 0],
            # [0, -1.37, 0],
            [0.7625, -1.37, 0],
        ]
    )
    f_ini = 1000
    estimator = ReprojOpt(U, X, f_ini, 720, 1280)
    f = estimator.optimize_focal_length()
    estimator.map_error(500, 5500)
    print(estimator.pose_est(f))

    # K = np.array(
    #     [
    #         [f, 0, 320],
    #         [0, f, 240],
    #         [0, 0, 1],
    #     ]
    # )
    # dist = np.zeros(5)
    # rvec, _ = cv2.Rodrigues(R)
    # projection, _ = cv2.projectPoints(X, rvec, T, K, dist)
    # projection = np.squeeze(projection)
    # print(projection)
    # print(U - projection)
