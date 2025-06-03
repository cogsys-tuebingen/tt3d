"""
Class for table calibration
"""
import cv2
from itertools import combinations
import numpy as np
from utils import postprocess, preprocess
import matplotlib.pyplot as plt
from PIL import Image
from reproj_optimizer import ReprojOpt
from table_segmenter import TableSegmenter
from feature_extractor import FeatureExtractor
import torch
from pathlib import Path
from quadrilateral_bounder import QuadrilateralBounder


MODEL_PATH = Path("./weights/table_segmentation.ckpt")


class TableCalibrator:
    def __init__(
        self, h=None, w=None, f_ini=1000, segmentation_model_path=MODEL_PATH
    ) -> None:
        self.f_ini = f_ini
        self.dist = np.zeros(5)
        self.er_thres = 10
        self.table_corners_3d = np.array(
            [
                [-0.7625, 1.37, 0],
                [0.7625, 1.37, 0],
                [0.7625, -1.37, 0],
                [-0.7625, -1.37, 0],
            ]
        )
        self.table_net_3d = np.array(
            [
                [-0.7625, 0, 0.1525],
                [0.7625, 0, 0.1525],
            ]
        )
        self.table_midline_3d = np.array(
            [
                [0, 1.37, 0],
                [0, -1.37, 0],
            ]
        )
        print("Using: ", segmentation_model_path)
        self.init_segmenter(segmentation_model_path)
        # self.qb = QuadrilateralBounder()
        if h is not None:
            self.set_size(h, w)

    def set_size(self, h, w):
        self.h = h
        self.w = w
        self.corner_reproj_optimizer = ReprojOpt(self.h, self.w, self.table_corners_3d)
        self.reproj_optimizer_2 = ReprojOpt(
            self.h,
            self.w,
            np.vstack([self.table_corners_3d, self.table_midline_3d]),
        )
        self.fe = FeatureExtractor(self.h, self.w)

    def init_segmenter(self, model_path):
        self.table_segmenter = TableSegmenter.load_from_checkpoint(str(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.table_segmenter.to(device)
        self.table_segmenter.eval()

    def segment(self, img):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        sizes = [pil_img.size[::-1]]

        # Run the table segmentation model
        with torch.no_grad():
            t_imgs = preprocess([pil_img])
            t_masks = self.table_segmenter.infer(t_imgs)
            masks = postprocess(t_masks, sizes)
        return masks[0].astype(np.uint8)

    def get_table_pose_extra(self, features):
        # First corner order
        f1, rvec1, tvec1, rmse1 = self.reproj_optimizer_2.optimize_f(features)

        # Second corner order
        features = features[[2, 1, 0, 3, 4, 5], :]
        f2, rvec2, tvec2, rmse2 = self.reproj_optimizer_2.optimize_f(features)

        if rmse2 > 200 * rmse1:
            # estimator1.map_error()
            self.update_K(f1)
            return rvec1, tvec1, f1, rmse1
        elif rmse1 > 200 * rmse2:
            # estimator2.map_error()
            self.update_K(f2)
            return rvec2, tvec2, f2, rmse2
        else:
            if tvec1[2] < tvec2[2]:
                self.update_K(f1)
                return rvec1, tvec1, f1, rmse1
            else:
                self.update_K(f2)
                return rvec2, tvec2, f2, rmse2

    def process_corners(self, corners2d):
        # First corner order
        f1, rvec1, tvec1, rmse1 = self.corner_reproj_optimizer.optimize_f(corners2d)

        # Second corner order
        corners2d = corners2d[[2, 1, 0, 3], :]
        f2, rvec2, tvec2, rmse2 = self.corner_reproj_optimizer.optimize_f(corners2d)

        # print(tvec1[2], tvec2[2])
        # print(f1, f2)
        # print(rmse1, rmse2)
        # print()
        if rmse2 > 200 * rmse1:
            # estimator1.map_error()
            # print("1")
            self.update_K(f1)
            return rvec1, tvec1, f1, rmse1
        elif rmse1 > 200 * rmse2:
            # print("2")
            # estimator2.map_error()
            self.update_K(f2)
            return rvec2, tvec2, f2, rmse2
        else:
            if tvec1[2] < tvec2[2]:
                # print("3")
                # estimator1.map_error(e
                self.update_K(f1)
                return rvec1, tvec1, f1, rmse1
            else:
                # print("4")
                # estimator2.map_error()
                self.update_K(f2)
                return rvec2, tvec2, f2, rmse2

    def process(self, img, debug=False):
        # Set the image dimensions
        h, w = img.shape[:2]
        self.set_size(h, w)

        mask = self.segment(img)

        # Clean the mask
        mask = self.fe.clean_mask(mask)

        # Get the contours of the table
        contours = self.fe.get_contours(mask)
        contours = self.fe.filter_contours(contours)
        # for i in range(len(contours)):
        #     epsilon = 0.005 * cv2.arcLength(contours[i], True)
        #     contours[i] = cv2.approxPolyDP(contours[i], epsilon, True)
        # quad = self.qb.find_best_parallelogram(contours[0])
        # self.qb.draw_parallelogram(img, quad)

        debug_img = self.fe.draw_contours(img, contours) if debug else img
        # plt.imshow(debug_img)
        # plt.show()

        # Extract edges of the table
        if len(contours) == 0:
            return None, None, None, None, debug_img
        # With convex, more lines are detected
        # cvx_contour = self.fe.get_cvx_contour(contours)
        # lines = self.fe.get_table_edges([cvx_contour])
        lines = self.fe.get_table_edges(contours)
        cvx_mask = self.fe.get_cvx_mask(contours)
        lines = self.fe.extend_lines(lines)
        # debug_img = self.fe.draw_lines(debug_img, lines, (255, 0, 255))
        lines = self.fe.filter_lines(lines, cvx_mask)
        # print(lines)
        # all_lines = self.fe.extend_lines(lines)
        # debug_img = self.fe.draw_lines(debug_img, all_lines, (255, 0, 255))

        if lines is None or len(lines) < 4 or len(lines.shape) == 1:
            print("Not enough lines")
            return None, None, None, None, debug_img

        # Fuse the similar lines and extend them to the whole picture
        # debug_img = self.fe.draw_lines(debug_img, lines, (255, 0, 255))
        lines = self.fe.bundler.process_lines(lines)

        # print(lines)
        if debug:
            debug_img = self.fe.draw_lines(debug_img, lines, (0, 0, 255))
        # plt.imshow(debug_img)
        # plt.show()
        if lines is None or len(lines) < 4 or len(lines.shape) == 1:
            print("Not enough lines")
            return None, None, None, None, debug_img
        elif len(lines) > 4 and len(lines) < 7:
            best_score = float("inf")
            best_lines = None
            for quad in combinations(lines, 4):
                corners = self.fe.get_corners(quad)
                if len(corners) != 4:
                    continue

                # Estimate initial pose
                rvec, tvec, f, er = self.process_corners(corners)
                # print(er)
                if er < best_score:
                    best_score = er
                    best_lines = quad
            if best_lines is None:
                print("Couldn't find combination")
                return None, None, None, None, debug_img
            lines = best_lines
            corners = self.fe.get_corners(lines)
        elif len(lines) == 4:
            # Get the intersection of the lines
            corners = self.fe.get_corners(lines)
        else:
            return None, None, None, None, debug_img

        if len(corners) != 4:
            print("Not enough corners")
            return None, None, None, None, debug_img
        if debug:
            debug_img = self.fe.draw_corners(debug_img, corners, verbose=True)

        # Estimate initial pose
        rvec, tvec, f, er = self.process_corners(corners)
        if er > self.er_thres:
            return None, None, None, er, debug_img
        # Sanity check for the table position
        if tvec[2] > 20 or tvec[2] < 1:
            return None, None, None, er, debug_img

        self.update_K(f)
        rvec = self.correct_rvec(rvec)

        # Refine pose with mid-line if possible
        # in_lines = self.fe.get_lines_within_mask(img, mask)
        # if len(in_lines) == 0:
        #     if debug:
        #         proj_corners = self.project_table_corners(rvec, tvec)
        #         debug_img = self.fe.draw_corners(
        #             debug_img, proj_corners, (0, 0, 255), verbose=True
        #         )
        #         debug_img = self.draw_coordinate_frame(debug_img, rvec, tvec)

        #     return rvec, tvec, f, er, debug_img
        # in_lines = self.fe.bundler.process_lines(in_lines)
        # in_lines = self.fe.extend_lines(in_lines)
        # best_mid_line = self.fe.find_best_mid_line(in_lines, rvec, tvec, self.K)
        # # debug_img = self.fe.draw_lines(debug_img, in_lines)

        # if best_mid_line is not None:
        #     if debug:
        #         debug_img = self.fe.draw_lines(debug_img, [best_mid_line])
        #     mid_points = self.fe.get_mid_points(best_mid_line, lines)
        #     if mid_points is not None:
        #         if debug:
        #             debug_img = self.fe.draw_mid_points(
        #                 debug_img, mid_points, verbose=True
        #             )

        #         all_features = self.fe.sort_all_features(corners, mid_points)
        #         rvec, tvec, f, er = self.get_table_pose_extra(all_features)
        #         rvec = self.correct_rvec(rvec)

        if debug:
            proj_corners = self.project_table_corners(rvec, tvec)
            debug_img = self.fe.draw_corners(
                debug_img, proj_corners, (0, 0, 255), verbose=True
            )
            debug_img = self.draw_coordinate_frame(debug_img, rvec, tvec)

        # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        return rvec, tvec, f, er, debug_img

    def correct_rvec(self, rvec):
        """
        Ensures that the Z vector of the table frame is always upward,
        compensating for any ambiguity due to the table's symmetry.

        Parameters:
            rvec (numpy.ndarray): Input rotation vector representing the orientation.

        Returns:
            numpy.ndarray: Corrected rotation vector with Z axis pointing upwards.
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Check if the Z axis points upward by projecting Z axis (0, 0, 1) through the rotation matrix
        z_direction = np.dot(rotation_matrix, np.array([0, 0, 1]))[1]

        # If Z points downward, flip the Y and Z axes
        if z_direction > 0:
            correction_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            corrected_rotation_matrix = np.dot(rotation_matrix, correction_matrix)
            rvec, _ = cv2.Rodrigues(corrected_rotation_matrix)

        return rvec

    def update_K(self, f):
        self.K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]])

    def project_table_corners(self, rvec, tvec, K=None, dist=None):
        if K is None:
            K = self.K
        if dist is None:
            dist = self.dist
        projected_corners = []
        for point in self.table_corners_3d:
            projection, _ = cv2.projectPoints(point, rvec, tvec, K, dist)
            projected_corners.append(
                np.array([projection[0][0][0], projection[0][0][1]], dtype="double")
            )

        return projected_corners

    # draw a coordinate frame in an img
    def draw_coordinate_frame(self, img, rvec, tvec, K=None, dist=None):
        if K is None:
            K = self.K
        if dist is None:
            dist = self.dist
        orig = np.array([[0], [0], [0.0001]], dtype=float)
        x_vec = np.array([[1], [0], [0]], dtype=float) * 0.7625
        y_vec = np.array([[0], [1], [0]], dtype=float) * 1.37
        z_vec = np.array([[0], [0], [1]], dtype=float) * 1

        orig, _ = cv2.projectPoints(orig, rvec, tvec, K, dist)
        x_vec, _ = cv2.projectPoints(x_vec, rvec, tvec, K, dist)
        y_vec, _ = cv2.projectPoints(y_vec, rvec, tvec, K, dist)
        z_vec, _ = cv2.projectPoints(z_vec, rvec, tvec, K, dist)

        cv2.line(
            img,
            (int(orig[0][0][0]), int(orig[0][0][1])),
            (int(x_vec[0][0][0]), int(x_vec[0][0][1])),
            (0, 0, 255),
            2,
        )
        cv2.line(
            img,
            (int(orig[0][0][0]), int(orig[0][0][1])),
            (int(y_vec[0][0][0]), int(y_vec[0][0][1])),
            (0, 255, 0),
            2,
        )
        cv2.line(
            img,
            (int(orig[0][0][0]), int(orig[0][0][1])),
            (int(z_vec[0][0][0]), int(z_vec[0][0][1])),
            (255, 0, 0),
            2,
        )
        return img
