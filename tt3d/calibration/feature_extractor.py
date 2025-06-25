"""
Extracting the features from the table
"""


from pathlib import Path
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from hough_bundler import HoughBundler
from numpy.linalg import norm
from utils import read_img


class FeatureExtractor:
    def __init__(self, h, w):
        # Assume no distortion from lens
        self.dist = np.zeros((4, 1))
        self.h = h
        self.w = w
        self.hough_thresh = 40
        # self.hough_thresh = 50
        self.hough_thresh_step = 5
        self.table_3dpoints = np.array(
            [
                [-0.7625, 1.37, 0],
                [0.7625, 1.37, 0],
                [0.7625, -1.37, 0],
                [-0.7625, -1.37, 0],
            ]
        )
        self.mid_line = np.array([[0, 1.37, 0], [0, -1.37, 0]])
        self.all_table_3dpoints = np.vstack((self.table_3dpoints, self.mid_line))
        self.bundler = HoughBundler(min_distance=6, min_angle=8)

    def filter_lines(self, lines, mask):
        h, w = mask.shape
        f_lines = []
        ratios = np.zeros(len(lines))

        # Get all pixel coordinates of the mask
        y_coords, x_coords = np.nonzero(mask)

        for i, (x1, y1, x2, y2) in enumerate(lines):
            # Compute vectors
            line_vec = np.array([x2 - x1, y2 - y1])  # Direction of the line
            point_vecs = np.stack(
                (x_coords - x1, y_coords - y1), axis=-1
            )  # Vectors from (x1, y1) to each pixel

            # Compute cross product (determinant of 2x2 matrix)
            cross_products = np.cross(line_vec, point_vecs)

            # Classify pixels based on the sign of the cross product
            left_pixels = cross_products > 0
            right_pixels = cross_products < 0

            # Count the number of pixels on each side
            num_left = np.sum(left_pixels)
            num_right = np.sum(right_pixels)

            # Compute ratio (avoiding division by zero)
            if num_left == 0 or num_right == 0:
                ratio = float("inf")  # Discard completely one-sided lines
            else:
                ratio = min(num_left / num_right, num_right / num_left)
                # print(ratio)

            ratios[i] = ratio
            if ratio < 0.15 or ratio > 0.85:
                f_lines.append([x1, y1, x2, y2])

        # Sort lines by best left/right balance and select top 4
        # sorted_indices = np.argsort(ratios)
        # f_lines = np.array(lines)[sorted_indices[:4]]
        return np.array(f_lines)

    def find_best_trapezoid(self, lines):
        best_combination = None
        best_score = float("inf")

        for quad in combinations(lines, 4):
            slopes = [
                (y2 - y1) / (x2 - x1 + 1e-9) for x1, y1, x2, y2 in quad
            ]  # Avoid division by zero
            parallel_pairs = [abs(slopes[0] - slopes[1]), abs(slopes[2] - slopes[3])]
            parallel_score = sum(parallel_pairs)

            if parallel_score < best_score:
                best_score = parallel_score
                best_combination = quad

        return np.array(best_combination) if best_combination else None

    def find_best_mid_line(self, in_lines, rvec, tvec, K):
        mid_line = self.project_mid_line(rvec, tvec, K=K)
        best_mid_line, best_cost = None, 10

        for line in in_lines:
            cost = min(
                self.angle_between_lines(mid_line, line),
                180 - self.angle_between_lines(mid_line, line),
            )
            if cost < best_cost:
                best_mid_line, best_cost = line, cost

        return best_mid_line

    def sort_all_features(self, corners, mid_points):
        # Get the distance from the predicted mid point positon to the observed to order the features
        # For the PnP
        mid_err = np.array(
            [
                np.linalg.norm(mid_points[0] - ((corners[0] + corners[1]) / 2))
                + np.linalg.norm(mid_points[1] - ((corners[2] + corners[3]) / 2)),
                np.linalg.norm(mid_points[0] - ((corners[2] + corners[3]) / 2))
                + np.linalg.norm(mid_points[1] - ((corners[0] + corners[1]) / 2)),
                np.linalg.norm(mid_points[0] - ((corners[0] + corners[3]) / 2))
                + np.linalg.norm(mid_points[1] - ((corners[1] + corners[2]) / 2)),
                np.linalg.norm(mid_points[0] - ((corners[1] + corners[2]) / 2))
                + np.linalg.norm(mid_points[1] - ((corners[0] + corners[3]) / 2)),
            ]
        )
        if np.argmin(mid_err) == 0:
            corner_order = np.array([0, 1, 2, 3])
            mid_point_order = np.array([0, 1])
        elif np.argmin(mid_err) == 1:
            corner_order = np.array([0, 1, 2, 3])
            mid_point_order = np.array([1, 0])
        elif np.argmin(mid_err) == 2:
            corner_order = np.array([2, 1, 0, 3])
            mid_point_order = np.array([1, 0])
        elif np.argmin(mid_err) == 3:
            corner_order = np.array([2, 1, 0, 3])
            mid_point_order = np.array([0, 1])
        all_features = np.concatenate(
            (corners[corner_order], mid_points[mid_point_order])
        )
        return all_features

    def get_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def is_contour_inside(self, contour1, contour2):
        """
        Check if contour1 is inside contour2.

        Parameters:
        contour1 (ndarray): First contour (inner contour) - array of points.
        contour2 (ndarray): Second contour (outer contour) - array of points.

        Returns:
        bool: True if contour1 is completely inside contour2, False otherwise.
        """
        for point in contour1:
            point = (float(point[0][0]), float(point[0][1]))
            result = cv2.pointPolygonTest(contour2, point, False)
            if result <= 0:
                return False
        return True

    def filter_contours(self, contours, area_ratio_thresh=1e-3):
        """
        Selects only the contours with a large area and only the 2 biggest
        :param contours [TODO:type]: [TODO:description]
        """
        ratio = 0.2
        center_x_min = self.w * ratio
        center_x_max = self.w * (1 - ratio)
        center_y_min = self.h * ratio
        center_y_max = self.h * (1 - ratio)

        # Filter contours by their area
        area_threshold = area_ratio_thresh * self.w * self.h

        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > area_threshold:
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue  # skip degenerate contours
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                if (
                    center_x_min <= cx <= center_x_max
                    and center_y_min <= cy <= center_y_max
                ):
                    filtered_contours.append(contour)
        # print(filtered_contours)
        # Sort the contours by area in ascending order
        sorted_contours = sorted(filtered_contours, key=cv2.contourArea)

        if len(sorted_contours) > 1:
            if self.is_contour_inside(sorted_contours[-2], sorted_contours[-1]):
                return sorted_contours[-1:]

        return sorted_contours[-3:]

    def shrink_contour(self, contour, offset):
        """
        Generate a new contour that is offset inside the given contour.

        Parameters:
        contour (ndarray): The original contour.
        offset (int): The number of pixels to shrink the contour.

        Returns:
        ndarray: The new, shrunken contour.
        """
        # Create a mask from the original contour
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h + 2 * offset, w + 2 * offset), dtype=np.uint8)
        contour_shifted = contour - [x - offset, y - offset]
        cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)

        # Apply erosion to shrink the contour
        kernel = np.ones((offset * 2, offset * 2), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        # Find the new, shrunken contour
        new_contours, _ = cv2.findContours(
            eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # There might be multiple contours, so we take the largest one
        if new_contours:
            new_contour = max(new_contours, key=cv2.contourArea)
            new_contour += [x - offset, y - offset]
        else:
            new_contour = None

        return new_contour

    def get_table_edges(self, contours, mask=None) -> np.ndarray:
        """
        Extract table edges using Hough Line Transform and format lines as (x1, y1, x2, y1).

        Args:
            contours (list): List of contours representing the table.
            mask (np.ndarray, optional): Binary mask to refine edges.

        Returns:
            np.ndarray: Array of lines with shape (N, 4), where each line is [x1, y1, x2, y1].
        """

        # Create masks
        contour_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        hull_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # Draw all contours (as edges)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)

        # Compute and draw convex hull of all points
        all_points = np.vstack(contours)
        hull = cv2.convexHull(all_points)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

        # Shrink the convex hull slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Tunable
        shrunken_hull = cv2.erode(hull_mask, kernel)

        # Get contour edges outside the shrunken convex hull
        outside_edges = cv2.bitwise_and(contour_mask, cv2.bitwise_not(shrunken_hull))
        # cv2.imshow("edges", outside_edges)
        # cv2.waitKey(1)

        # Estimate an initial Hough threshold based on contour area
        initial_thresh = int(0.4 * np.sqrt(cv2.contourArea(contours[0])))
        min_line_length = int(self.h * 0.01)
        max_line_gap = int(self.h * 0.25)

        # Initialize variables
        threshold = initial_thresh
        lines = []
        attempts = 0
        max_attempts = 10

        while len(lines) < 8 and attempts < max_attempts:
            detected = cv2.HoughLinesP(
                outside_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap,
            )

            if detected is not None and len(detected) > 0:
                lines = np.squeeze(detected)
                if lines.ndim == 1:
                    lines = lines[np.newaxis, :]  # ensure 2D shape
            else:
                lines = []

            threshold = max(
                1, threshold - self.hough_thresh_step
            )  # avoid zero threshold
            attempts += 1

        # Return as clean numpy array
        if len(lines) == 0:
            return np.empty((0, 4), dtype=int)

        return np.array(lines, dtype=int)

    def clean_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cleaned_mask

    def extend_lines(self, lines):
        extended_lines = []
        img_rect = (0, 0, self.w, self.h)  # x, y, width, height

        for line in lines:
            x1, y1, x2, y2 = line

            # Extend the line far beyond the image borders
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue  # Skip degenerate line

            # Extend line in both directions
            scale = max(self.w, self.h) * 2  # extend enough to go past any edge
            x1_ext = int(x1 - scale * dx)
            y1_ext = int(y1 - scale * dy)
            x2_ext = int(x2 + scale * dx)
            y2_ext = int(y2 + scale * dy)

            # Clip the extended line to image bounds
            success, pt1, pt2 = cv2.clipLine(
                img_rect, (x1_ext, y1_ext), (x2_ext, y2_ext)
            )
            if success:
                extended_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        return np.array(extended_lines, dtype=int)

    def get_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Calculate the determinants
        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Check if the lines are parallel
        if det == 0:
            return None  # Lines are parallel, no intersection

        # Calculate the intersection point coordinates
        intersection_x = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / det
        intersection_y = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / det

        return intersection_x, intersection_y

    def sort_corners(self, corners):
        """
        Sorts 4 corners in clockwise order using the mean of all corners as a reference.

        Args:
            corners (np.ndarray): An array of shape (4, 2) containing the (x, y) coordinates.

        Returns:
            np.ndarray: The corners sorted in clockwise order.
        """
        # Compute the mean point (center of all corners)
        center = np.mean(corners, axis=0)

        # Sort corners based on their angle relative to the center
        sorted_corners = sorted(
            corners,
            key=lambda corner: np.arctan2(corner[1] - center[1], corner[0] - center[0]),
        )

        return np.array(sorted_corners)

    def find_most_clustered_points(self, points):
        """
        Find the four points in an array that are the most clustered together.

        Args:
            points (np.ndarray): Array of points with shape (N, 2), where each row is [x, y].

        Returns:
            np.ndarray: Array of the four most clustered points, shape (4, 2).
        """
        if points.shape[0] < 4:
            raise ValueError("Input must contain at least 4 points.")

        # Generate all combinations of 4 points
        point_combinations = list(combinations(points, 4))

        # Calculate the total pairwise distance for each combination
        min_distance = float("inf")
        best_cluster = None

        for cluster in point_combinations:
            cluster = np.array(cluster)
            # Calculate pairwise distances
            dist_matrix = np.linalg.norm(cluster[:, None] - cluster[None, :], axis=-1)
            total_distance = np.sum(dist_matrix)

            if total_distance < min_distance:
                min_distance = total_distance
                best_cluster = cluster

        return best_cluster

    def get_corners(self, lines):
        corners = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                corner = self.get_intersection(lines[i], lines[j])
                if corner is None:
                    continue
                # Check if intersection is in the image
                if corner[0] < 0 or self.w < corner[0]:
                    continue
                if corner[1] < 0 or self.h < corner[1]:
                    continue

                # print(corner)
                corners.append(corner)
        corners = np.array(corners)
        # Keep the 4 most clustered ones
        if len(corners) > 4:
            corners = self.find_most_clustered_points(corners)

        if len(corners) == 4:
            sorted_corners = self.sort_corners(corners)
            # # Sort the points clockwise
            # sorted_corners = np.copy(corners)
            # sorted_corners[2, :] = corners[3, :]
            # sorted_corners[3, :] = corners[2, :]
            return sorted_corners
        # print(corners)

        return corners

    def get_mid_line(self, img, roi_pt):
        pad = int(50)
        roi_pt = roi_pt.astype(np.uint8)
        gray = cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_BGR2GRAY,
        )
        # gray = cv2.cvtColor(
        #     img[roi_pt[0] - pad : roi_pt[0] + pad, roi_pt[1] - pad : roi_pt[1] + pad],
        #     cv2.COLOR_BGR2GRAY,
        # )
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        # cv2.imshow("dst", img)
        # if cv2.waitKey(0) & 0xFF == 27:
        #     cv2.destroyAllWindows()
        # Threshold for an optimal value, it may vary depending on the image.
        P = np.argwhere(dst > 0.01 * dst.max())
        P[:, 0] += roi_pt[0] - pad
        P[:, 1] += roi_pt[1] - pad
        # print(P)
        return P

    def draw_lines(self, img, lines, color=(0, 255, 0)):
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
        return img

    def get_cvx_contour(self, contours):
        # Combine all points from the contours into a single array
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])

        # Calculate the convex hull of the combined points
        combined_hull = cv2.convexHull(all_points)

        return combined_hull

    def get_cvx_mask(self, contours):
        img_shape = (self.h, self.w)
        # Combine all points from the contours into a single array
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])

        # Calculate the convex hull of the combined points
        combined_hull = cv2.convexHull(all_points)

        # Create a blank binary mask with the same size as the image
        binary_mask = np.zeros(img_shape, dtype=np.uint8)

        # Fill the convex hull area with white (255)
        cv2.fillPoly(binary_mask, [combined_hull], 255)

        return binary_mask

    def draw_contours(self, img, contours, color=(255, 0, 0), mask=None):
        out_img = cv2.drawContours(img, contours, -1, color, 1)
        if mask is not None:
            shrinked_cvx_hull = self.shrink_contour(mask, 7)
            out_img = cv2.drawContours(out_img, [shrinked_cvx_hull], -1, 0, -1)
        return out_img

    def draw_mid_points(self, img, corners, color=(0, 255, 255), verbose=False):
        for i in range(len(corners)):
            if corners[i] is not None:
                # print(corner)
                loc = np.array([corners[i][0], corners[i][1]], dtype=int)
                cv2.circle(
                    img,
                    loc,
                    3,
                    color,
                    -1,
                )
                if verbose:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    thickness = 2
                    cv2.putText(
                        img,
                        str(i + 4),
                        loc + 15,
                        font,
                        fontScale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
        return img

    def draw_corners(self, img, corners, color=(0, 255, 255), verbose=False):
        for i in range(len(corners)):
            if corners[i] is not None:
                # print(corner)
                loc = np.array([corners[i][0], corners[i][1]], dtype=int)
                cv2.circle(
                    img,
                    loc,
                    3,
                    color,
                    -1,
                )
                if verbose:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    thickness = 2
                    cv2.putText(
                        img,
                        str(i),
                        loc + 15,
                        font,
                        fontScale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
        return img

    def find_mask_center(self, mask):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found in the mask")

        # Assuming there's only one contour, you can calculate its center
        w_tot = 0
        x, y = 0, 0
        for contour in contours:
            M = cv2.moments(contour)
            area = cv2.contourArea(contour)
            w_tot += area

            # Calculate the center coordinates
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            x += area * center_x
            y += area * center_y
        x = x / w_tot
        y = y / w_tot

        return x, y

    def get_line_mask_ratio(self, mask, line):
        """
        Calculates the number of pixels within the mask on both sides of a given line using the cross product.
        Optimized version using vectorized operations.

        Parameters:
            mask (numpy.ndarray): Binary mask (1 for mask region, 0 for background).
            line (tuple): Line endpoints as ((x1, y1), (x2, y2)).

        Returns:
            float: The ratio of pixels on one side of the line to the total number of pixels in the mask.
        """
        # Extract line endpoints
        x1, y1, x2, y2 = line

        # Vector from (x1, y1) to (x2, y2)
        line_vector = np.array([x2 - x1, y2 - y1])

        # Create a grid of pixel coordinates
        y_indices, x_indices = np.where(
            mask != 0
        )  # Get coordinates of non-zero pixels (i.e., mask region)

        # Vector from (x1, y1) to each pixel (x, y)
        points_vectors = np.column_stack((x_indices - x1, y_indices - y1))

        # Cross product: line_vector x point_vector
        cross_products = (
            line_vector[0] * points_vectors[:, 1]
            - line_vector[1] * points_vectors[:, 0]
        )

        # Count pixels on each side of the line
        pixels_side_1 = np.sum(cross_products > 0)  # Pixels on one side of the line
        pixels_side_2 = np.sum(
            cross_products < 0
        )  # Pixels on the other side of the line

        # Return the ratio of pixels on side 1 to the total mask area
        total_pixels = pixels_side_1 + pixels_side_2
        if total_pixels == 0:
            return 0  # Avoid division by zero if no pixels are on either side

        return pixels_side_1 / total_pixels

    def get_lines_within_mask(
        self,
        img,
        mask,
        edge_threshold1=30,
        edge_threshold2=50,
        hough_threshold=60,
        min_line_length=100,
        max_line_gap=50,
    ):
        """
        Detects lines within the region defined by the mask in the input image.

        Parameters:
            img (numpy.ndarray): The input image (in color or grayscale).
            mask (numpy.ndarray): A binary mask defining the region of interest (same width and height as img).
            edge_threshold1 (int): First threshold for the hysteresis procedure in Canny edge detection.
            edge_threshold2 (int): Second threshold for the hysteresis procedure in Canny edge detection.
            hough_threshold (int): Threshold for Hough Line Transform.
            min_line_length (int): Minimum length of a line to be detected.
            max_line_gap (int): Maximum allowed gap between line segments to treat them as a single line.

        Returns:
            list: A list of lines detected, where each line is represented as [(x1, y1), (x2, y2)].
        """
        # Ensure the image is in grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Apply the mask to the grayscale image
        masked_img = cv2.bitwise_and(gray, gray, mask=mask)

        # Perform edge detection on the masked region
        edges = cv2.Canny(masked_img, edge_threshold1, edge_threshold2)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        # Format detected lines as [(x1, y1), (x2, y2)]
        # inside_mask = self.erode_mask(mask, kernel_size=3, iterations=1)
        detected_lines = []
        if lines is not None:
            for line in lines:
                # print(line[0])
                x1, y1, x2, y2 = line[0]
                ratio = self.get_line_mask_ratio(mask, line[0])
                # print(ratio)
                if ratio > 0.3 and ratio < 0.7:
                    detected_lines.append([x1, y1, x2, y2])

        # edges = self.draw_lines(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), detected_lines)
        # bot = 829
        # top = 246
        # right = 1450
        # left = 465
        # cv2.imshow("show", edges[top:bot, left:right])
        # cv2.waitKey(0)
        return np.array(detected_lines)

    def angle_between_lines(self, line1, line2):
        """
        Calculate the angle in degrees between two lines given by their endpoints.

        Parameters:
            line1 (tuple): Line 1 endpoints as ((x1, y1), (x2, y2))
            line2 (tuple): Line 2 endpoints as ((x3, y3), (x4, y4))

        Returns:
            float: The angle in degrees between the two lines.
        """
        # Unpack points
        x1, y1, x2, y2 = line1.flatten()
        x3, y3, x4, y4 = line2.flatten()

        # Vector for line 1
        v1 = np.array([x2 - x1, y2 - y1])
        # Vector for line 2
        v2 = np.array([x4 - x3, y4 - y3])

        # Calculate the dot product and magnitudes of the vectors
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (mag_v1 * mag_v2)

        # Numerical precision safeguard (cosine values should be between -1 and 1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians and convert to degrees
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def get_mid_points(self, mid_line, lines):
        mid_points = []
        angles = []
        for line in lines:
            angles.append(self.angle_between_lines(mid_line, line))
        angles = np.array(angles)
        angles[angles > 90] = angles[angles > 90] - 180
        # print("angles", angles)
        idxs = np.argsort(np.abs(angles))[2:]
        for idx in idxs:
            mid_point = self.get_intersection(mid_line, lines[idx])
            if mid_point is None:
                return None
            mid_points.append(mid_point)
        # print("Midpoint", mid_points)
        return np.array(mid_points)

    def project_mid_line(self, rvec, tvec, K=None, dist=None):
        if K is None:
            K = self.K
        if dist is None:
            dist = self.dist
        projected_mid = []
        for point in self.mid_line:
            projection, _ = cv2.projectPoints(point, rvec, tvec, K, dist)
            projected_mid.append(
                np.array([projection[0][0][0], projection[0][0][1]], dtype="double")
            )

        projected_mid = np.array(projected_mid)
        return projected_mid


if __name__ == "__main__":
    pass
