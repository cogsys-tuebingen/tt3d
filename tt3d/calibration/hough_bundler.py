"""
Merges the detected lines together
From : https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
"""

import math

import numpy as np


class HoughBundler:
    def __init__(self, min_distance=6, min_angle=5):
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self, line):
        orientation = math.atan2((line[3] - line[1]), (line[2] - line[0]))
        return math.degrees(orientation)

    def check_is_line_different(
        self, line_1, groups, min_distance_to_merge, min_angle_to_merge
    ):
        for group in groups:
            for line_2 in group:
                # print()
                # print(line_1)
                # print(line_2)
                # print(self.get_distance(line_2, line_1))
                # print()
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    # print(orientation_1)
                    # print(orientation_2)
                    # print(orientation_1 - orientation_2)
                    delta_orientation = abs(orientation_1 - orientation_2)
                    if delta_orientation < -90:
                        delta_orientation += 180
                    elif delta_orientation > 90:
                        delta_orientation -= 180
                    # print(delta_orientation)
                    if delta_orientation < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def point_to_segment_distance(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line
        line_mag = np.hypot(x2 - x1, y2 - y1)
        if line_mag == 0:
            return np.hypot(px - x1, py - y1)
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag**2)
        if u < 0:
            closest_x, closest_y = x1, y1
        elif u > 0:
            closest_x, closest_y = x2, y2
        else:
            closest_x = x1 + u * (x2 - x1)
            closest_y = y1 + u * (y2 - y1)
        return np.hypot(px - closest_x, py - closest_y)

    def get_distance(self, a_line, b_line):
        distances = [
            self.point_to_segment_distance(a_line[:2], b_line),
            self.point_to_segment_distance(a_line[2:], b_line),
            self.point_to_segment_distance(b_line[:2], a_line),
            self.point_to_segment_distance(b_line[:2], a_line),
        ]

        return min(distances)
        # print(np.min(np.array(distances)))
        # return np.min(np.array(distances))

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # print(lines[0])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(
                line_new, groups, self.min_distance, self.min_angle
            ):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
        for i, line in enumerate(lines):
            if abs(self.get_orientation(line) - orientation) > 90:
                lines[i] = np.concatenate((line[2:], line[:2]))
        x1_mean = np.mean([line[0] for line in lines])
        y1_mean = np.mean([line[1] for line in lines])
        x2_mean = np.mean([line[2] for line in lines])
        y2_mean = np.mean([line[3] for line in lines])
        return [x1_mean, y1_mean, x2_mean, y2_mean]

    # def merge_line_segments(self, lines):
    #     orientation = self.get_orientation(lines[0])

    #     if len(lines) == 1:
    #         return np.block([[lines[0][:2], lines[0][2:]]])

    #     points = []
    #     for line in lines:
    #         points.append(line[:2])
    #         points.append(line[2:])
    #     if 45 < orientation <= 90:
    #         # sort by y
    #         points = sorted(points, key=lambda point: point[1])
    #     else:
    #         # sort by x
    #         points = sorted(points, key=lambda point: point[0])

    #     return np.block([[points[0], points[-1]]])

    def process_lines(self, lines):
        groups = self.merge_lines_into_groups(lines)
        merged_lines = []
        for group in groups:
            merged_lines.append(self.merge_line_segments(group))
        merged_lines = np.asarray(merged_lines).squeeze()
        if len(merged_lines.shape) == 1:
            return merged_lines[None, :]
        return merged_lines
