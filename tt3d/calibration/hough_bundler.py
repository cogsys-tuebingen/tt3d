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
        x1, y1, x2, y2 = line
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        return math.degrees(angle_rad)

    def point_to_segment_distance(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line
        line_mag = np.hypot(x2 - x1, y2 - y1)
        if line_mag == 0:
            return np.hypot(px - x1, py - y1)
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag**2)
        if u < 0:
            closest = (x1, y1)
        elif u > 1:
            closest = (x2, y2)
        else:
            closest = (x1 + u * (x2 - x1), y1 + u * (y2 - y1))
        return np.hypot(px - closest[0], py - closest[1])

    @staticmethod
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def do_lines_intersect(self, line1, line2):
        A = line1[:2]
        B = line1[2:]
        C = line2[:2]
        D = line2[2:]
        return (self.ccw(A, C, D) != self.ccw(B, C, D)) and (
            self.ccw(A, B, C) != self.ccw(A, B, D)
        )

    def get_distance(self, line1, line2):
        if self.do_lines_intersect(line1, line2):
            return 0.0
        return min(
            [
                self.point_to_segment_distance(line1[:2], line2),
                self.point_to_segment_distance(line1[2:], line2),
                self.point_to_segment_distance(line2[:2], line1),
                self.point_to_segment_distance(line2[2:], line1),
            ]
        )

    def check_is_line_different(self, line, groups, min_dist, min_angle):
        for group in groups:
            for ref_line in group:
                if self.get_distance(line, ref_line) < min_dist:
                    angle1 = self.get_orientation(line)
                    angle2 = self.get_orientation(ref_line)
                    delta_angle = abs(angle1 - angle2)
                    delta_angle = min(delta_angle, 180 - delta_angle)  # normalize
                    if delta_angle < min_angle:
                        group.append(line)
                        return False
        return True

    def merge_lines_into_groups(self, lines):
        if len(lines) == 0:
            return []
        groups = [[lines[0]]]
        for line in lines[1:]:
            if self.check_is_line_different(
                line, groups, self.min_distance, self.min_angle
            ):
                groups.append([line])
        return groups

    def merge_line_segments(self, lines):
        # Flip line direction if needed for consistency
        base_angle = self.get_orientation(lines[0])
        for i, line in enumerate(lines):
            angle = self.get_orientation(line)
            if abs(angle - base_angle) > 90:
                lines[i] = [line[2], line[3], line[0], line[1]]
        # Average endpoints
        x1 = np.mean([line[0] for line in lines])
        y1 = np.mean([line[1] for line in lines])
        x2 = np.mean([line[2] for line in lines])
        y2 = np.mean([line[3] for line in lines])
        return [x1, y1, x2, y2]

    def process_lines(self, lines):
        if len(lines) == 0:
            return np.empty((0, 4), dtype=float)
        lines = np.asarray(lines)
        groups = self.merge_lines_into_groups(lines)
        merged_lines = [self.merge_line_segments(group) for group in groups]
        return np.array(merged_lines)
