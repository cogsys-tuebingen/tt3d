"""
Fint the quadrilateral that best bounds the contour
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1fbd43f3827fffeb76641a9c5ab5b625eb5a75ba
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sympy


class QuadrilateralBounder:
    def __init__(self):
        pass

    def compute_intersection(self, p1, p2, q1, q2):
        """Compute the intersection of two lines defined by (p1, p2) and (q1, q2)."""
        A1, B1, C1 = p2[1] - p1[1], p1[0] - p2[0], p2[0] * p1[1] - p1[0] * p2[1]
        A2, B2, C2 = q2[1] - q1[1], q1[0] - q2[0], q2[0] * q1[1] - q1[0] * q2[1]
        det = A1 * B2 - A2 * B1
        if det == 0:
            return None  # Lines are parallel
        x = (B2 * C1 - B1 * C2) / det
        y = (A1 * C2 - A2 * C1) / det
        return np.array([x, y])

    def find_best_parallelogram(self, contour, n=4):
        """Finds the best-fitting convex quadrilateral parallelogram for a given contour."""

        hull = cv2.convexHull(contour)
        hull = np.array(hull).reshape((len(hull), 2))

        # to sympy land
        hull = [sympy.Point(*pt) for pt in hull]

        # run until we cut down to n vertices
        while len(hull) > n:
            best_candidate = None

            # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
            for edge_idx_1 in range(len(hull)):
                edge_idx_2 = (edge_idx_1 + 1) % len(hull)

                adj_idx_1 = (edge_idx_1 - 1) % len(hull)
                adj_idx_2 = (edge_idx_1 + 2) % len(hull)

                edge_pt_1 = sympy.Point(*hull[edge_idx_1])
                edge_pt_2 = sympy.Point(*hull[edge_idx_2])
                adj_pt_1 = sympy.Point(*hull[adj_idx_1])
                adj_pt_2 = sympy.Point(*hull[adj_idx_2])

                subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
                angle1 = subpoly.angles[edge_pt_1]
                angle2 = subpoly.angles[edge_pt_2]

                # we need to first make sure that the sum of the interior angles the edge
                # makes with the two adjacent edges is more than 180°
                if sympy.N(angle1 + angle2) <= sympy.pi:
                    continue

                # find the new vertex if we delete this edge
                adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
                adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
                intersect = adj_edge_1.intersection(adj_edge_2)[0]

                # the area of the triangle we'll be adding
                area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
                # should be the lowest
                if best_candidate and best_candidate[1] < area:
                    continue

                # delete the edge and add the intersection of adjacent edges to the hull
                better_hull = list(hull)
                better_hull[edge_idx_1] = intersect
                del better_hull[edge_idx_2]
                best_candidate = (better_hull, area)

            if not best_candidate:
                raise ValueError("Could not find the best fit n-gon!")

            hull = best_candidate[0]
            print(len(hull))

        # back to python land
        hull = [(int(x), int(y)) for x, y in hull]

        return np.array(hull).astype(int)

    def process_image(self, img):
        _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        # plt.imshow(thresh)
        # plt.show()
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(contours)
        contour = max(contours, key=cv2.contourArea)  # Get the largest contour
        return self.find_best_parallelogram(contour)

    def draw_parallelogram(self, img, quad):
        cv2.polylines(img, [quad], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("Best-Fit Parallelogram", img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


# Example usage
image_path = "/home/gossard/Pictures/parallelogram.png"
bounder = QuadrilateralBounder()
# bounder.draw_parallelogram()
