"""
Geometry utilities
"""
import numpy as np
import cv2


def intersect_ray_plane(rvec, tvec, img_point, K, offset=np.array([0, 0, 0])):
    """
    Calculate the intersection of a ray from the camera through an image point with a plane.

    Args:
        rvec (array): Rotation vector (3,).
        tvec (array): Translation vector (3,).
        img_point (array): Image point (2,) [u, v].
        K (array): Camera intrinsic matrix (3x3).
        offset (array): Offset vector for the plane position in camera space (3,).

    Returns:
        array: Intersection point in 3D space (3,).
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Plane properties
    plane_normal = R[:, 2]  # Plane normal in camera space
    plane_point = tvec + R @ offset  # A point on the plane in 3D space (with offset)

    # Compute ray direction in camera space
    img_point_h = np.array([*img_point, 1.0])  # Homogeneous coordinates
    ray_dir = np.linalg.inv(K) @ img_point_h  # Ray in camera space
    ray_dir /= np.linalg.norm(ray_dir)  # Normalize the ray direction

    # Compute intersection of the ray with the plane
    denom = np.dot(plane_normal, ray_dir)
    if abs(denom) < 1e-6:
        raise ValueError("Ray is parallel to the plane and does not intersect.")

    d = np.dot(plane_normal, plane_point) / denom
    intersection = d * ray_dir

    return intersection


def get_transform(rvec, tvec):
    """
    Compute the 4x4 transformation matrix from a rotation vector and translation vector.

    Args:
        rvec (array-like): Rotation vector (3,).
        tvec (array-like): Translation vector (3,).

    Returns:
        np.ndarray: The 4x4 transformation matrix.
    """
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(np.array(rvec))

    # Create the 4x4 transformation matrix
    transform = np.eye(4)  # Start with the identity matrix
    transform[:3, :3] = R  # Set the rotation part
    transform[:3, 3] = tvec  # Set the translation part

    return transform
