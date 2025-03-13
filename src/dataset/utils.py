import numpy as np
from .config import *


def translate_landmarks(landmarks, ref_point_idx):
    # Get the reference point coordinates
    reference_point = landmarks[ref_point_idx]

    # Translate all landmarks so that the reference point is at the origin
    translated_landmarks = landmarks - reference_point

    return translated_landmarks


def scale_landmarks(landmarks, max_dist=1):
    # Compute all pairwise distances
    distances = np.linalg.norm(
        landmarks[:, np.newaxis] - landmarks[np.newaxis, :], axis=2
    )

    # Find the maximum distance in the current set of landmarks
    current_max_distance = distances.max()

    # Calculate the scale factor
    scale_factor = max_dist / current_max_distance

    # Scale the landmarks
    scaled_landmarks = landmarks * scale_factor

    return scaled_landmarks


def preprocess_landmarks(landmarks, ref_point_idx=0, max_dist=1):
    translated_landmarks = translate_landmarks(landmarks, ref_point_idx)
    scaled_landmarks = scale_landmarks(translated_landmarks, max_dist)
    return scaled_landmarks


def calculate_angle(p1, p2, p3):
    # Create vectors from the points
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Compute the dot product
    dot_product = np.dot(v1_norm, v2_norm)

    # Ensure the dot product is within the range [-1, 1] for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(dot_product)

    return angle_rad


def cal_all_finger_angles(landmarks):
    angles = {}
    for joint, (p1, p2, p3) in FINGER_JOINTS.items():
        angle = calculate_angle(landmarks[p1], landmarks[p2], landmarks[p3])
        angles[joint] = angle

    return angles
