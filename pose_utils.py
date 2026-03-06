"""Shared pose detection and landmark utilities for posture project."""

import math
from typing import Optional

from cvzone.PoseModule import PoseDetector

# MediaPipe pose landmark IDs
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def build_detector() -> PoseDetector:
    """Create and return a PoseDetector instance."""
    return PoseDetector()


def extract_landmarks(frame, detector: PoseDetector) -> tuple:
    """
    Detect pose and extract landmarks from a frame.

    Args:
        frame: BGR image from webcam
        detector: PoseDetector instance

    Returns:
        Tuple of (annotated frame with pose drawn, lmList or None if no pose)
    """
    frame = detector.findPose(frame)
    lmList, _ = detector.findPosition(frame, draw=True)
    return frame, lmList if lmList else None


def flatten_landmarks(lmList: list) -> list[float]:
    """
    Convert landmark list to flat list [x0, y0, x1, y1, ...].

    Args:
        lmList: List of [x, y, z] per landmark (index = landmark id) from findPosition

    Returns:
        Flat list of x,y coordinates
    """
    row: list[float] = []
    for lm in lmList:
        if len(lm) >= 2:
            row.extend([float(lm[0]), float(lm[1])])
    return row


def _angle_deg(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    """Angle at p2 between vectors p2->p1 and p2->p3, in degrees [-180, 180]."""
    a = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    b = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    return math.degrees(a - b)


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def _slope_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Angle of line p1->p2 relative to horizontal, in degrees [-90, 90]."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def extract_posture_features(lmList: list) -> Optional[list[float]]:
    """
    Extract camera-invariant posture features from landmarks.

    Uses angles and ratios instead of raw pixel coordinates for robustness
    across different camera positions and distances.

    Args:
        lmList: List of [x, y, z] per landmark (index = landmark id) from findPosition

    Returns:
        Fixed-length feature vector, or None if required landmarks missing
    """
    lm_dict = {i: (float(lm[0]), float(lm[1])) for i, lm in enumerate(lmList) if len(lm) >= 2}
    required = {
        NOSE, LEFT_EAR, RIGHT_EAR,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
    }
    if not required.issubset(lm_dict):
        return None

    ns = lm_dict[NOSE]
    le = lm_dict[LEFT_EAR]
    re = lm_dict[RIGHT_EAR]
    ls = lm_dict[LEFT_SHOULDER]
    rs = lm_dict[RIGHT_SHOULDER]
    lh = lm_dict[LEFT_HIP]
    rh = lm_dict[RIGHT_HIP]
    lk = lm_dict[LEFT_KNEE]
    rk = lm_dict[RIGHT_KNEE]

    shoulder_width = _dist(ls, rs)
    hip_width = _dist(lh, rh)
    if shoulder_width < 1:
        shoulder_width = 1

    # Angles: shoulder-hip-knee (trunk vs thigh)
    angle_l = _angle_deg(ls, lh, lk)
    angle_r = _angle_deg(rs, rh, rk)

    # Spinal lean: ear-shoulder-hip
    spine_l = _angle_deg(le, ls, lh)
    spine_r = _angle_deg(re, rs, rh)

    # Shoulder and hip tilt relative to horizontal
    shoulder_tilt = _slope_deg(ls, rs)
    hip_tilt = _slope_deg(lh, rh)

    # Midpoints
    shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    head_mid = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)

    # Neck height (head to shoulder midpoint) / shoulder width
    neck_height = _dist(head_mid, shoulder_mid)
    neck_height_ratio = neck_height / shoulder_width

    # Head x offset from shoulder midpoint, normalized by shoulder width
    head_offset_x = (head_mid[0] - shoulder_mid[0]) / shoulder_width
    head_offset_y = (head_mid[1] - shoulder_mid[1]) / shoulder_width

    # Hip-to-shoulder vertical span / shoulder width (trunk length ratio)
    trunk_height = abs((lh[1] + rh[1]) / 2 - (ls[1] + rs[1]) / 2)
    trunk_ratio = trunk_height / shoulder_width

    return [
        angle_l,
        angle_r,
        spine_l,
        spine_r,
        shoulder_tilt,
        hip_tilt,
        neck_height_ratio,
        head_offset_x,
        head_offset_y,
        trunk_ratio,
        hip_width / shoulder_width,
    ]


FEATURE_COLUMNS = [
    "angle_shoulder_hip_knee_L",
    "angle_shoulder_hip_knee_R",
    "spine_angle_L",
    "spine_angle_R",
    "shoulder_tilt",
    "hip_tilt",
    "neck_height_ratio",
    "head_offset_x",
    "head_offset_y",
    "trunk_ratio",
    "hip_shoulder_width_ratio",
]
