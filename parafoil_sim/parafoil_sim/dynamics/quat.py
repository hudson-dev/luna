"""Quaternion utilities. Convention: scalar-first [w, x, y, z], q rotates
body-frame vectors into the NED inertial frame: v_ned = R(q) @ v_body."""
from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that v_ned = R @ v_body."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """qdot = 0.5 * q ⊗ [0, omega_body]."""
    w, x, y, z = q
    p, r, s = omega_body
    return 0.5 * np.array([
        -x * p - y * r - z * s,
        w * p + y * s - z * r,
        w * r + z * p - x * s,
        w * s + x * r - y * p,
    ])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler angles to quaternion."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion to (roll, pitch, yaw), ZYX convention."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(s)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))
