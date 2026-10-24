#!/usr/bin/env python3
"""
ver13_1 autonomous driving -- green-light start, camera+LiDAR CNN steering,
rule-based orange-curve/cone speed policy.

Extracted from deeplearning_joystick_ridar_ver13_1.ipynb: every cell that is
only needed for DATA COLLECTION or TRAINING (sections [1] and 1.3/1.3b) is
left out. What remains is exactly what section 1.4c (green-light gated
multimodal inference loop) needs to run:

    0.1 constants  -- speed/steering caps, camera size, LiDAR vector shape
                      (the FusionDrivingNet torch class itself is NOT needed
                      here -- inference loads the already-exported model.onnx
                      through onnxruntime, not through torch)
    0.2 API        -- camera(), drive(), look(), read_lidar() and friends
    1.4a           -- the ver13 curve/cone speed policy (unchanged)
    1.4c           -- green-circle start gate + the CNN+policy driving loop

ipywidgets/IPython.display (notebook-only) are replaced with plain console
output; everything else is a straight copy of the notebook logic.

MODEL PATH NOTE: the notebook used DIR = "assets/deeplearning-joystick-ridar"
and MODEL_PATH = DIR/models_ver13/model.onnx. On this machine the trained
model actually lives directly at <this folder>/models_ver13/model.onnx (no
assets/deeplearning-joystick-ridar prefix) -- verified by listing the folder
before writing this file. MODEL_PATH below matches the real layout, not the
notebook's DIR constant.

Usage:
    python3 drive_ver13_1.py
    python3 drive_ver13_1.py --base-url http://localhost:8000
    python3 drive_ver13_1.py --model /path/to/model.onnx

Ctrl+C stops the car (drive(0, 0) is sent in every exit path).
"""

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests

HERE = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# 0.1 Constants (from ver13_1 cell 3 -- FusionDrivingNet/torch itself is not
# needed: inference runs through the already-exported model.onnx)
# ==========================================================================
MAX_SPEED = 1.2                    # ver6 training / model-output normalization
MAX_DRIVE_COMMAND_SPEED = 3.0      # ver8 runtime command ceiling
MAX_STEERING = 20.0                # degrees
CAMERA_W, CAMERA_H = 160, 120
CAMERA_PAN, CAMERA_TILT = 0.0, -15.0

LIDAR_STEP_DEG = 2.0
LIDAR_VECTOR_BINS = 73
LIDAR_VECTOR_FOV_DEG = 90.0
LIDAR_VECTOR_MAX_M = 4.5
OBSTACLE_DETECTION_MAX_M = 4.5
FRONT_CORRIDOR_HALF_WIDTH_M = 0.42

# ==========================================================================
# 0.2 Camera, motor and LiDAR API (from ver13_1 cell 5)
# ==========================================================================
_HTTP_LOCAL = threading.local()


def _thread_http():
    if not hasattr(_HTTP_LOCAL, "session"):
        _HTTP_LOCAL.session = requests.Session()
    return _HTTP_LOCAL.session


def camera(width=None, height=None):
    params = {}
    if width is not None:
        params["width"] = int(width)
    if height is not None:
        params["height"] = int(height)
    response = _thread_http().get(f"{BASE_URL}/camera", params=params, timeout=2)
    response.raise_for_status()
    image = cv2.imdecode(
        np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("camera returned an invalid JPEG")
    return image


def drive(speed, steering):
    speed = float(np.clip(speed, -MAX_DRIVE_COMMAND_SPEED, MAX_DRIVE_COMMAND_SPEED))
    steering = float(np.clip(steering, -MAX_STEERING, MAX_STEERING))
    _thread_http().post(
        f"{BASE_URL}/speed", json={"value": speed}, timeout=2).raise_for_status()
    _thread_http().post(
        f"{BASE_URL}/steering", json={"value": math.radians(steering)},
        timeout=2).raise_for_status()


def look(pan, tilt):
    _thread_http().post(
        f"{BASE_URL}/camera/pan", json={"value": math.radians(float(pan))},
        timeout=2).raise_for_status()
    _thread_http().post(
        f"{BASE_URL}/camera/tilt", json={"value": math.radians(float(tilt))},
        timeout=2).raise_for_status()


def _lidar_distance(value):
    if isinstance(value, dict):
        value = value.get("distance", value.get("range"))
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _normalise_angles_deg(angles):
    angles = np.asarray(angles, dtype=np.float32)
    return (angles + 180.0) % 360.0 - 180.0


def read_lidar(step_deg=LIDAR_STEP_DEG):
    """Read /lidar; 0 deg is forward and +90 deg is left."""
    response = _thread_http().get(
        f"{BASE_URL}/lidar", params={"step": float(step_deg)}, timeout=2)
    response.raise_for_status()
    data = response.json()
    metadata = data if isinstance(data, dict) else {}
    ranges = angles = None
    source_format = "unknown"

    if isinstance(data, dict) and "ranges" in data:
        payload = data["ranges"]
        if isinstance(payload, dict):
            pairs = []
            for angle, distance in payload.items():
                try:
                    pairs.append((float(angle), _lidar_distance(distance)))
                except (TypeError, ValueError):
                    continue
            pairs.sort(key=lambda pair: pair[0])
            angles = np.asarray([p[0] for p in pairs], dtype=np.float32)
            ranges = np.asarray([p[1] for p in pairs], dtype=np.float32)
            source_format = "ranges-map"
        elif isinstance(payload, (list, tuple)):
            ranges = np.asarray(
                [_lidar_distance(value) for value in payload], dtype=np.float32)
            explicit = data.get("angles", data.get("angles_deg"))
            if isinstance(explicit, (list, tuple)) and len(explicit) == len(ranges):
                angles = np.asarray(explicit, dtype=np.float32)
                if len(angles) and np.nanmax(np.abs(angles)) <= 2 * math.pi + 0.2:
                    angles = np.degrees(angles)
                source_format = "ranges+angles"
            else:
                count = len(ranges)
                angle_min = data.get("angle_min")
                increment = data.get("angle_increment")
                if angle_min is not None and increment is not None:
                    angle_min = float(angle_min)
                    increment = float(increment)
                    if abs(increment) <= 0.2 and abs(angle_min) <= 2 * math.pi + 0.2:
                        angles = np.degrees(
                            angle_min + np.arange(count, dtype=np.float32) * increment)
                        source_format = "ros-radians"
                    else:
                        angles = angle_min + np.arange(count, dtype=np.float32) * increment
                        source_format = "degree-increment"
                else:
                    actual_step = float(data.get("step", 360.0 / max(1, count)))
                    angles = np.arange(count, dtype=np.float32) * actual_step
                    source_format = "ranges-list"
    elif isinstance(data, dict):
        pairs = []
        for angle, distance in data.items():
            try:
                pairs.append((float(angle), _lidar_distance(distance)))
            except (TypeError, ValueError):
                continue
        pairs.sort(key=lambda pair: pair[0])
        if pairs:
            angles = np.asarray([p[0] for p in pairs], dtype=np.float32)
            ranges = np.asarray([p[1] for p in pairs], dtype=np.float32)
            source_format = "top-level-angle-map"
    elif isinstance(data, (list, tuple)):
        ranges = np.asarray([_lidar_distance(value) for value in data], dtype=np.float32)
        angles = np.arange(len(ranges), dtype=np.float32) * float(step_deg)
        source_format = "bare-ranges-list"

    if ranges is None or angles is None or len(ranges) == 0:
        raise RuntimeError("unrecognised or empty LiDAR response")
    if len(ranges) != len(angles):
        raise RuntimeError("LiDAR angle/range length mismatch")

    return {
        "ranges": ranges,
        "angles_deg": _normalise_angles_deg(angles),
        "range_min": float(metadata.get("range_min", 0.15)),
        "range_max": float(metadata.get("range_max", 16.0)),
        "source_format": source_format,
        "requested_step_deg": float(step_deg),
    }


def validate_lidar_scan(scan, min_points=3):
    ranges = np.asarray(scan["ranges"], dtype=np.float32)
    valid = (
        np.isfinite(ranges)
        & (ranges >= scan["range_min"])
        & (ranges <= scan["range_max"])
    )
    count = int(valid.sum())
    if count < int(min_points):
        raise RuntimeError(f"invalid LiDAR scan: {count}/{len(ranges)} valid points")
    return count


def lidar_to_vector(scan):
    vector = np.ones(LIDAR_VECTOR_BINS, dtype=np.float32)
    angles = scan["angles_deg"]
    ranges = scan["ranges"]
    valid = (
        np.isfinite(ranges)
        & (ranges >= scan["range_min"])
        & (ranges <= scan["range_max"])
        & (np.abs(angles) <= LIDAR_VECTOR_FOV_DEG)
    )
    for angle, distance in zip(angles[valid], ranges[valid]):
        fraction = (float(angle) + LIDAR_VECTOR_FOV_DEG) / (2.0 * LIDAR_VECTOR_FOV_DEG)
        index = int(round(fraction * (LIDAR_VECTOR_BINS - 1)))
        index = int(np.clip(index, 0, LIDAR_VECTOR_BINS - 1))
        normalized = min(float(distance), LIDAR_VECTOR_MAX_M) / LIDAR_VECTOR_MAX_M
        vector[index] = min(vector[index], normalized)
    padded = np.pad(vector, (1, 1), constant_values=1.0)
    return np.minimum.reduce([padded[:-2], padded[1:-1], padded[2:]])


def front_obstacle_distance(scan):
    """Nearest compact cluster inside the vehicle's forward corridor."""
    points = []
    for angle, distance in zip(scan["angles_deg"], scan["ranges"]):
        if not np.isfinite(distance):
            continue
        if not (scan["range_min"] <= distance <= min(scan["range_max"], OBSTACLE_DETECTION_MAX_M)):
            continue
        angle_rad = math.radians(float(angle))
        lateral = float(distance) * math.sin(angle_rad)
        forward = float(distance) * math.cos(angle_rad)
        if forward < 0.12 or abs(lateral) > FRONT_CORRIDOR_HALF_WIDTH_M:
            continue
        points.append((float(angle), lateral, forward, float(distance)))

    points.sort(key=lambda point: point[0])
    scan_step_deg = float(scan.get("requested_step_deg", LIDAR_STEP_DEG))
    groups = []
    current = []
    for point in points:
        if not current:
            current = [point]
            continue
        previous = current[-1]
        gap = math.hypot(point[1] - previous[1], point[2] - previous[2])
        angle_gap = point[0] - previous[0]
        threshold = 0.10 + 0.04 * min(point[3], previous[3])
        if angle_gap <= 1.8 * scan_step_deg and gap <= threshold:
            current.append(point)
        else:
            groups.append(current)
            current = [point]
    if current:
        groups.append(current)

    distances = []
    for group in groups:
        if len(group) < 2:
            continue
        xs = np.asarray([point[1] for point in group], dtype=np.float32)
        ys = np.asarray([point[2] for point in group], dtype=np.float32)
        diameter = max(float(np.ptp(xs)), float(np.ptp(ys))) + 0.05
        if diameter <= 0.65:
            distances.append(float(np.median(ys)))
    return min(distances) if distances else math.inf


# ==========================================================================
# 1.4a ver13 speed policy -- distance envelopes (from ver13_1 cell 17)
# ==========================================================================
INFERENCE_HZ = 15.0
INFERENCE_LIDAR_HZ = 10.0
INFERENCE_LIDAR_STEP_DEG = 1.0
MAX_INFERENCE_LIDAR_AGE_S = 0.30
INFERENCE_LIDAR_STALE_SPEED_CAP = 0.80

RACE_CLEAR_SPEED_MPS = 3.00
RACE_GENTLE_CURVE_SPEED_MPS = 1.80
RACE_NORMAL_CURVE_SPEED_MPS = 1.45
RACE_TIGHT_CURVE_SPEED_MPS = 1.10
RACE_OBSTACLE_SPEED_MPS = 1.20

OBSTACLE_BRAKE_START_M = 3.50
CURVE_PREVIEW_DISTANCE_M = 2.40
A_DEC = 2.5
A_LAT = 1.5
T_LATENCY = 0.30
LATERAL_SHIFT = 0.30

CURVE_GENTLE_BEND_PX = 3.5
CURVE_NORMAL_BEND_PX = 6.0
CURVE_TIGHT_BEND_PX = 10.0
CURVE_CLEAR_BEND_PX = 2.5
CURVE_CLASSIFY_FRAMES = 2
CURVE_ORANGE_HSV_LOWER = (8, 180, 120)
CURVE_ORANGE_HSV_UPPER = (25, 255, 255)
CURVE_ROI_TOP_FRACTION = 0.42
CURVE_ROI_BOTTOM_FRACTION = 0.97
CURVE_MIN_COMPONENTS = 3
CURVE_MIN_ROWS = 12
CURVE_MIN_ROW_SPAN_PX = 28.0
CURVE_RELEASE_FRAMES = 3
CURVE_LOST_LINE_RELEASE_FRAMES = 6
CLEAR_SPEED_CONFIRM_FRAMES = 3
RACE_ACCEL_RATE_MPS2 = 2.0


def maneuver_distance(v_pass=RACE_OBSTACLE_SPEED_MPS, shift=LATERAL_SHIFT, a_lat=A_LAT):
    return 2.0 * float(v_pass) * math.sqrt(float(shift) / float(a_lat))


def braking_envelope(remaining_m, target_speed, a_dec=A_DEC, latency=T_LATENCY,
                      max_speed=RACE_CLEAR_SPEED_MPS):
    remaining_m = float(remaining_m)
    target_speed = float(target_speed)
    if remaining_m <= 0.0:
        return target_speed
    a_t = float(a_dec) * float(latency)
    speed = -a_t + math.sqrt(a_t * a_t + target_speed * target_speed + 2.0 * float(a_dec) * remaining_m)
    return float(np.clip(speed, target_speed, max_speed))


def obstacle_speed(front_m):
    if not math.isfinite(front_m) or front_m > OBSTACLE_BRAKE_START_M:
        return RACE_CLEAR_SPEED_MPS
    remaining = float(front_m) - maneuver_distance()
    return braking_envelope(remaining, RACE_OBSTACLE_SPEED_MPS)


def curve_speed(distance_to_curve_m, kappa_ahead):
    corner_speed = math.sqrt(A_LAT / max(abs(float(kappa_ahead)), 1e-3))
    return braking_envelope(distance_to_curve_m, min(corner_speed, RACE_CLEAR_SPEED_MPS))


class RaceObstacleTracker:
    """Hold one missed 10 Hz scan without adding post-pass steering."""

    def __init__(self):
        self.active = False
        self.clear_hits = 0
        self.last_sample_time = None
        self.distance = math.inf

    def update(self, distance, speed_mps, sample_time=None):
        if sample_time is not None and sample_time == self.last_sample_time:
            return self.distance
        dt = (
            1.0 / INFERENCE_LIDAR_HZ
            if self.last_sample_time is None or sample_time is None
            else float(np.clip(sample_time - self.last_sample_time, 0.0, 0.25))
        )
        self.last_sample_time = sample_time
        if math.isfinite(distance) and distance <= OBSTACLE_DETECTION_MAX_M:
            self.active = True
            self.clear_hits = 0
            self.distance = float(distance)
        elif self.active:
            self.clear_hits += 1
            if self.clear_hits >= 2:
                self.active = False
                self.distance = math.inf
            else:
                self.distance = max(0.0, self.distance - abs(float(speed_mps)) * dt)
        else:
            self.distance = math.inf
        return self.distance


def orange_curve_bend(image_bgr):
    """Measure orange-line bending, not perspective or lateral slope."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, CURVE_ORANGE_HSV_LOWER, CURVE_ORANGE_HSV_UPPER)
    height, _ = orange.shape
    orange[:int(CURVE_ROI_TOP_FRACTION * height), :] = 0
    orange[int(CURVE_ROI_BOTTOM_FRACTION * height):, :] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(orange, 8)
    clean = np.zeros_like(orange)
    component_count = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= 10 and component_height >= 2:
            clean[labels == label] = 255
            component_count += 1

    row_y, row_x = [], []
    for y in range(height):
        xs = np.flatnonzero(clean[y])
        if len(xs):
            row_y.append(float(y))
            row_x.append(float(np.median(xs)))
    if component_count < CURVE_MIN_COMPONENTS or len(row_y) < CURVE_MIN_ROWS:
        return math.nan, component_count

    ys = np.asarray(row_y, dtype=np.float32)
    xs = np.asarray(row_x, dtype=np.float32)
    if float(np.ptp(ys)) < CURVE_MIN_ROW_SPAN_PX:
        return math.nan, component_count
    a, b, c = np.polyfit(ys, xs, 2)
    far_y = float(np.percentile(ys, 10.0))
    near_y = float(np.percentile(ys, 90.0))
    mid_y = 0.5 * (far_y + near_y)
    far_x = a * far_y ** 2 + b * far_y + c
    near_x = a * near_y ** 2 + b * near_y + c
    mid_x = a * mid_y ** 2 + b * mid_y + c
    bend = mid_x - 0.5 * (far_x + near_x)
    return float(bend), component_count


class CurveSeverityGate:
    """Use only orange-line bend; enter gentle before deeper classification."""

    NAMES = ("clear", "gentle", "normal", "tight")

    def __init__(self):
        self.severity = 0
        self.classify_hits = 0
        self.clear_hits = 0
        self.lost_hits = 0
        self.last_candidate = 0

    @staticmethod
    def _candidate(bend_px):
        bend_abs = abs(float(bend_px)) if np.isfinite(bend_px) else 0.0
        if bend_abs >= CURVE_TIGHT_BEND_PX:
            return 3
        if bend_abs >= CURVE_NORMAL_BEND_PX:
            return 2
        if bend_abs >= CURVE_GENTLE_BEND_PX:
            return 1
        return 0

    def update(self, bend_px):
        measured = bool(np.isfinite(bend_px))
        candidate = self._candidate(bend_px) if measured else 0

        if measured and candidate > 0:
            self.clear_hits = 0
            self.lost_hits = 0
            if self.severity == 0:
                self.severity = 1
            if candidate >= 2:
                self.classify_hits = (
                    self.classify_hits + 1 if candidate == self.last_candidate else 1)
                if self.classify_hits >= CURVE_CLASSIFY_FRAMES:
                    self.severity = max(self.severity, candidate)
                    self.classify_hits = 0
            else:
                self.classify_hits = 0
        elif measured:
            self.classify_hits = 0
            self.lost_hits = 0
            if self.severity > 0:
                if abs(float(bend_px)) <= CURVE_CLEAR_BEND_PX:
                    self.clear_hits += 1
                    if self.clear_hits >= CURVE_RELEASE_FRAMES:
                        self.severity = 0
                        self.clear_hits = 0
                else:
                    self.clear_hits = 0
        else:
            self.classify_hits = 0
            self.clear_hits = 0
            if self.severity > 0:
                self.lost_hits += 1
                if self.lost_hits >= CURVE_LOST_LINE_RELEASE_FRAMES:
                    self.severity = 0
                    self.lost_hits = 0
        self.last_candidate = candidate
        return self.severity


CURVE_TARGET_SPEEDS = {
    0: RACE_CLEAR_SPEED_MPS,
    1: RACE_GENTLE_CURVE_SPEED_MPS,
    2: RACE_NORMAL_CURVE_SPEED_MPS,
    3: RACE_TIGHT_CURVE_SPEED_MPS,
}


class CurveDistancePlanner:
    """Start a 2.4 m braking countdown when a curve first becomes visible."""

    def __init__(self):
        self.severity = 0
        self.remaining_m = CURVE_PREVIEW_DISTANCE_M
        self.last_time = None

    def update(self, now, severity, current_speed):
        now = float(now)
        dt = (
            1.0 / INFERENCE_HZ if self.last_time is None
            else float(np.clip(now - self.last_time, 0.0, 0.20))
        )
        self.last_time = now
        severity = int(severity)
        if severity <= 0:
            self.severity = 0
            self.remaining_m = CURVE_PREVIEW_DISTANCE_M
            return RACE_CLEAR_SPEED_MPS, self.remaining_m, RACE_CLEAR_SPEED_MPS

        if self.severity == 0:
            self.remaining_m = CURVE_PREVIEW_DISTANCE_M
        self.severity = max(self.severity, severity)
        self.remaining_m = max(0.0, self.remaining_m - abs(float(current_speed)) * dt)
        target = CURVE_TARGET_SPEEDS[self.severity]
        nominal_kappa = A_LAT / max(target * target, 1e-6)
        command = min(RACE_GENTLE_CURVE_SPEED_MPS, curve_speed(self.remaining_m, nominal_kappa))
        return command, self.remaining_m, target


def ver10_target_speed(curve_command, obstacle_command, extra_cap=None):
    candidates = [RACE_CLEAR_SPEED_MPS, float(curve_command), float(obstacle_command)]
    if extra_cap is not None:
        candidates.append(float(extra_cap))
    return min(candidates)


class RaceSpeedController:
    """Immediate braking, confirmed and rate-limited acceleration."""

    def __init__(self):
        self.sent_speed = 0.0
        self.clear_hits = 0
        self.last_time = None

    def update(self, now, desired_speed):
        now = float(now)
        desired_speed = float(desired_speed)
        dt = (
            1.0 / INFERENCE_HZ if self.last_time is None
            else float(np.clip(now - self.last_time, 0.0, 0.20))
        )
        self.last_time = now

        if desired_speed >= RACE_CLEAR_SPEED_MPS - 1e-6:
            self.clear_hits += 1
            if self.clear_hits < CLEAR_SPEED_CONFIRM_FRAMES:
                desired_speed = min(desired_speed, max(self.sent_speed, RACE_GENTLE_CURVE_SPEED_MPS))
        else:
            self.clear_hits = 0

        if desired_speed <= self.sent_speed:
            self.sent_speed = desired_speed
        else:
            self.sent_speed = min(desired_speed, self.sent_speed + RACE_ACCEL_RATE_MPS2 * dt)
        return self.sent_speed


# ==========================================================================
# 1.4c Green-light start gate + multimodal inference loop (from cell 21)
# ==========================================================================
START_SIGNAL_ROI = (0.50, 0.00, 1.00, 1.00)
START_GREEN_LOWER = np.array((58, 65, 100), dtype=np.uint8)
START_GREEN_UPPER = np.array((85, 255, 255), dtype=np.uint8)
START_GREEN_MIN_AREA_RATIO = 0.00080
START_GREEN_MAX_AREA_RATIO = 0.01000
START_GREEN_MIN_ASPECT_RATIO = 0.70
START_GREEN_MAX_ASPECT_RATIO = 1.35
START_GREEN_MIN_CIRCULARITY = 0.70
START_GREEN_MIN_CIRCLE_FILL = 0.65
START_GREEN_CONFIRM_FRAMES = 3
START_SIGNAL_HZ = 15.0
START_GREEN_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# If no green light is confirmed within this many seconds, stop waiting and
# drive on the trained model anyway (0 = wait forever, matching the notebook).
GREEN_WAIT_TIMEOUT_S = 20.0


def _start_signal_roi_pixels(frame):
    height, width = frame.shape[:2]
    x_min, y_min, x_max, y_max = START_SIGNAL_ROI
    x0 = max(0, min(width - 1, int(round(width * x_min))))
    y0 = max(0, min(height - 1, int(round(height * y_min))))
    x1 = max(x0 + 1, min(width, int(round(width * x_max))))
    y1 = max(y0 + 1, min(height, int(round(height * y_max))))
    return x0, y0, x1, y1


def detect_start_green_circle(frame):
    """Return the best tested HSV-green circular candidate in the right ROI."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _start_signal_roi_pixels(frame)
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, START_GREEN_LOWER, START_GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, START_GREEN_MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, START_GREEN_MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(height * width)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / frame_area
        if not START_GREEN_MIN_AREA_RATIO <= area_ratio <= START_GREEN_MAX_AREA_RATIO:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        _, _, box_width, box_height = cv2.boundingRect(contour)
        aspect_ratio = float(box_width) / max(float(box_height), 1.0)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        enclosing_area = math.pi * float(radius) * float(radius)
        circle_fill = area / max(enclosing_area, 1.0)

        if not (START_GREEN_MIN_ASPECT_RATIO <= aspect_ratio <= START_GREEN_MAX_ASPECT_RATIO):
            continue
        if circularity < START_GREEN_MIN_CIRCULARITY:
            continue
        if circle_fill < START_GREEN_MIN_CIRCLE_FILL:
            continue

        score = area_ratio * circularity * circle_fill
        candidate = {
            "found": True, "area_ratio": area_ratio, "aspect_ratio": aspect_ratio,
            "circularity": circularity, "circle_fill": circle_fill,
        }
        if best is None or score > best[0]:
            best = (score, candidate)

    if best is None:
        return {"found": False, "area_ratio": 0.0, "aspect_ratio": 0.0,
                "circularity": 0.0, "circle_fill": 0.0}
    return best[1]


def inference_lidar_loop(lidar_stop, lidar_lock, lidar_buffer, lidar_error):
    period = 1.0 / INFERENCE_LIDAR_HZ
    while not lidar_stop.is_set():
        started = time.monotonic()
        try:
            request_start = time.monotonic()
            scan = read_lidar(INFERENCE_LIDAR_STEP_DEG)
            request_end = time.monotonic()
            validate_lidar_scan(scan)
            item = {
                "time": 0.5 * (request_start + request_end),
                "vector": lidar_to_vector(scan),
                "front": front_obstacle_distance(scan),
            }
            with lidar_lock:
                lidar_buffer.append(item)
                lidar_error["text"] = ""
        except Exception as exc:
            with lidar_lock:
                lidar_error["text"] = str(exc)
        lidar_stop.wait(max(0.0, period - (time.monotonic() - started)))


def run(model_path):
    # Stop any command left over from a previous run before loading the model.
    drive(0.0, 0.0)

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    if set(input_names) != {"camera", "lidar"}:
        raise RuntimeError(f"Expected camera+LiDAR inputs, got {input_names}")
    print(f"[INFO] loaded {model_path}")

    lidar_stop = threading.Event()
    lidar_lock = threading.Lock()
    lidar_buffer = deque(maxlen=20)
    lidar_error = {"text": "not read yet"}
    lidar_thread = threading.Thread(
        target=inference_lidar_loop,
        args=(lidar_stop, lidar_lock, lidar_buffer, lidar_error),
        daemon=True,
    )
    lidar_thread.start()

    obstacle_tracker = RaceObstacleTracker()
    curve_gate = CurveSeverityGate()
    curve_planner = CurveDistancePlanner()
    speed_controller = RaceSpeedController()
    look(CAMERA_PAN, CAMERA_TILT)
    frame_number = 0

    try:
        # The LiDAR thread is already warming its buffer while the car waits.
        green_streak = 0
        signal_period = 1.0 / START_SIGNAL_HZ
        wait_start = time.monotonic()
        timed_out = False
        drive(0.0, 0.0)
        print(f"[WAIT_GREEN] 0/{START_GREEN_CONFIRM_FRAMES} · vehicle stopped, "
              "watching right-half ROI")

        while green_streak < START_GREEN_CONFIRM_FRAMES:
            signal_tick = time.monotonic()
            if GREEN_WAIT_TIMEOUT_S > 0 and signal_tick - wait_start >= GREEN_WAIT_TIMEOUT_S:
                timed_out = True
                break
            drive(0.0, 0.0)
            signal_image = camera(CAMERA_W, CAMERA_H)
            signal_detection = detect_start_green_circle(signal_image)
            green_streak = green_streak + 1 if signal_detection["found"] else 0
            print(f"\r[WAIT_GREEN] {green_streak}/{START_GREEN_CONFIRM_FRAMES} · "
                  f"circle {signal_detection['circularity']:.2f} · "
                  f"fill {signal_detection['circle_fill']:.2f} · "
                  f"{GREEN_WAIT_TIMEOUT_S - (signal_tick - wait_start):4.1f}s left", end="")
            sys.stdout.flush()
            time.sleep(max(0.0, signal_period - (time.monotonic() - signal_tick)))

        # From this point onward detect_start_green_circle is deliberately
        # never called again -- the first motion command comes from the
        # camera+LiDAR inference path below, not a fixed straight command.
        if timed_out:
            print(f"\n[GREEN TIMEOUT] no green light in {GREEN_WAIT_TIMEOUT_S:.0f}s; "
                  "starting ver13 inference anyway (forced start)")
        else:
            print("\n[GREEN CONFIRMED] signal detector disabled; ver13 inference started")

        while True:
            started = time.monotonic()
            image = camera(CAMERA_W, CAMERA_H)
            with lidar_lock:
                latest = lidar_buffer[-1] if lidar_buffer else None
                current_error = lidar_error["text"]

            if latest is None:
                drive(0, 0)
                print(f"\r[WAIT_LIDAR] {current_error}", end="")
                sys.stdout.flush()
                time.sleep(max(0.0, 1.0 / INFERENCE_HZ - (time.monotonic() - started)))
                continue

            lidar_age = time.monotonic() - latest["time"]
            stale = lidar_age > MAX_INFERENCE_LIDAR_AGE_S
            lidar_vector = (
                np.ones(LIDAR_VECTOR_BINS, dtype=np.float32) if stale else latest["vector"])
            raw_front = math.inf if stale else latest["front"]
            tracked_front = (
                obstacle_tracker.distance if stale
                else obstacle_tracker.update(raw_front, speed_controller.sent_speed, latest["time"])
            )
            obstacle_command = obstacle_speed(tracked_front)

            camera_input = np.ascontiguousarray(image.transpose(2, 0, 1)[None].astype(np.float32))
            lidar_input = np.ascontiguousarray(lidar_vector[None].astype(np.float32))
            output = session.run(None, {"camera": camera_input, "lidar": lidar_input})[0][0]

            learned_speed = float(np.clip(output[0], 0.0, 1.0) * MAX_SPEED)
            learned_steering = float(np.clip(output[1], -1.0, 1.0) * MAX_STEERING)
            predicted_front = float(np.clip(output[2], 0.0, 1.0) * LIDAR_VECTOR_MAX_M)

            curve_bend, orange_components = orange_curve_bend(image)
            curve_severity = curve_gate.update(curve_bend)
            curve_command, curve_remaining, curve_target = curve_planner.update(
                started, curve_severity, speed_controller.sent_speed)

            # Steering is always the learned camera+LiDAR command. LiDAR rules
            # only ever lower the speed; they never override left/right.
            final_steering = learned_steering
            desired_speed = ver10_target_speed(
                curve_command, obstacle_command,
                INFERENCE_LIDAR_STALE_SPEED_CAP if stale else None)
            final_speed = speed_controller.update(started, desired_speed)
            drive(final_speed, final_steering)

            if frame_number % 3 == 0:
                raw_front_text = "none" if not np.isfinite(raw_front) else f"{raw_front:.2f}m"
                tracked_front_text = (
                    "none" if not np.isfinite(tracked_front) else f"{tracked_front:.2f}m")
                bend_text = "none" if not np.isfinite(curve_bend) else f"{curve_bend:+.1f}px"
                curve_name = CurveSeverityGate.NAMES[curve_severity]
                active_modes = []
                if obstacle_command < RACE_CLEAR_SPEED_MPS - 1e-6:
                    active_modes.append(f"cone {obstacle_command:.2f}")
                if curve_severity > 0:
                    active_modes.append(f"{curve_name} curve {curve_command:.2f}/{curve_target:.2f}")
                mode = " + ".join(active_modes) if active_modes else "clear 3.00"
                if stale:
                    mode += f" · LiDAR stale, capped {INFERENCE_LIDAR_STALE_SPEED_CAP:.2f}"
                print(
                    f"\r{mode} · target {desired_speed:.3f} -> sent {final_speed:.3f} m/s · "
                    f"steering {final_steering:+.1f} deg · front raw/tracked "
                    f"{raw_front_text}/{tracked_front_text} · curve remaining "
                    f"{curve_remaining:.2f}m · bend {bend_text} ({orange_components} comp) · "
                    f"demo speed {learned_speed:.2f} m/s · aux front {predicted_front:.2f}m · "
                    f"LiDAR {INFERENCE_LIDAR_STEP_DEG:.1f}deg/{lidar_age:.3f}s   ",
                    end="")
                sys.stdout.flush()
            frame_number += 1
            time.sleep(max(0.0, 1.0 / INFERENCE_HZ - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        drive(0, 0)
        lidar_stop.set()
        lidar_thread.join(timeout=1.0)
        print("\n[INFO] stopped")


def main():
    global BASE_URL, GREEN_WAIT_TIMEOUT_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", type=str, default="http://localhost:8000",
                     help="PhysiCar API base URL (default matches ver13_1.ipynb)")
    ap.add_argument("--model", type=str, default="",
                     help="path to model.onnx (default: <this folder>/models_ver13/model.onnx)")
    ap.add_argument("--green-timeout", type=float, default=GREEN_WAIT_TIMEOUT_S,
                     help="seconds to wait for a green light before forcing a "
                          "start (0 = wait forever)")
    args = ap.parse_args()

    BASE_URL = args.base_url
    GREEN_WAIT_TIMEOUT_S = args.green_timeout

    model_path = Path(args.model) if args.model else Path(HERE, "models_ver13", "model.onnx")
    run(model_path)


if __name__ == "__main__":
    main()
