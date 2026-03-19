"""
Hand skeleton visualization using OpenCV.

Renders 42-joint hand skeleton from LMC frame data onto a numpy image,
suitable for embedding in a PyQt5 QLabel via QImage.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# --- Joint connectivity (MediaPipe hand landmark topology) ---
# Each tuple (parent, child) defines a bone to draw.
# Per-hand (0-indexed within 21 joints):
_HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# Colors (BGR)
_COLOR_LEFT = (255, 180, 50)    # Blue-ish for left hand
_COLOR_RIGHT = (50, 180, 255)   # Orange-ish for right hand
_COLOR_BG = (30, 30, 30)        # Dark background
_COLOR_TEXT = (200, 200, 200)
_COLOR_JOINT = (255, 255, 255)  # White joints

NUM_JOINTS = 42
NUM_FEATURES = 7


class HandVisualizer:
    """
    Renders hand skeletons from (42, 7) frame data onto an OpenCV image.

    Joint layout: left hand = indices 0-20, right hand = indices 21-41.
    Features: [x, y, z, vx, vy, vz, confidence].
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        show_confidence: bool = True,
        flip_x: bool = True,
    ):
        """
        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            show_confidence: Draw joints with varying size based on confidence.
            flip_x: Mirror the x-axis so it looks natural (LMC right = screen right).
        """
        self._width = width
        self._height = height
        self._show_confidence = show_confidence
        self._flip_x = flip_x
        self._canvas = np.full((height, width, 3), _COLOR_BG, dtype=np.uint8)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Render a single frame onto a fresh canvas.

        Args:
            frame: (42, 7) numpy array. Positions in mm from LMC coordinate system.

        Returns:
            BGR image (height, width, 3) as uint8.
        """
        canvas = np.full((self._height, self._width, 3), _COLOR_BG, dtype=np.uint8)

        # Check if any hand is present (non-zero confidence)
        left_conf = frame[:21, 6].max() if frame[:21, 6].sum() > 0 else 0.0
        right_conf = frame[21:, 6].max() if frame[21:, 6].sum() > 0 else 0.0

        if left_conf > 0:
            self._draw_hand(canvas, frame[:21], _COLOR_LEFT, "Left")
        if right_conf > 0:
            self._draw_hand(canvas, frame[21:], _COLOR_RIGHT, "Right")

        if left_conf == 0 and right_conf == 0:
            cv2.putText(
                canvas, "No hands detected",
                (self._width // 2 - 100, self._height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COLOR_TEXT, 1, cv2.LINE_AA,
            )

        self._canvas = canvas
        return canvas

    def _project(self, x_mm: float, z_mm: float) -> Tuple[int, int]:
        """
        Project 3D LMC coordinates (x, z) to 2D screen coordinates.

        LMC coordinate system (top-down view):
          - X: left/right (mm), origin at sensor center
          - Y: up (mm), not used for 2D projection
          - Z: forward/backward (mm)

        We use X for horizontal and Z for vertical on screen.
        """
        # Map from LMC mm range to pixel coordinates
        # Typical interaction box: X ∈ [-250, 250], Z ∈ [-150, 150]
        scale = self._width / 500.0  # ~500mm wide interaction zone
        cx, cy = self._width // 2, self._height // 2

        px = int(cx + x_mm * scale)
        py = int(cy + z_mm * scale)

        if self._flip_x:
            px = self._width - px

        # Clamp to image bounds
        px = max(0, min(self._width - 1, px))
        py = max(0, min(self._height - 1, py))

        return px, py

    def _draw_hand(
        self,
        canvas: np.ndarray,
        joints: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
    ):
        """
        Draw one hand (21 joints) on the canvas.

        Args:
            joints: (21, 7) array.
            color: BGR color for bones.
            label: "Left" or "Right".
        """
        # Project all joints to 2D
        points_2d = []
        confidences = []
        for j in range(21):
            x, y, z = joints[j, 0], joints[j, 1], joints[j, 2]
            conf = joints[j, 6]
            px, py = self._project(x, z)
            points_2d.append((px, py))
            confidences.append(conf)

        # Draw bones
        for parent, child in _HAND_CONNECTIONS:
            if confidences[parent] > 0 and confidences[child] > 0:
                cv2.line(canvas, points_2d[parent], points_2d[child], color, 2, cv2.LINE_AA)

        # Draw joints
        for j in range(21):
            if confidences[j] > 0:
                radius = 4 if not self._show_confidence else max(2, int(confidences[j] * 5))
                cv2.circle(canvas, points_2d[j], radius, _COLOR_JOINT, -1, cv2.LINE_AA)
                # Smaller colored ring
                cv2.circle(canvas, points_2d[j], radius + 1, color, 1, cv2.LINE_AA)

        # Label near wrist
        if confidences[0] > 0:
            wx, wy = points_2d[0]
            cv2.putText(
                canvas, label, (wx - 15, wy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    @property
    def canvas(self) -> np.ndarray:
        return self._canvas

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
