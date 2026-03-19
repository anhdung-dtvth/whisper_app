"""
Real-time preprocessing pipeline for Leap Motion Controller data.

Transforms raw LMC frames (42, 7) into model-ready tensors by applying:
    1. Build fixed-duration sliding windows with overlap retention
    2. Coordinate normalization (Leap mm -> [0, 1])
    3. Velocity recomputation from buffered positions
    4. Spatial normalization (hand-centric)
    5. Scale normalization (bone-length-based)

Uses a thread-safe FrameBuffer for accumulating frames from the LMC callback.
"""

import threading
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

NUM_JOINTS = 42
NUM_FEATURES = 7


class FrameBuffer:
    """
    Thread-safe circular buffer storing recent LMC frames.

    Used by the LMC callback thread (producer) and the inference thread (consumer).
    """

    def __init__(self, max_frames: int = 180, fps: float = 60.0):
        """
        Args:
            max_frames: Maximum number of frames to store (window_duration * fps).
            fps: Expected frame rate, used for velocity computation.
        """
        self._max_frames = max_frames
        self._fps = fps
        self._buffer = np.zeros((max_frames, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
        self._timestamps = np.zeros(max_frames, dtype=np.int64)
        self._count = 0  # total frames added (monotonically increasing)
        self._last_window_end_count = 0
        self._min_interval_us = int(1_000_000 / max(float(fps), 1e-6))
        self._last_kept_timestamp_us = -1
        self._lock = threading.Lock()

    def add_frame(self, timestamp_us: int, frame: np.ndarray):
        """
        Add a frame to the buffer. Thread-safe.

        Args:
            timestamp_us: Frame timestamp in microseconds.
            frame: (42, 7) numpy array.
        """
        with self._lock:
            # Keep processing cadence aligned with target fps (e.g., 60Hz)
            # when Leap events arrive at higher raw rate (typically ~120Hz).
            if self._last_kept_timestamp_us >= 0 and timestamp_us > self._last_kept_timestamp_us:
                if (timestamp_us - self._last_kept_timestamp_us) < self._min_interval_us:
                    return

            idx = self._count % self._max_frames
            self._buffer[idx] = frame
            self._timestamps[idx] = timestamp_us
            self._count += 1
            self._last_kept_timestamp_us = timestamp_us

    def get_window(self) -> Tuple[np.ndarray, int]:
        """
        Get the current window of frames in chronological order.

        Returns:
            (data, length) where data is (max_frames, 42, 7) and length
            is the number of valid frames (may be < max_frames).
        """
        with self._lock:
            length = min(self._count, self._max_frames)
            if length == 0:
                return np.zeros((self._max_frames, NUM_JOINTS, NUM_FEATURES), dtype=np.float32), 0

            if self._count < self._max_frames:
                # Buffer not full yet — return what we have
                data = self._latest_n_unlocked(length)
                padded = np.zeros((self._max_frames, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
                padded[:length] = data
                return padded, length

            # Buffer is full — return the latest max_frames in chronological order
            result = self._latest_n_unlocked(self._max_frames)
            return result, self._max_frames

    def get_sliding_window(self, window_frames: int, overlap: float = 0.5) -> Tuple[np.ndarray, int]:
        """
        Return one fixed-size window when enough new frames are available.

        This mirrors the Option B loop:
          1) build a fixed-duration window
          2) keep overlap frames for the next inference step

        Returns:
            (window, window_frames) when ready, otherwise (zeros, 0)
        """
        if window_frames <= 0:
            return np.zeros((0, NUM_JOINTS, NUM_FEATURES), dtype=np.float32), 0

        overlap = float(np.clip(overlap, 0.0, 0.99))
        step_frames = max(1, int(window_frames * (1.0 - overlap)))

        with self._lock:
            if self._count < window_frames:
                return np.zeros((window_frames, NUM_JOINTS, NUM_FEATURES), dtype=np.float32), 0

            next_emit_count = window_frames
            if self._last_window_end_count > 0:
                next_emit_count = self._last_window_end_count + step_frames

            if self._count < next_emit_count:
                return np.zeros((window_frames, NUM_JOINTS, NUM_FEATURES), dtype=np.float32), 0

            window = self._latest_n_unlocked(window_frames)
            self._last_window_end_count = self._count
            return window, window_frames

    def _latest_n_unlocked(self, n: int) -> np.ndarray:
        """Return the latest n frames in chronological order (lock must be held)."""
        available = min(self._count, self._max_frames)
        n = int(min(max(n, 0), available))
        if n == 0:
            return np.zeros((0, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)

        if self._count <= self._max_frames:
            start = self._count - n
            return self._buffer[start:self._count].copy()

        end_idx = self._count % self._max_frames
        start_idx = (end_idx - n) % self._max_frames

        if start_idx < end_idx:
            return self._buffer[start_idx:end_idx].copy()

        return np.concatenate(
            [
                self._buffer[start_idx:],
                self._buffer[:end_idx],
            ],
            axis=0,
        ).copy()

    def is_ready(self, min_frames: int = 30) -> bool:
        """Whether buffer has at least min_frames of data."""
        with self._lock:
            return self._count >= min_frames

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer[:] = 0
            self._timestamps[:] = 0
            self._count = 0
            self._last_window_end_count = 0
            self._last_kept_timestamp_us = -1

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._count

    @property
    def fps(self) -> float:
        return self._fps


class RealTimePreprocessor:
    """
    Applies the full preprocessing pipeline on a window of frames:
      1. Normalize Leap coordinates (mm -> [0, 1])
      2. Recompute per-joint velocities from positions
      3. Spatial normalization (hand-centric)
      4. Scale normalization (bone-length-based)

    Designed to match the Option B inference path and training normalization
    (Whisper_modification/src/data/normalization.py).
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        spatial_normalization: bool = True,
        scale_normalization: bool = True,
        fps: float = 60.0,
        coordinate_normalization: bool = True,
        sensor_range_mm: float = 500.0,
    ):
        self._fps = fps
        self._smoothing_window = smoothing_window  # kept for backward compatibility
        self._use_spatial = spatial_normalization
        self._use_scale = scale_normalization
        self._use_coordinate_norm = coordinate_normalization
        self._sensor_range_mm = max(float(sensor_range_mm), 1e-6)

        # Lazy import from Whisper_modification to avoid hard dependency at module level
        self._spatial_normalizer = None
        self._scale_normalizer = None
        self._init_components()

    def _init_components(self):
        """Initialize normalization components from Whisper_modification."""
        try:
            from Whisper_modification.src.data.normalization import SpatialNormalizer, ScaleNormalizer
            if self._use_spatial:
                self._spatial_normalizer = SpatialNormalizer()
            if self._use_scale:
                self._scale_normalizer = ScaleNormalizer()
        except ImportError:
            logger.warning("Normalization modules not available; skipping normalization.")

    def process(self, data: np.ndarray, length: int) -> Tuple[np.ndarray, int]:
        """
        Apply full preprocessing pipeline.

        Args:
            data: (T, 42, 7) numpy array from FrameBuffer window retrieval.
            length: Number of valid frames in data.

        Returns:
            (processed_data, length) — same shape (T, 42, 7).
        """
        if length == 0:
            return data, length

        # Work only on valid frames
        valid = data[:length].copy()

        # 1. Normalize Leap coordinates (mm -> [0, 1]) to match Option B adapter behavior.
        if self._use_coordinate_norm:
            valid = self._normalize_coordinates(valid)

        # 2. Recompute per-joint velocities from positions
        valid = self._compute_velocities(valid)

        # 3. Spatial normalization
        if self._spatial_normalizer is not None:
            valid = self._spatial_normalizer.normalize(valid)

        # 4. Scale normalization
        if self._scale_normalizer is not None:
            valid = self._scale_normalizer.normalize(valid)

        # Put back into full-size array
        result = data.copy()
        result[:length] = valid
        return result, length

    def _normalize_coordinates(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize Leap Motion coordinates into [0, 1], matching LeapMotionAdapter.

        Mapping:
          x' = clip((x / range) + 0.5, 0, 1)
          y' = clip(y / range, 0, 1)
          z' = clip((z / range) + 0.5, 0, 1)
        """
        coords = keypoints[:, :, :3]
        coords[:, :, 0] = np.clip((coords[:, :, 0] / self._sensor_range_mm) + 0.5, 0.0, 1.0)
        coords[:, :, 1] = np.clip(coords[:, :, 1] / self._sensor_range_mm, 0.0, 1.0)
        coords[:, :, 2] = np.clip((coords[:, :, 2] / self._sensor_range_mm) + 0.5, 0.0, 1.0)
        keypoints[:, :, :3] = coords
        return keypoints

    def _compute_velocities(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Compute per-joint velocities from positions using np.gradient.

        Args:
            keypoints: (T, 42, 7) with positions in [:, :, :3].

        Returns:
            keypoints with velocities filled in [:, :, 3:6].
        """
        if keypoints.shape[0] > 1:
            dt = 1.0 / self._fps
            velocity = np.gradient(keypoints[:, :, :3], dt, axis=0)
            keypoints[:, :, 3:6] = velocity
        return keypoints
