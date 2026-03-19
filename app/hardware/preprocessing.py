"""
Real-time preprocessing pipeline for Leap Motion Controller data.

Transforms raw LMC frames (42, 7) into model-ready tensors by applying:
  1. Smoothing (MovingAverageSmoothing)
  2. Spatial normalization (hand-centric)
  3. Scale normalization (bone-length-based)
  4. Velocity recomputation from buffered positions

Uses a thread-safe FrameBuffer for accumulating frames from the LMC callback.
"""

import threading
import time
import logging
import numpy as np
from typing import Optional, Tuple

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
        self._lock = threading.Lock()

    def add_frame(self, timestamp_us: int, frame: np.ndarray):
        """
        Add a frame to the buffer. Thread-safe.

        Args:
            timestamp_us: Frame timestamp in microseconds.
            frame: (42, 7) numpy array.
        """
        with self._lock:
            idx = self._count % self._max_frames
            self._buffer[idx] = frame
            self._timestamps[idx] = timestamp_us
            self._count += 1

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

            if self._count <= self._max_frames:
                # Buffer not full yet — return what we have
                data = self._buffer[:length].copy()
                # Pad to max_frames
                padded = np.zeros((self._max_frames, NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
                padded[:length] = data
                return padded, length
            else:
                # Buffer is full — unroll circular buffer
                start = self._count % self._max_frames
                result = np.concatenate([
                    self._buffer[start:],
                    self._buffer[:start],
                ], axis=0)
                return result.copy(), self._max_frames

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
      1. Recompute per-joint velocities from positions
      2. Moving average smoothing
      3. Spatial normalization (hand-centric)
      4. Scale normalization (bone-length-based)

    Designed to match the preprocessing used during model training
    (Whisper_modification/src/data/normalization.py and smoothing.py).
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        spatial_normalization: bool = True,
        scale_normalization: bool = True,
        fps: float = 60.0,
    ):
        self._fps = fps
        self._smoothing_window = smoothing_window
        self._use_spatial = spatial_normalization
        self._use_scale = scale_normalization

        # Lazy import from Whisper_modification to avoid hard dependency at module level
        self._smoother = None
        self._spatial_normalizer = None
        self._scale_normalizer = None
        self._init_components()

    def _init_components(self):
        """Initialize preprocessing components from Whisper_modification."""
        try:
            from Whisper_modification.src.utils.smoothing import MovingAverageSmoothing
            self._smoother = MovingAverageSmoothing(window_size=self._smoothing_window)
        except ImportError:
            logger.warning("MovingAverageSmoothing not available; skipping smoothing.")

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
            data: (T, 42, 7) numpy array from FrameBuffer.get_window().
            length: Number of valid frames in data.

        Returns:
            (processed_data, length) — same shape (T, 42, 7).
        """
        if length == 0:
            return data, length

        # Work only on valid frames
        valid = data[:length].copy()

        # 1. Recompute per-joint velocities from positions
        valid = self._compute_velocities(valid)

        # 2. Smoothing
        if self._smoother is not None:
            valid = self._smoother.smooth(valid)

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
