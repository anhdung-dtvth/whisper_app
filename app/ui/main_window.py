"""
Main application window using PyQt5.

Layout:
  - Left panel: Hand skeleton visualization (OpenCV canvas → QLabel)
  - Right panel: Prediction text display with history
  - Bottom bar: Status (FPS, connection, model, device)
  - Toolbar: Start/Stop, toggle skeleton
"""

import logging
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QTextEdit, QStatusBar, QFrame, QSplitter,
    QToolBar, QAction, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

from ..ui.visualization import HandVisualizer

NUM_JOINTS = 42
NUM_FEATURES = 7

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main window for the WhisperSign real-time recognition application.

    Signals from background threads (inference results, new frames) are
    delivered via Qt signals to keep the UI thread-safe.
    """

    # Signals for cross-thread updates
    prediction_received = pyqtSignal(str, float)   # (text, confidence)
    frame_received = pyqtSignal(np.ndarray)         # (42, 7) frame
    model_loading_started = pyqtSignal()
    model_loading_finished = pyqtSignal(bool, str)  # (model_loaded, device)

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        vis_fps: int = 30,
        show_skeleton: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("WhisperSign — Real-time VSL Recognition")
        self.resize(width, height)
        # Keep UI geometry fixed to prevent layout jumps while text updates.
        self.setFixedSize(width, height)

        self._vis_fps = vis_fps
        self._show_skeleton = show_skeleton
        self._right_panel_ratio = 0.38
        self._visualizer = HandVisualizer(width=640, height=480, show_confidence=True)
        self._current_frame = np.zeros((NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
        self._fps_counter = _FpsCounter()
        self._is_running = False
        self._model_loading = False
        # Tracks only the latest accepted output for dedupe/reset behavior.
        self._output_stack = []

        # --- Callbacks to controller (set externally via set_callbacks) ---
        self._on_start = None
        self._on_stop = None

        self._init_ui()
        self._init_signals()
        self._init_timer()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        main_layout.addWidget(splitter)

        # --- Left panel: Skeleton visualization ---
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._vis_label = QLabel()
        self._vis_label.setAlignment(Qt.AlignCenter)
        self._vis_label.setMinimumSize(400, 300)
        self._vis_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._vis_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        left_layout.addWidget(self._vis_label)

        splitter.addWidget(left_frame)

        # --- Right panel: Predictions ---
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_panel_width = int(self.width() * self._right_panel_ratio)
        right_frame.setMinimumWidth(right_panel_width)
        right_frame.setMaximumWidth(right_panel_width)

        # Current prediction (large)
        self._current_label = QLabel("—")
        self._current_label.setAlignment(Qt.AlignCenter)
        self._current_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self._current_label.setStyleSheet(
            "color: #00e676; background-color: #1e1e1e; "
            "border: 1px solid #444; border-radius: 8px; padding: 16px;"
        )
        self._current_label.setMinimumHeight(110)
        self._current_label.setMaximumHeight(110)
        right_layout.addWidget(self._current_label)

        # Confidence
        self._confidence_label = QLabel("Confidence: —")
        self._confidence_label.setAlignment(Qt.AlignCenter)
        self._confidence_label.setFont(QFont("Segoe UI", 12))
        self._confidence_label.setStyleSheet("color: #aaa;")
        right_layout.addWidget(self._confidence_label)

        # History
        history_label = QLabel("Prediction History")
        history_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        history_label.setStyleSheet("color: #ccc; margin-top: 8px;")
        right_layout.addWidget(history_label)

        self._history_text = QTextEdit()
        self._history_text.setReadOnly(True)
        self._history_text.setFont(QFont("Consolas", 10))
        self._history_text.setStyleSheet(
            "background-color: #1e1e1e; color: #ddd; border: 1px solid #444;"
        )
        right_layout.addWidget(self._history_text)

        splitter.addWidget(right_frame)
        splitter.setStretchFactor(0, 3)  # vis takes 60%
        splitter.setStretchFactor(1, 2)  # text takes 40%
        splitter.setSizes([self.width() - right_panel_width, right_panel_width])

        # Disable user drag so panel sizes remain stable.
        handle = splitter.handle(1)
        if handle is not None:
            handle.setEnabled(False)

        # --- Toolbar ---
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)

        self._start_action = QAction("Start", self)
        self._start_action.triggered.connect(self._on_start_clicked)
        toolbar.addAction(self._start_action)

        self._stop_action = QAction("Stop", self)
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._on_stop_clicked)
        toolbar.addAction(self._stop_action)

        toolbar.addSeparator()

        
        self._skeleton_action = QAction("Toggle Skeleton", self)
        self._skeleton_action.setCheckable(True)
        self._skeleton_action.setChecked(self._show_skeleton)
        self._skeleton_action.triggered.connect(self._toggle_skeleton)
        toolbar.addAction(self._skeleton_action)

        # --- Status bar ---
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._fps_status = QLabel("FPS: —")
        self._connection_status = QLabel("LMC: Disconnected")
        self._model_status = QLabel("Model: Not loaded")
        self._device_status = QLabel("Device: —")

        for lbl in [self._fps_status, self._connection_status,
                     self._model_status, self._device_status]:
            lbl.setStyleSheet("padding: 0 8px;")
            self._statusbar.addPermanentWidget(lbl)

    def _init_signals(self):
        self.prediction_received.connect(self._update_prediction)
        self.frame_received.connect(self._update_frame)
        self.model_loading_started.connect(self._on_model_loading_started)
        self.model_loading_finished.connect(self._on_model_loading_finished)

    def _init_timer(self):
        """Timer to refresh the visualization at the target FPS."""
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_frame)
        # Don't start until running

    # ------------------------------------------------------------------
    # Public API (called by AppController)
    # ------------------------------------------------------------------

    def set_callbacks(self, on_start, on_stop):
        """Set start/stop callbacks to the AppController."""
        self._on_start = on_start
        self._on_stop = on_stop

    def update_status(
        self,
        fps: float = None,
        lmc_connected: bool = None,
        model_loaded: bool = None,
        device: str = None,
    ):
        """Update the status bar labels."""
        if fps is not None:
            self._fps_status.setText(f"FPS: {fps:.0f}")
        if lmc_connected is not None:
            status = "Connected" if lmc_connected else "Disconnected"
            color = "#0f0" if lmc_connected else "#f44"
            self._connection_status.setText(f"LMC: {status}")
            self._connection_status.setStyleSheet(f"color: {color}; padding: 0 8px;")
        if model_loaded is not None:
            status = "Loaded" if model_loaded else "Not loaded"
            self._model_status.setText(f"Model: {status}")
        if device is not None:
            self._device_status.setText(f"Device: {device}")

    def on_prediction(self, text: str, confidence: float):
        """Thread-safe: emit signal for prediction update."""
        self.prediction_received.emit(text, confidence)

    def on_frame(self, frame: np.ndarray):
        """Thread-safe: emit signal for new frame data."""
        self.frame_received.emit(frame)

    def on_model_loading_started(self):
        """Thread-safe: emit signal when model loading starts."""
        self.model_loading_started.emit()

    def on_model_loaded(self, model_loaded: bool, device: str):
        """Thread-safe: emit signal when model loading completes."""
        self.model_loading_finished.emit(model_loaded, device)

    def set_running(self, running: bool):
        self._is_running = running
        self._start_action.setEnabled((not running) and (not self._model_loading))
        self._stop_action.setEnabled(running)
        if running:
            interval_ms = max(1, int(1000 / self._vis_fps))
            self._render_timer.start(interval_ms)
        else:
            self._render_timer.stop()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot(str, float)
    def _update_prediction(self, text: str, confidence: float):
        if not text:
            self._current_label.setText("...")
            return

        if self._output_stack and text == self._output_stack[-1]:
            return

        # Output changed: clear stack and keep only the newest accepted output.
        self._output_stack.clear()
        self._output_stack.append(text)

        self._current_label.setText(text)
        self._confidence_label.setText(f"Confidence: {confidence:.1%}")
        self._history_text.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        logger.info("Accepted output: %s (confidence=%.3f)", text, confidence)

        # Auto-scroll
        scrollbar = self._history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(np.ndarray)
    def _update_frame(self, frame: np.ndarray):
        self._current_frame = frame
        self._fps_counter.tick()

    @pyqtSlot()
    def _on_model_loading_started(self):
        self._model_loading = True
        if not self._is_running:
            self._start_action.setEnabled(False)
        self._model_status.setText("Model: Loading...")

    @pyqtSlot(bool, str)
    def _on_model_loading_finished(self, model_loaded: bool, device: str):
        self._model_loading = False
        if not self._is_running:
            self._start_action.setEnabled(True)
        self.update_status(model_loaded=model_loaded, device=device)

    def _render_frame(self):
        """Called by the timer to render the current frame."""
        if not self._show_skeleton:
            return

        canvas = self._visualizer.render(self._current_frame)

        # Update FPS in status
        fps = self._fps_counter.get_fps()
        self._fps_status.setText(f"FPS: {fps:.0f}")

        # Convert BGR → RGB → QImage → QPixmap
        h, w, ch = canvas.shape
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to label size keeping aspect ratio
        scaled = pixmap.scaled(
            self._vis_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._vis_label.setPixmap(scaled)

    def _on_start_clicked(self):
        if self._on_start:
            self._on_start()

    def _on_stop_clicked(self):
        if self._on_stop:
            self._on_stop()

    def _toggle_skeleton(self, checked: bool):
        self._show_skeleton = checked

    def closeEvent(self, event):
        """Ensure cleanup on window close."""
        if self._on_stop and self._is_running:
            self._on_stop()
        event.accept()


class _FpsCounter:
    """Simple FPS counter using a sliding window of timestamps."""

    def __init__(self, window: int = 30):
        self._window = window
        self._timestamps: list = []

    def tick(self):
        now = time.monotonic()
        self._timestamps.append(now)
        # Keep only last N
        if len(self._timestamps) > self._window:
            self._timestamps = self._timestamps[-self._window:]

    def get_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt
