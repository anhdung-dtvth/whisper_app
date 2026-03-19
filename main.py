"""
WhisperSign — Real-time Vietnamese Sign Language Recognition

Entry point. Initializes all components and starts the PyQt5 event loop.

Usage:
    python main.py [--config path/to/config.yaml] [--mock]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure the project root is on sys.path so that both `app` and
# `Whisper_modification` packages can be imported.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication

from app.utils.config import load_config
from app.core.controller import AppController
from app.ui.main_window import MainWindow


def setup_logging(config: dict):
    """Configure root logger from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(description="WhisperSign real-time VSL recognition")
    parser.add_argument("--config", type=str, default=None, help="Path to app_config.yaml")
    parser.add_argument("--mock", action="store_true", help="Use mock LMC capture (no hardware)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("WhisperSign starting...")

    # Qt app
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("WhisperSign")
    qt_app.setStyle("Fusion")

    # Main window
    ui_cfg = config["ui"]
    window = MainWindow(
        width=ui_cfg.get("window_width", 1200),
        height=ui_cfg.get("window_height", 800),
        vis_fps=ui_cfg.get("visualization_fps", 30),
        show_skeleton=ui_cfg.get("show_skeleton", True),
    )

    # Controller
    controller = AppController(config, window)
    controller.initialize()

    # Show and run
    window.show()
    logger.info("Application window shown. Ready.")

    exit_code = qt_app.exec_()

    # Cleanup
    controller.stop()
    logger.info("WhisperSign exited.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()