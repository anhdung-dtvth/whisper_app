"""
Model Inference Service for real-time sign language recognition.

Loads the WhisperSign model, runs sliding-window inference on buffered data,
and maps predicted token IDs to human-readable sign glosses.
"""

import json
import logging
import threading
import time
from collections import Counter
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


class GlossMapper:
    """
    Maps model token IDs to sign gloss strings.

    Expects a JSON vocabulary file: {"0": "<blank>", "1": "xin_chào", ...}
    or a plain-text file with one gloss per line (line number = token ID).
    """

    def __init__(self, vocab_path: Optional[str] = None):
        self._id_to_gloss: dict = {0: "<blank>"}
        if vocab_path is not None:
            self.load(vocab_path)

    def load(self, vocab_path: str):
        """Load vocabulary from a JSON or text file."""
        path = Path(vocab_path)
        if not path.exists():
            logger.warning(f"Vocabulary file not found: {vocab_path}. Using empty mapping.")
            return

        if path.suffix == ".json":
            text = self._read_text_with_fallback(path)
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON in vocabulary file: {vocab_path}. Using blank-only mapping."
                )
                self._id_to_gloss = {0: "<blank>"}
                return
            self._id_to_gloss = self._parse_json_mapping(raw, vocab_path)
        else:
            # Plain text: one gloss per line
            self._id_to_gloss = {}
            text = self._read_text_with_fallback(path)
            for idx, line in enumerate(text.splitlines()):
                self._id_to_gloss[idx] = line.strip()

        # Ensure CTC blank token exists.
        self._id_to_gloss.setdefault(0, "<blank>")

        logger.info(f"Loaded {len(self._id_to_gloss)} glosses from {vocab_path}")

    @staticmethod
    def _parse_json_mapping(raw: Any, vocab_path: str) -> Dict[int, str]:
        """
        Parse JSON mapping in one of two formats:
          1) {"0": "<blank>", "1": "xin_chao", ...}    (id -> gloss)
          2) {"<blank>": 0, "xin_chao": 1, ...}            (gloss -> id)
        """
        if not isinstance(raw, dict):
            logger.warning(
                f"Invalid vocabulary format in {vocab_path}: expected JSON object, got {type(raw).__name__}."
            )
            return {0: "<blank>"}

        # Format 1: id -> gloss
        try:
            return {int(k): str(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            pass

        # Format 2: gloss -> id
        inverted: Dict[int, str] = {}
        for gloss, idx in raw.items():
            if not isinstance(idx, int):
                logger.warning(
                    f"Invalid label_map entry in {vocab_path}: value for '{gloss}' is not int ({type(idx).__name__})."
                )
                continue
            inverted[idx] = str(gloss)

        if not inverted:
            logger.warning(
                f"Unable to parse vocabulary mapping in {vocab_path}. Using blank-only mapping."
            )
            return {0: "<blank>"}

        return inverted

    @staticmethod
    def _read_text_with_fallback(path: Path) -> str:
        """Read text file using common encodings encountered on Windows datasets."""
        encodings = ("utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1")
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            logger.warning(
                f"Could not decode {path} with known encodings. Using utf-8 with replacement."
            )
            return f.read()

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs to a readable string.

        Removes blank tokens (ID 0) and joins glosses with spaces.
        """
        glosses = []
        for tid in token_ids:
            if tid == 0:
                continue
            glosses.append(self._id_to_gloss.get(tid, f"<unk:{tid}>"))
        return " ".join(glosses)

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_gloss)


class InferenceService:
    """
    Loads WhisperSign model and runs inference on preprocessed windows.

    Supports periodic automatic inference via a background timer thread.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        device: str = "cuda",
        window_duration: float = 2.0,
        sample_rate: int = 60,
        inference_interval: float = 0.5,
    ):
        """
        Args:
            checkpoint_path: Path to the model checkpoint (.pt file).
            vocab_path: Path to the vocab JSON/text file.
            device: "cuda" or "cpu". Falls back to CPU if CUDA unavailable.
            window_duration: Sliding window duration in seconds.
            sample_rate: Expected frame rate (Hz).
            inference_interval: Seconds between automatic inference runs.
        """
        self._checkpoint_path = checkpoint_path
        self._device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU.")
        self._window_frames = int(window_duration * sample_rate)
        self._sample_rate = sample_rate
        self._inference_interval = inference_interval
        self._cycle_duration = max(inference_interval, 1e-3)

        self._model = None
        self._gloss_mapper = GlossMapper(vocab_path)

        # For periodic inference
        self._timer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._inference_callback: Optional[Callable] = None
        self._cycle_start_time: Optional[float] = None
        self._cycle_predictions: List[Tuple[float, List[int], str, float]] = []

        # Latest result
        self._lock = threading.Lock()
        self._last_prediction: List[int] = []
        self._last_text: str = ""
        self._last_confidence: float = 0.0
        self._model_loaded = False
        self._model_vocab_size: Optional[int] = None

    def load_model(self):
        """Load the WhisperSign model from checkpoint."""
        checkpoint_path = Path(self._checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {self._checkpoint_path}")
            logger.info("Running in mock mode — predictions will be empty.")
            self._model_loaded = False
            return

        try:
            from Whisper_modification.src.model.whisper_sign import WhisperSignModel
            model, checkpoint = WhisperSignModel.load_checkpoint(
                str(checkpoint_path), device=self._device
            )
            model = model.to(self._device)
            model.eval()
            self._model = model
            self._model_loaded = True

            # Check vocabulary compatibility so <unk:id> output is easier to diagnose.
            checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            decoder_cfg = checkpoint_cfg.get("decoder", {}) if isinstance(checkpoint_cfg, dict) else {}
            raw_vocab_size = decoder_cfg.get("vocab_size")
            self._model_vocab_size = raw_vocab_size if isinstance(raw_vocab_size, int) else None

            mapper_vocab_size = self._gloss_mapper.vocab_size
            if mapper_vocab_size <= 1:
                logger.warning(
                    "No usable vocabulary mapping loaded. Predictions will appear as <unk:id>. "
                    "Set model.vocab_path to the label_map/vocab used during training."
                )
            elif self._model_vocab_size is not None and mapper_vocab_size < self._model_vocab_size:
                logger.warning(
                    "Vocabulary mapping has %d entries but checkpoint expects %d classes. "
                    "Unknown tokens like <unk:id> are expected until you provide the correct mapping file.",
                    mapper_vocab_size,
                    self._model_vocab_size,
                )

            logger.info(
                f"Model loaded from {self._checkpoint_path} "
                f"on {self._device} ({model.get_num_params(trainable_only=False)} params)"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self._model_loaded = False

    @torch.no_grad()
    def predict(self, features: np.ndarray, length: int) -> Tuple[List[int], str, float]:
        """
        Run inference on a single window of preprocessed data.

        Args:
            features: (T, 42, 7) numpy array (preprocessed).
            length: Number of valid frames in features.

        Returns:
            (token_ids, text, confidence)
        """
        if self._model is None:
            return [], "", 0.0

        if length == 0:
            return [], "", 0.0

        # Convert to tensor: (1, T, 42, 7)
        x = torch.from_numpy(features).float().unsqueeze(0).to(self._device)
        lengths = torch.tensor([length], device=self._device)

        predictions = self._model.decode(x, lengths)
        token_ids = predictions[0] if predictions else []

        text = self._gloss_mapper.decode(token_ids)

        # Estimate a basic confidence from CTC output
        confidence = 1.0 if token_ids else 0.0

        with self._lock:
            self._last_prediction = token_ids
            self._last_text = text
            self._last_confidence = confidence

        return token_ids, text, confidence

    def start_periodic_inference(
        self,
        get_window_fn: Callable,
        on_result_fn: Callable,
    ):
        """
        Start a background thread that periodically runs inference.

        Args:
            get_window_fn: Callable returning (features, length) from the FrameBuffer.
            on_result_fn: Callable receiving (token_ids, text, confidence) for each
                aggregated cycle output.
        """
        self._inference_callback = on_result_fn
        self._reset_cycle_state()
        self._stop_event.clear()
        self._timer_thread = threading.Thread(
            target=self._inference_loop,
            args=(get_window_fn, on_result_fn),
            name="InferenceTimer",
            daemon=True,
        )
        self._timer_thread.start()
        logger.info(
            "Periodic inference started (interval=%.3fs, cycle=%.3fs)",
            self._inference_interval,
            self._cycle_duration,
        )

    def stop_periodic_inference(self):
        """Stop the periodic inference thread."""
        self._stop_event.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=3.0)
            self._timer_thread = None
        self._reset_cycle_state()
        logger.info("Periodic inference stopped.")

    def _reset_cycle_state(self):
        """Reset temporary predictions used for cycle-level majority voting."""
        self._cycle_start_time = None
        self._cycle_predictions = []

    def _add_cycle_prediction(
        self,
        timestamp: float,
        token_ids: List[int],
        text: str,
        confidence: float,
    ):
        """Append a non-empty prediction to the active cycle buffer."""
        if not text:
            return
        if self._cycle_start_time is None:
            self._cycle_start_time = timestamp
        self._cycle_predictions.append((timestamp, list(token_ids), text, confidence))

    def _select_cycle_majority(self) -> Optional[Tuple[List[int], str, float]]:
        """
        Pick majority text in the current cycle.

        If multiple texts share the highest count, choose the newest one in the cycle.
        """
        if not self._cycle_predictions:
            return None

        counts = Counter(text for _, _, text, _ in self._cycle_predictions)
        max_count = max(counts.values())
        tied_texts = {text for text, count in counts.items() if count == max_count}

        for _, token_ids, text, _ in reversed(self._cycle_predictions):
            if text in tied_texts:
                majority_confidence = max_count / len(self._cycle_predictions)
                return token_ids, text, majority_confidence

        return None

    def _finalize_cycle_if_due(self, timestamp: float) -> Optional[Tuple[List[int], str, float]]:
        """Return an aggregated cycle result when cycle duration has elapsed."""
        if self._cycle_start_time is None:
            return None

        if (timestamp - self._cycle_start_time) < self._cycle_duration:
            return None

        cycle_result = self._select_cycle_majority()
        self._cycle_start_time = timestamp
        self._cycle_predictions = []
        return cycle_result

    def _inference_loop(self, get_window_fn, on_result_fn):
        """Background loop that runs inference and emits cycle-majority outputs."""
        while not self._stop_event.is_set():
            start = time.monotonic()
            try:
                features, length = get_window_fn()
                now = time.monotonic()

                # Close and emit the previous cycle before appending new predictions.
                cycle_result = self._finalize_cycle_if_due(now)
                if cycle_result is not None:
                    on_result_fn(*cycle_result)

                if length > 0:
                    token_ids, text, confidence = self.predict(features, length)
                    self._add_cycle_prediction(time.monotonic(), token_ids, text, confidence)
            except Exception as e:
                logger.error(f"Inference error: {e}", exc_info=True)

            elapsed = time.monotonic() - start
            sleep_time = max(0, self._inference_interval - elapsed)
            self._stop_event.wait(sleep_time)

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def last_prediction(self) -> Tuple[List[int], str, float]:
        with self._lock:
            return self._last_prediction, self._last_text, self._last_confidence

    @property
    def device(self) -> str:
        return self._device
