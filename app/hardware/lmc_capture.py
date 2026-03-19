"""
Leap Motion Controller Capture Module
Kết nối với Leap Motion Controller thông qua leapc-python-api,
trích xuất 42 joint landmarks (21 tay trái + 21 tay phải) mỗi frame,
và đưa ra tensor (42, 7) tương thích với WhisperSign model input.

Output format per joint: [x, y, z, vx, vy, vz, confidence]
  - x, y, z   : tọa độ mm (sẽ được normalize ở preprocessing)
  - vx, vy, vz: vận tốc mm/s lấy từ palm velocity (per-joint velocity tính ở buffer)
  - confidence: hand.confidence từ LeapC

Joint index mapping (giống MediaPipe Hand Landmarks):
  Left hand  → index 0–20
  Right hand → index 21–41
  Mỗi hand:
    0  = wrist (palm position)
    1–4  = thumb  (metacarpal_tip, proximal_tip, intermediate_tip, distal_tip)
    5–8  = index
    9–12 = middle
    13–16= ring
    17–20= pinky
"""

import threading
import queue
import logging
import numpy as np
from typing import Callable, Optional

try:
    import leap
    from leap import event_listener
    from leap.enums import HandType, TrackingMode
    LEAP_AVAILABLE = True
except ImportError:
    LEAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Số features mỗi joint, phải khớp với config model
NUM_JOINTS = 42
NUM_FEATURES = 7  # x, y, z, vx, vy, vz, confidence


# ---------------------------------------------------------------------------
# Helper: trích xuất 21 joint positions từ một Hand object
# ---------------------------------------------------------------------------

def _extract_hand_joints(hand) -> np.ndarray:
    """
    Trích xuất 21 landmark positions từ một leap.Hand object.

    Layout (giống MediaPipe):
        0  : wrist (= palm.position)
        1-4 : thumb  [metacarpal.next_joint, proximal.next_joint,
                      intermediate.next_joint, distal.next_joint]
        5-8 : index  (tương tự)
        9-12: middle
        13-16: ring
        17-20: pinky

    Returns:
        joints: ndarray (21, 3) — tọa độ (x, y, z) mm
    """
    joints = np.zeros((21, 3), dtype=np.float32)

    # Joint 0: wrist / palm
    palm = hand.palm.position
    joints[0] = [palm.x, palm.y, palm.z]

    # Joints 1-20: 5 ngón tay, mỗi ngón 4 joints (next_joint của 4 bones)
    finger_order = [hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky]
    for finger_idx, digit in enumerate(finger_order):
        base = 1 + finger_idx * 4
        for bone_idx, bone in enumerate(digit.bones):
            tip = bone.next_joint
            joints[base + bone_idx] = [tip.x, tip.y, tip.z]

    return joints


# ---------------------------------------------------------------------------
# Listener: nhận tracking events từ LeapC
# ---------------------------------------------------------------------------

class _TrackingListener(event_listener.Listener if LEAP_AVAILABLE else object):
    """
    Leap Listener subclass. Được gọi bởi Connection trên background thread.
    Mỗi tracking frame → chuyển thành (42, 7) numpy array → đẩy vào queue.
    """

    def __init__(self, frame_queue: queue.Queue, max_queue_size: int = 120):
        self._queue = frame_queue
        self._max_queue_size = max_queue_size
        self._prev_palm_velocity = {
            HandType.Left: np.zeros(3, dtype=np.float32),
            HandType.Right: np.zeros(3, dtype=np.float32),
        }

    def on_connection_event(self, event):
        logger.info("Leap Motion: connected to service.")

    def on_connection_lost_event(self, event):
        logger.warning("Leap Motion: connection to service lost.")

    def on_device_event(self, event):
        device = getattr(event, "device", None)
        serial = getattr(device, "serial", None)
        if serial:
            logger.info(f"Leap Motion: device found (serial={serial})")
        else:
            logger.info("Leap Motion: device found.")

    def on_device_lost_event(self, event):
        logger.warning("Leap Motion: device lost.")

    def on_tracking_event(self, event):
        """
        Được gọi mỗi khi Leap Motion gửi một tracking frame (~115–120 Hz).
        Chuyển đổi sang tensor (42, 7) và đẩy vào queue.
        """
        frame_tensor = np.zeros((NUM_JOINTS, NUM_FEATURES), dtype=np.float32)
        timestamp_us = event.timestamp  # microseconds

        for hand in event.hands:
            hand_type = hand.type  # HandType.Left hoặc HandType.Right
            offset = 0 if hand_type == HandType.Left else 21
            confidence = float(hand.confidence)

            # --- Positions (x, y, z) ---
            positions = _extract_hand_joints(hand)   # (21, 3)
            frame_tensor[offset: offset + 21, 0:3] = positions

            # --- Velocity (vx, vy, vz): lấy palm velocity từ LeapC ---
            # LeapC cung cấp palm velocity trực tiếp (mm/s)
            palm_vel = hand.palm.velocity
            velocity = np.array([palm_vel.x, palm_vel.y, palm_vel.z], dtype=np.float32)
            # Gán cùng giá trị velocity cho tất cả joints của tay này
            # (per-joint velocity sẽ được tính chính xác hơn ở RealTimeBuffer)
            frame_tensor[offset: offset + 21, 3:6] = velocity

            # --- Confidence ---
            frame_tensor[offset: offset + 21, 6] = confidence

            # Lưu lại để buffer dùng nếu cần
            self._prev_palm_velocity[hand_type] = velocity

        # Đẩy vào queue, bỏ frame cũ nếu queue đầy (drop oldest)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait((timestamp_us, frame_tensor))
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# LeapMotionCapture: public API
# ---------------------------------------------------------------------------

class LeapMotionCapture:
    """
    Quản lý kết nối với Leap Motion Controller và cung cấp frame data.

    Cách dùng (callback mode — dùng cho real-time pipeline):
        def on_frame(timestamp_us, frame_np):
            # frame_np: ndarray (42, 7)
            buffer.push(frame_np)

        cap = LeapMotionCapture(on_frame_callback=on_frame)
        cap.start()
        ...
        cap.stop()

    Cách dùng (polling mode):
        cap = LeapMotionCapture()
        cap.start()
        ts, frame = cap.get_latest_frame(timeout=0.1)
        cap.stop()
    """

    def __init__(
        self,
        on_frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        tracking_mode: "TrackingMode" = None,
        queue_maxsize: int = 120,
        **kwargs,
    ):
        """
        Args:
            on_frame_callback: Hàm gọi khi có frame mới.
                Signature: callback(timestamp_us: int, frame: np.ndarray[42, 7])
                Chạy trên dispatcher thread riêng (không block LeapC thread).
            tracking_mode: TrackingMode.Desktop (mặc định) hoặc
                           TrackingMode.HMD / TrackingMode.ScreenTop
            queue_maxsize: Kích thước tối đa của internal frame queue.
        """
        if not LEAP_AVAILABLE:
            raise ImportError(
                "leapc-python-api không tìm thấy. "
                "Cài đặt bằng: pip install leapc-python-api"
            )

        self._callback = on_frame_callback
        self._tracking_mode = tracking_mode
        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._listener = _TrackingListener(self._frame_queue, queue_maxsize)
        self._connection: Optional[leap.Connection] = None
        self._is_running = False

        # Dispatcher thread: gọi callback mà không block LeapC polling thread
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 10.0):
        """
        Mở kết nối và bắt đầu nhận tracking data.

        Args:
            timeout: Thời gian chờ kết nối (giây).

        Raises:
            RuntimeError: Nếu không thể kết nối hoặc không có thiết bị.
            ImportError: Nếu leapc-python-api chưa được cài.
        """
        if self._is_running:
            logger.warning("LeapMotionCapture đã đang chạy.")
            return

        logger.info("Đang khởi động Leap Motion capture...")

        self._stop_event.clear()
        self._connection = leap.Connection(listeners=[self._listener])
        self._connection.connect(auto_poll=True, timeout=timeout) 
        # auto_poll=True → LeapC sẽ tự động gọi listener.on_tracking_event() trên background thread mỗi khi có frame mới

        # Đặt tracking mode nếu được chỉ định
        if self._tracking_mode is not None:
            self._connection.set_tracking_mode(self._tracking_mode)
            logger.info(f"Tracking mode: {self._tracking_mode}")

        self._is_running = True
        logger.info("Leap Motion capture đã sẵn sàng.")

        # Khởi động dispatcher thread nếu có callback
        # dispatcher là thread riêng để gọi callback mà không block thread của LeapC (tránh làm chậm quá trình nhận frame mới)
        
        if self._callback is not None:
            self._dispatcher_thread = threading.Thread(
                target=self._dispatch_loop,
                name="LeapDispatcher",
                daemon=True,
            ) # Daemon thread sẽ tự động dừng khi main thread kết thúc
            self._dispatcher_thread.start()

    def stop(self):
        """Dừng kết nối và giải phóng tài nguyên."""
        if not self._is_running:
            return

        logger.info("Đang dừng Leap Motion capture...")
        self._stop_event.set()

        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=2.0)
            self._dispatcher_thread = None

        if self._connection is not None:
            self._connection.disconnect()
            self._connection = None

        self._is_running = False
        logger.info("Leap Motion capture đã dừng.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ------------------------------------------------------------------
    # Polling API
    # ------------------------------------------------------------------

    def get_latest_frame(
        self, timeout: float = 0.1
    ) -> Optional[tuple]:
        """
        Lấy frame mới nhất từ queue (blocking cho đến khi có frame hoặc timeout).

        Args:
            timeout: Thời gian chờ tối đa (giây).

        Returns:
            (timestamp_us, frame_np) hoặc None nếu timeout.
            timestamp_us: int — microseconds từ epoch Leap
            frame_np: ndarray shape (42, 7), dtype float32
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_frames(self) -> list:
        """
        Lấy tất cả frames hiện có trong queue (non-blocking).

        Returns:
            List of (timestamp_us, frame_np) tuples
        """
        frames = []
        while True:
            try:
                frames.append(self._frame_queue.get_nowait())
            except queue.Empty:
                break
        return frames

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def queue_size(self) -> int:
        """Số frames hiện đang chờ trong queue."""
        return self._frame_queue.qsize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dispatch_loop(self):
        """
        Background thread: liên tục lấy frame từ queue và gọi callback.
        Chạy cho đến khi stop_event được set.
        """
        while not self._stop_event.is_set():
            result = self.get_latest_frame(timeout=0.05)
            if result is not None and self._callback is not None:
                timestamp_us, frame_np = result
                try:
                    self._callback(timestamp_us, frame_np)
                except Exception as e:
                    logger.error(f"Lỗi trong on_frame_callback: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Stub khi Leap không available (dùng để test UI mà không có phần cứng)
# ---------------------------------------------------------------------------

class MockLeapMotionCapture:
    """
    Giả lập Leap Motion Capture với dữ liệu ngẫu nhiên.
    Dùng để test pipeline khi không có phần cứng.
    """

    def __init__(
        self,
        on_frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
        fps: int = 60,
        **kwargs,
    ):
        self._callback = on_frame_callback
        self._fps = fps
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, **kwargs):
        self._is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._generate_loop,
            name="MockLeap",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"MockLeapMotionCapture started at {self._fps} Hz")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _generate_loop(self):
        import time
        interval = 1.0 / self._fps
        t = 0
        while not self._stop_event.is_set():
            frame = np.random.randn(NUM_JOINTS, NUM_FEATURES).astype(np.float32) * 80
            # Cả hai tay đều "detected"
            frame[:, 6] = 1.0  # confidence = 1.0
            timestamp_us = int(t * 1_000_000)
            if self._callback:
                try:
                    self._callback(timestamp_us, frame)
                except Exception as e:
                    logger.error(f"Mock callback error: {e}")
            t += interval
            time.sleep(interval)

    @property
    def is_running(self) -> bool:
        return self._is_running


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_capture(
    on_frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    mock: bool = False,
    **kwargs,
) -> "LeapMotionCapture | MockLeapMotionCapture":
    """
    Tạo capture instance phù hợp.

    Args:
        on_frame_callback: Hàm nhận (timestamp_us, frame_np[42,7]) mỗi frame.
        mock: Nếu True, trả về MockLeapMotionCapture (test không cần phần cứng).
        **kwargs: Các tham số khác truyền vào constructor.

    Returns:
        LeapMotionCapture nếu mock=False và LEAP_AVAILABLE=True,
        ngược lại MockLeapMotionCapture.
    """
    if mock or not LEAP_AVAILABLE:
        if not mock:
            logger.warning(
                "leapc-python-api không tìm thấy, dùng MockLeapMotionCapture."
            )
        return MockLeapMotionCapture(on_frame_callback=on_frame_callback, **kwargs)
    return LeapMotionCapture(on_frame_callback=on_frame_callback, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test  (chạy trực tiếp: python lmc_capture.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import argparse
    import collections
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="lmc_capture self-test")
    parser.add_argument("--mock", action="store_true",
                        help="Dùng MockLeapMotionCapture (không cần phần cứng)")
    parser.add_argument("--fps", type=int, default=60,
                        help="FPS cho mock mode (default: 60)")
    parser.add_argument("--warmup", type=float, default=1.5,
                        help="Warmup trước khi đo FPS (giây, default: 1.5)")
    args = parser.parse_args()

    USE_MOCK = args.mock or not LEAP_AVAILABLE
    WARMUP   = args.warmup
    MOCK_FPS = args.fps

    # ── Counters ──────────────────────────────────────────────────────────
    frame_count       = 0
    measure_count     = 0
    shape_errors      = 0
    dtype_errors      = 0
    measure_wall_times: list = []
    first_frame_time  = [None]
    warmup_done       = [False]
    hands_seen        = [set()]       # track loại tay đã thấy
    last_frame_cache  = [None]        # lưu frame cuối để in khi dừng

    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    BONE_NAMES   = ["Meta ", "Prox ", "Inter", "Distal"]
    JOINT_LABELS = ["Wrist "]
    for _fn in FINGER_NAMES:
        for _bn in BONE_NAMES:
            JOINT_LABELS.append(f"{_fn[:5]}/{_bn}")

    def print_frame_table(frame: np.ndarray, title: str = ""):
        """In bảng 42 khớp của 1 frame."""
        if title:
            print(f"\n  {title}")
        print(f"  {'─'*72}")
        print(f"  {'Jnt':>3}  {'Label':<12}  "
              f"{'x':>10} {'y':>10} {'z':>10}  "
              f"{'vx':>8} {'vy':>8} {'vz':>8}  {'conf':>5}")
        print(f"  {'─'*72}")
        for _side, offset, side_label in [("L", 0, "LEFT "), ("R", 21, "RIGHT")]:
            has_data = np.any(frame[offset:offset+21] != 0)
            status = "" if has_data else "  (không phát hiện)"
            print(f"  ── {side_label} HAND{status} ──")
            for i in range(21):
                idx = offset + i
                x, y, z, vx, vy, vz, conf = frame[idx]
                label = JOINT_LABELS[i]
                print(f"  {idx:>3}  {label:<12}  "
                      f"{x:>10.4f} {y:>10.4f} {z:>10.4f}  "
                      f"{vx:>8.4f} {vy:>8.4f} {vz:>8.4f}  {conf:>5.2f}")
        print(f"  {'─'*72}")

    # onframe callback: đếm frame, đo FPS, kiểm tra shape/dtype, in bảng mỗi 20 frames
    def on_frame(timestamp_us: int, frame: np.ndarray):
        global frame_count, measure_count, shape_errors, dtype_errors
        t_now = time.perf_counter()
        frame_count += 1
        last_frame_cache[0] = frame.copy()

        if first_frame_time[0] is None:
            first_frame_time[0] = t_now

        # ── Warmup ──
        if not warmup_done[0]:
            if t_now - first_frame_time[0] >= WARMUP:
                warmup_done[0] = True
                print(f"\n  ... warmup xong ({WARMUP}s), đang capture real-time ...")
                print(f"  ... nhấn Ctrl+C để dừng ...\n")
            return

        measure_count += 1
        measure_wall_times.append(t_now)

        # Kiểm tra shape & dtype
        if frame.shape != (NUM_JOINTS, NUM_FEATURES):
            shape_errors += 1
        if frame.dtype != np.float32:
            dtype_errors += 1

        # Track tay nào visible
        left_visible  = np.any(frame[0:21] != 0)
        right_visible = np.any(frame[21:42] != 0)
        if left_visible:
            hands_seen[0].add("Left")
        if right_visible:
            hands_seen[0].add("Right")

        # In bảng 42 khớp ở frame đầu tiên SAU warmup
        if measure_count == 1:
            print_frame_table(frame, "Frame đầu tiên sau warmup:")
            print()

        # ── Live data mỗi ~0.5s (mỗi 20 frames ở 38Hz) ──
        if measure_count > 1 and (measure_count % 20 == 0):
            elapsed = t_now - measure_wall_times[0]
            live_fps = (measure_count - 1) / elapsed if elapsed > 0 else 0
            elapsed_total = t_now - first_frame_time[0]

            # Palm positions + velocity + confidence (cập nhật real-time)
            lp = frame[0, :3]   # left palm xyz
            lv = frame[0, 3:6]  # left palm velocity
            lc = frame[0, 6]    # left confidence
            rp = frame[21, :3]  # right palm xyz
            rv = frame[21, 3:6] # right palm velocity
            rc = frame[21, 6]   # right confidence

            # Fingertip positions (index=8, middle=12 — distal tips)
            l_index_tip = frame[8, :3]
            l_mid_tip   = frame[12, :3]
            r_index_tip = frame[29, :3]
            r_mid_tip   = frame[33, :3]

            print(f"  ┌─ Frame #{measure_count:<6}  ⏱ {elapsed_total:.1f}s  │  {live_fps:.1f} Hz  │  err: {shape_errors+dtype_errors}")
            if left_visible:
                speed_l = np.linalg.norm(lv)
                print(f"  │ LEFT   conf={lc:.2f}  palm=({lp[0]:>7.1f}, {lp[1]:>7.1f}, {lp[2]:>7.1f})  "
                      f"vel=({lv[0]:>7.1f}, {lv[1]:>7.1f}, {lv[2]:>7.1f})  speed={speed_l:.1f}mm/s")
                print(f"  │        index_tip=({l_index_tip[0]:>7.1f}, {l_index_tip[1]:>7.1f}, {l_index_tip[2]:>7.1f})  "
                      f"middle_tip=({l_mid_tip[0]:>7.1f}, {l_mid_tip[1]:>7.1f}, {l_mid_tip[2]:>7.1f})")
            else:
                print(f"  │ LEFT   (không phát hiện)")
            if right_visible:
                speed_r = np.linalg.norm(rv)
                print(f"  │ RIGHT  conf={rc:.2f}  palm=({rp[0]:>7.1f}, {rp[1]:>7.1f}, {rp[2]:>7.1f})  "
                      f"vel=({rv[0]:>7.1f}, {rv[1]:>7.1f}, {rv[2]:>7.1f})  speed={speed_r:.1f}mm/s")
                print(f"  │        index_tip=({r_index_tip[0]:>7.1f}, {r_index_tip[1]:>7.1f}, {r_index_tip[2]:>7.1f})  "
                      f"middle_tip=({r_mid_tip[0]:>7.1f}, {r_mid_tip[1]:>7.1f}, {r_mid_tip[2]:>7.1f})")
            else:
                print(f"  │ RIGHT  (không phát hiện)")
            print(f"  └─")

    # ── Chạy ──────────────────────────────────────────────────────────────
    mode_label = f"Mock @ {MOCK_FPS} Hz" if USE_MOCK else "Real Leap Motion"
    print(f"\n{'─'*60}")
    print(f"  LMC Capture — Real-time  ({mode_label})")
    print(f"  Warmup: {WARMUP}s → chạy liên tục cho đến Ctrl+C")
    print(f"{'─'*60}")

    if USE_MOCK:
        cap = MockLeapMotionCapture(on_frame_callback=on_frame, fps=MOCK_FPS)
    else:
        cap = LeapMotionCapture(on_frame_callback=on_frame)

    cap.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n\n  ⏹  Ctrl+C — đang dừng capture...")

    cap.stop()

    # ── Kết quả ───────────────────────────────────────────────────────────
    actual_fps = 0.0
    total_elapsed = 0.0
    if len(measure_wall_times) >= 2:
        total_elapsed = measure_wall_times[-1] - measure_wall_times[0]
        actual_fps = (len(measure_wall_times) - 1) / total_elapsed if total_elapsed > 0 else 0.0

    if USE_MOCK:
        min_acceptable_fps = MOCK_FPS * 0.9
    else:
        min_acceptable_fps = 25

    fps_ok = actual_fps >= min_acceptable_fps
    hands_str = "+".join(sorted(hands_seen[0])) if hands_seen[0] else "không phát hiện"

    # In frame cuối cùng
    if last_frame_cache[0] is not None:
        print_frame_table(last_frame_cache[0], "Frame cuối cùng:")

    print(f"\n{'─'*60}  KẾT QUẢ")
    print(f"  Thời gian chạy   : {total_elapsed:.1f}s  (+ {WARMUP}s warmup)")
    print(f"  Warmup frames    : {frame_count - measure_count}")
    print(f"  Measured frames  : {measure_count}")
    print(f"  Total frames     : {frame_count}")
    print(f"  FPS thực tế      : {actual_fps:.1f} Hz  (min: {min_acceptable_fps:.0f} Hz)")
    print(f"  Hands detected   : {hands_str}")
    print(f"  Shape errors     : {shape_errors}")
    print(f"  Dtype errors     : {dtype_errors}")

    if not USE_MOCK:
        if actual_fps >= 100:
            hw_guess = "Leap Motion Controller 2 / Ultraleap 3Di"
        elif actual_fps >= 50:
            hw_guess = "Leap Motion Controller 1 (hoặc LMC2 USB 2.0)"
        else:
            hw_guess = "Leap Motion Controller 1"
        print(f"  Hardware guess   : {hw_guess}")

    all_ok = (shape_errors == 0 and dtype_errors == 0
              and measure_count > 0 and fps_ok)

    if all_ok:
        print(f"\n  ✓  PASS — pipeline real-time ổn định, {actual_fps:.0f} Hz")
    else:
        reasons = []
        if shape_errors > 0:     reasons.append(f"shape errors: {shape_errors}")
        if dtype_errors > 0:     reasons.append(f"dtype errors: {dtype_errors}")
        if measure_count == 0:   reasons.append("không nhận được frame nào")
        if not fps_ok:           reasons.append(f"FPS {actual_fps:.1f} < {min_acceptable_fps}")
        print(f"\n  ✗  FAIL — {'; '.join(reasons)}")
    print(f"{'─'*60}\n")
