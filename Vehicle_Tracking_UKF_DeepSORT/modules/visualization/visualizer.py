import cv2
import time
import numpy as np

class Visualizer:
    def __init__(self, output_path: str = "demo_result.avi", fps: int = 30):
        """
        Class chuyên trách việc vẽ Bounding Box, ID và xuất video.
        Tuân thủ Single Responsibility Principle.
        """
        self.output_path = output_path
        self.target_fps = fps
        self.writer = None
        self.last_time = time.time()
        
        # Cache màu sắc để tăng tốc độ, không phải băm (hash) lại liên tục
        self.color_cache = {}

    def _get_color(self, track_id: int) -> tuple:
        if track_id not in self.color_cache:
            idx = track_id * 3
            self.color_cache[track_id] = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return self.color_cache[track_id]

    def draw_and_save(self, frame: np.ndarray, tracks: list, frame_idx: int) -> np.ndarray:
        """
        Tính FPS, vẽ Bounding Box, hiển thị ID và ghi frame vào video.
        """
        # 1. Tính System FPS
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        fps_sys = 1.0 / delta_time if delta_time > 0 else 0
        
        print(f"Processing frame {frame_idx:05d} | System FPS: {fps_sys:.2f}")

        # 2. Khởi tạo VideoWriter (chỉ chạy 1 lần ở frame đầu tiên)
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.target_fps, (w, h))

        # 3. Vẽ dữ liệu của từng track lên frame
        for track in tracks:
            # Chỉ vẽ những track đã được xác nhận và mới được cập nhật
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            x, y, w_box, h_box = map(int, track.to_tlwh())
            color = self._get_color(track.track_id)
            label = f"ID: {track.track_id}"

            # Vẽ Box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Vẽ Label nền
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - 20), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. Ghi frame vào video
        self.writer.write(frame)
        return frame

    def release(self):
        """Giải phóng tài nguyên VideoWriter khi kết thúc."""
        if self.writer is not None:
            self.writer.release()
            print(f"Xong! Video đã được lưu tại: {self.output_path}")