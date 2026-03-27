# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from .ukf_tracker import UKF_Tracker # Đã tích hợp từ Giai đoạn 4
from . import linear_assignment
from . import iou_matching
from .track import Track

class Tracker:
    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, lambda_weight=0.5):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        
        # [Task 69] Định nghĩa MAX_AGE: Số frame duy trì track 'Lost' trước khi xóa
        self.max_age = max_age
        # [Task 70] Định nghĩa N_INIT: Số frame liên tiếp để chuyển từ 'Tentative' sang 'Confirmed'
        self.n_init = n_init
        
        # [Task 62] Tham số lambda để trọng số hóa Mahalanobis và Cosine
        self.lambda_weight = lambda_weight

        # Khởi tạo bộ lọc UKF
        self.kf = UKF_Tracker()
        
        self.tracks = []
        self._next_id = 1

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        # Chạy logic liên kết dữ liệu
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # [Task 66] Logic cập nhật: Các cặp ghép thành công sẽ kích hoạt update của UKF
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
                
        # [Task 68] Logic mất dấu: Track không được ghép cặp sẽ bị đánh dấu missed (chuyển sang Lost)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        # [Task 67] Logic sinh mới: Detection không ghép cặp sẽ tạo Track mới (Tentative)
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
            
        # [Task 71] Quản lý bộ nhớ: Tự động xóa vĩnh viễn các Track vượt quá MAX_AGE
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Cập nhật thư viện đặc trưng hình ảnh cho thuật toán Cosine
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # [Task 59] Lập trình Ma trận Chi phí (Cost Matrix)
        def combined_gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # [Task 61] 1. Tích hợp khoảng cách Cosine (Ngoại hình)
            appearance_cost = self.metric.distance(features, targets)
            
            # [Task 60] 2. Tính toán khoảng cách Mahalanobis (Động học từ UKF)
            motion_cost = np.zeros_like(appearance_cost)
            measurements = np.asarray([dets[i].to_xyah() for i in detection_indices])
            for row, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                
                # Trích xuất bình phương khoảng cách Mahalanobis
                maha_dist = self.kf.gating_distance(
                    track.mean, track.covariance, measurements, track.mean[3], track.predicted_sigma_points)
                
                # Chuẩn hóa Mahalanobis (Ngưỡng Chi-square 95% cho 4 bậc tự do là 9.4877)
                # Việc chia cho 9.4877 giúp đưa Mahalanobis về xấp xỉ khoảng [0, 1] 
                # để có thể cộng tỷ lệ công bằng với khoảng cách Cosine.
                motion_cost[row, :] = maha_dist / 9.4877 

            # [Task 62] 3. Viết hàm tổng hợp: Trọng số hóa bằng tham số Lambda
            cost_matrix = self.lambda_weight * motion_cost + (1 - self.lambda_weight) * appearance_cost
            
            # [Task 63] 4. Rào cản cổng (Gating Mechanism) bằng ngưỡng Chi-square
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices, only_position=True)
            return cost_matrix

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # [Task 65] Cơ chế Matching Cascade: Ưu tiên phân bổ cho Track vừa xuất hiện
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                combined_gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        MAX_IOU_WAIT = 3 
        
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update <= MAX_IOU_WAIT]
            
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update > MAX_IOU_WAIT]
            
        # So khớp những phần còn lại bằng IoU (Intersection over Union)
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # Khởi tạo ma trận Mean và Covariance ban đầu từ UKF
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def camera_update(self, H):
        for track in self.tracks:
            track.camera_update(H)