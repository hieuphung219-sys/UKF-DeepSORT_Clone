# vim: expandtab:ts=4:sw=4
import numpy as np


class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        
        # [UKF Specific] Thêm thuộc tính lưu Sigma Points
        self.predicted_sigma_points = None

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        # [UKF Specific] Nhận 3 giá trị trả về
        self.mean, self.covariance, self.predicted_sigma_points = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        # [UKF Specific] Cập nhật tọa độ
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah(), self.predicted_sigma_points)
        
        # --- THÊM LOGIC FEATURE EMA TẠI ĐÂY ---
        alpha = 0.9 # Hệ số làm mượt
        if len(self.features) > 0:
            # Lấy feature hiện tại của detection và feature cũ nhất của track
            current_feat = detection.feature
            old_feat = self.features[-1]
            
            # Cập nhật theo công thức Trung bình động hàm mũ
            smoothed_feat = alpha * current_feat + (1 - alpha) * old_feat
            
            # Chuẩn hóa L2 (Bắt buộc để Cosine Distance hoạt động đúng)
            smoothed_feat /= np.linalg.norm(smoothed_feat)
            self.features.append(smoothed_feat)
        else:
            self.features.append(detection.feature)
        # --------------------------------------

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
    
    def camera_update(self, H):
        """ Bù trừ chuyển động camera bằng ma trận Affine H (2x3) """
        x, y = self.mean[0], self.mean[1]
        
        # Dịch chuyển tọa độ (x, y) theo ma trận chuyển đổi bối cảnh
        new_x = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        new_y = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        
        self.mean[0] = new_x
        self.mean[1] = new_y