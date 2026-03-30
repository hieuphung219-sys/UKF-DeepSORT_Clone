import numpy as np
import scipy.linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

def fx(state, dt):
    """
    Hàm mô hình chuyển động (State Transition Function)
    Chuyển trạng thái từ thời điểm t-1 sang t dựa trên mô hình Gia tốc không đổi.
    Không gian trạng thái 8 chiều: [x, y, a, h, vx, vy, ax, ay]
    """
    x, y, a, h, vx, vy, ax, ay = state
    
    new_x = x + vx * dt + 0.5 * ax * dt**2
    new_y = y + vy * dt + 0.5 * ay * dt**2
    new_a = a
    new_h = h
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_ax = ax
    new_ay = ay
    
    return np.array([new_x, new_y, new_a, new_h, new_vx, new_vy, new_ax, new_ay])

def hx(state):
    """
    Hàm đo lường (Measurement Function)
    Trích xuất các thành phần mà camera (YOLO) có thể nhìn thấy: [x, y, a, h]
    """
    return np.array([state[0], state[1], state[2], state[3]])


class UKF_Tracker:
    def __init__(self):
        self.ndim = 8 # Không gian trạng thái
        self.dt = 1.0 # Bước thời gian (frame)
        
        # Thiết lập các điểm Sigma (kappa = 0 để an toàn cho ma trận > 3 chiều)
        self.points = MerweScaledSigmaPoints(n=self.ndim, alpha=0.1, beta=2., kappa=0)
        
        # Ma trận Nhiễu Hệ thống Q (Process Noise)
        std_Q = [
            0.01, 0.01, # x, y 
            0.1, 0.1, # a, h 
            0.05, 0.05, # vx, vy 
            0.5, 0.5    # ax, ay 
        ]
        self.Q = np.diag(np.square(std_Q))

    def initiate(self, measurement):
        """
        Khởi tạo Tracker lần đầu tiên khi YOLO phát hiện xe.
        measurement: [x, y, a, h]
        """
        mean = np.array([
            measurement[0], measurement[1], 
            measurement[2], measurement[3], 
            0.0, 0.0, 
            0.0, 0.0
        ])
        
        # Ma trận Hiệp phương sai ban đầu P
        std_P = [
            2 * 0.05 * measurement[3],  # x
            2 * 0.05 * measurement[3],  # y
            1e-2,                       # a
            2 * 0.05 * measurement[3],  # h
            10 * 0.05 * measurement[3], # vx
            10 * 0.05 * measurement[3], # vy
            0.1,                        # ax
            0.1                         # ay
        ]
        covariance = np.diag(np.square(std_P))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Bước Dự đoán (Prediction)
        """
        # Ép ma trận P luôn Xác định dương để tránh lỗi Cholesky (NaN)
        covariance = (covariance + covariance.T) / 2.0
        covariance += np.eye(self.ndim) * 1e-4

        ukf = UnscentedKalmanFilter(dim_x=self.ndim, dim_z=4, dt=self.dt, fx=fx, hx=hx, points=self.points)
        ukf.x = mean
        ukf.P = covariance
        ukf.Q = self.Q
        
        ukf.predict()
        
        return ukf.x, ukf.P, ukf.sigmas_f

    def update(self, mean, covariance, measurement, predicted_sigma_points=None):
        """
        Bước Cập nhật (Update) với dữ liệu từ YOLO
        """
        covariance = (covariance + covariance.T) / 2.0
        covariance += np.eye(self.ndim) * 1e-4

        ukf = UnscentedKalmanFilter(dim_x=self.ndim, dim_z=4, dt=self.dt, fx=fx, hx=hx, points=self.points)
        ukf.x = mean
        ukf.P = covariance

        # Phục hồi Sigma Points để mở khóa Kalman Gain (K)
        if predicted_sigma_points is not None:
            ukf.sigmas_f = predicted_sigma_points
        else:
            ukf.sigmas_f = ukf.points.sigma_points(ukf.x, ukf.P)

        # Ma trận Nhiễu Đo lường R (Measurement Noise)
        std_R = [
            0.2 * measurement[3] + 5.0, # x
            0.2 * measurement[3] + 5.0, # y
            0.1,                  # a
            0.2 * measurement[3] + 5.0 # h
        ]
        ukf.R = np.diag(np.square(std_R))

        ukf.update(measurement)
        
        return ukf.x, ukf.P

    def project(self, mean, covariance):
        """
        Chiếu không gian trạng thái (8D) xuống không gian đo lường (4D).
        Dùng để tính khoảng cách Mahalanobis trong thuật toán DeepSORT gốc.
        """
        projected_mean = mean[:4].copy()
        projected_cov = covariance[:4, :4].copy()
        
        # Ma trận nhiễu R khi chiếu
        std_R = [
            0.2 * mean[3] + 5.0,
            0.2 * mean[3] + 5.0,
            0.1,
            0.2 * mean[3] + 5.0
        ]
        R = np.diag(np.square(std_R))
        
        return projected_mean, projected_cov + R

    def gating_distance(self, mean, covariance, measurements, height=None, predicted_sigma_points=None, only_position=False):
        """
        Tính bình phương khoảng cách Mahalanobis giữa dự đoán và các Bounding Box.
        Đã cập nhật để đón toàn bộ tham số được truyền từ tracker.py
        """
        # Chiếu từ không gian 8D (UKF) xuống không gian 4D (YOLO)
        mean_proj, covariance_proj = self.project(mean, covariance)
        
        if only_position:
            mean_proj, covariance_proj = mean_proj[:2], covariance_proj[:2, :2]
            measurements = measurements[:, :2]
            
        # Ép ma trận xác định dương
        covariance_proj = (covariance_proj + covariance_proj.T) / 2.0
        covariance_proj += np.eye(covariance_proj.shape[0]) * 1e-4
        
        # Phân tích Cholesky giải hệ phương trình khoảng cách
        cholesky_factor = np.linalg.cholesky(covariance_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        
        return squared_maha