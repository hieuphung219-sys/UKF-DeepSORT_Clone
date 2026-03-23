import numpy as np
import scipy.linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ==========================================
# CÁC HÀM TOÁN HỌC CHO BỘ LỌC UKF
# ==========================================

def fx(state, dt):
    """ Hàm chuyển trạng thái (Non-linear State Transition) """
    x, y, a, h, vx, vy, va, vh = state
    return np.array([x + vx*dt, y + vy*dt, a + va*dt, h + vh*dt, vx, vy, va, vh])

def hx(state):
    """ Hàm đo lường ánh xạ từ không gian trạng thái 8D ra 4D (YOLO) """
    return state[:4]

# ==========================================
# CLASS UKF_TRACKER (WRAPPER TƯƠNG THÍCH DEEPSORT)
# ==========================================

class UKF_Tracker:
    def __init__(self):
        self.ndim = 8
        self.dt = 1.0
        # [Task 45] Khởi tạo thuật toán sinh điểm Sigma theo Merwe
        self.points = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=-1)
        
        # [Task 49] Thiết lập ma trận Nhiễu quá trình (Q)
        # Trọng số của vận tốc vx, vy được đặt cao hơn (0.2) vì xe cộ thay đổi gia tốc/hướng đi rất nhanh
        std_Q = [
            0.05, 0.05, 0.01, 0.05,  # Nhiễu vị trí x, y, a, h
            0.2, 0.2, 0.01, 0.05     # Nhiễu vận tốc vx, vy, va, vh
        ]
        self.Q = np.diag(np.square(std_Q))

    def initiate(self, measurement):
        """ Khởi tạo Track mới """
        mean = np.r_[measurement, np.zeros(4)]
        
        # [Task 48] Khởi tạo ma trận Hiệp phương sai sai số (P)
        std_P = [
            2 * 0.05 * measurement[3], 2 * 0.05 * measurement[3], 1e-2, 2 * 0.05 * measurement[3],
            10 * 0.01 * measurement[3], 10 * 0.01 * measurement[3], 1e-5, 10 * 0.01 * measurement[3]
        ]
        covariance = np.diag(np.square(std_P))
        return mean, covariance

    def predict(self, mean, covariance):
        """ [Task 46] Lập trình bước Dự đoán (Predict step) """
        ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=self.dt, fx=fx, hx=hx, points=self.points)
        ukf.x = mean
        ukf.P = covariance
        ukf.Q = self.Q
        
        ukf.predict()
        # Trả về thêm sigmas_f để duy trì cấu trúc API với DeepSORT
        return ukf.x, ukf.P, ukf.sigmas_f

    def project(self, mean, covariance, height):
        """ Ánh xạ về không gian đo lường để tính khoảng cách Mahalanobis """
        # [Task 50] Thiết lập ma trận Nhiễu đo lường (R) dựa trên chiều cao bbox (YOLO)
        std_R = [0.05 * height, 0.05 * height, 1e-1, 0.1 * height]
        R = np.diag(np.square(std_R))
        
        mean_projected = hx(mean)
        # Xấp xỉ chiếu hiệp phương sai P xuống không gian 4D
        covariance_projected = covariance[:4, :4] + R
        return mean_projected, covariance_projected

    def update(self, mean, covariance, measurement, predicted_sigma_points=None):
        """ [Task 47] Lập trình bước Cập nhật (Update step) """
        ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=self.dt, fx=fx, hx=hx, points=self.points)
        ukf.x = mean
        ukf.P = covariance
        
        # [Task 50] Update ma trận R theo BBox thực tế
        height = measurement[3]
        std_R = [0.05 * height, 0.05 * height, 1e-1, 0.1 * height]
        ukf.R = np.diag(np.square(std_R))
        
        ukf.update(measurement[:4])
        return ukf.x, ukf.P

    def gating_distance(self, mean, covariance, measurements, height, predicted_sigma_points=None, only_position=False):
        """ Loại bỏ các BBox quá vô lý (Tính khoảng cách Mahalanobis) """
        mean_proj, cov_proj = self.project(mean, covariance, height)
        if only_position:
            mean_proj, cov_proj = mean_proj[:2], cov_proj[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)