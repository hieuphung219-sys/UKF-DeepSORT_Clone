import numpy as np
import scipy.linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ==========================================
# CÁC HÀM TOÁN HỌC CHO BỘ LỌC UKF
# ==========================================

def fx(state, dt):
    """ Hàm chuyển trạng thái CTRV (Constant Turn Rate and Velocity) """
    x, y, a, h, v, yaw, omega, vh = state
    
    # Tránh lỗi chia cho 0 khi xe đi thẳng (tốc độ xoay omega ~ 0)
    if abs(omega) > 1e-4:
        x_next = x + (v / omega) * (np.sin(yaw + omega * dt) - np.sin(yaw))
        y_next = y + (v / omega) * (-np.cos(yaw + omega * dt) + np.cos(yaw))
    else:
        x_next = x + v * np.cos(yaw) * dt
        y_next = y + v * np.sin(yaw) * dt
        
    yaw_next = yaw + omega * dt
    a_next = a
    h_next = h + vh * dt
    
    return np.array([x_next, y_next, a_next, h_next, v, yaw_next, omega, vh])

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
        """ Khởi tạo Track mới với mô hình CTRV """
        # Khởi tạo vector 8D: [x, y, a, h, v=0, yaw=0, omega=0, vh=0]
        mean = np.array([measurement[0], measurement[1], measurement[2], measurement[3], 0.0, 0.0, 0.0, 0.0])
        
        std_P = [
            2 * 0.05 * measurement[3], # x
            2 * 0.05 * measurement[3], # y
            1e-2,                      # a
            2 * 0.05 * measurement[3], # h
            10 * 0.01 * measurement[3],# v (vận tốc)
            np.pi / 4,                 # yaw (bắt đầu có thể lệch 45 độ)
            1e-2,                      # omega (tốc độ xoay)
            10 * 0.01 * measurement[3] # vh
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
        
        if predicted_sigma_points is not None:
            ukf.sigmas_f = predicted_sigma_points

        # [Task 50] Update ma trận R theo BBox thực tế
        height = measurement[3]
        std_R = [0.05 * height, 0.05 * height, 1e-1, 0.1 * height]
        ukf.R = np.diag(np.square(std_R))
        
        ukf.update(measurement[:4])
        print(f"Vận tốc vx, vy sau update: {ukf.x[4]:.2f}, {ukf.x[5]:.2f}")
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