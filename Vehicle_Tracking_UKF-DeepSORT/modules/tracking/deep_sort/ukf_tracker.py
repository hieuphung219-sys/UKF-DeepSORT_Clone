import numpy as np
import scipy.linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

# ==========================================
# MÔ HÌNH CONSTANT ACCELERATION (CA) CHO 2D IMAGE
# ==========================================
def fx(state, dt):
    """ Mô hình Constant Acceleration (Gia tốc không đổi) 
        Vector: [x, y, a, h, vx, vy, ax, ay] 
    """
    x, y, a, h, vx, vy, ax, ay = state
    
    # Cập nhật vị trí với gia tốc (giúp bám cua gắt mà không cần biết yaw)
    x_next = x + vx * dt + 0.5 * ax * (dt ** 2)
    y_next = y + vy * dt + 0.5 * ay * (dt ** 2)
    
    a_next = a
    h_next = h # Giả sử chiều cao ít biến đổi đột ngột
    
    # Cập nhật vận tốc
    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    
    return np.array([x_next, y_next, a_next, h_next, vx_next, vy_next, ax, ay])

def hx(state):
    """ Ánh xạ từ 8D xuống 4D (YOLO đo lường) """
    return state[:4]

class UKF_Tracker:
    def __init__(self):
        self.ndim = 8
        self.dt = 1.0
        self.points = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=-5) # kappa=-5 cho 8D là chuẩn tối ưu
        
        # Ma trận Nhiễu quá trình (Q) - Đã fix logic
        # Cho phép gia tốc ax, ay (vị trí 6, 7) có độ nhiễu lớn để xe thoải mái rẽ hướng
        std_Q = [
            0.05, 0.05, 0.1, 0.05,  # Nhiễu vị trí x, y, a, h
            0.1,  0.1,               # Nhiễu vận tốc vx, vy
            1.5,  1.5                # Nhiễu gia tốc ax, ay (Rất quan trọng khi rẽ)
        ]
        self.Q = np.diag(np.square(std_Q))

    def initiate(self, measurement):
        mean = np.array([measurement[0], measurement[1], measurement[2], measurement[3], 0.0, 0.0, 0.0, 0.0])
        
        # Khởi tạo ma trận Hiệp phương sai (P) lớn cho các biến chưa biết
        std_P = [
            2 * 0.05 * measurement[3], # x
            2 * 0.05 * measurement[3], # y
            1e-2,                      # a
            2 * 0.05 * measurement[3], # h
            10 * 0.05 * measurement[3],# vx (Đã sửa lại thành vx)
            10 * 0.05 * measurement[3],# vy (Đã sửa lại thành vy)
            5.0,                       # ax (Cho phép gia tốc thay đổi linh hoạt ban đầu)
            5.0                        # ay
        ]
        covariance = np.diag(np.square(std_P))
        return mean, covariance

    def predict(self, mean, covariance):
        ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=self.dt, fx=fx, hx=hx, points=self.points)
        ukf.x = mean
        ukf.P = covariance
        ukf.Q = self.Q
        
        ukf.predict()
        return ukf.x, ukf.P, ukf.sigmas_f

    def project(self, mean, covariance, height):
        std_R = [0.05 * height, 0.05 * height, 1e-1, 0.1 * height]
        R = np.diag(np.square(std_R))
        mean_projected = hx(mean)
        covariance_projected = covariance[:4, :4] + R
        return mean_projected, covariance_projected

    def update(self, mean, covariance, measurement, predicted_sigma_points=None):
        ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=self.dt, fx=fx, hx=hx, points=self.points)
        ukf.x = mean
        ukf.P = covariance
        if predicted_sigma_points is not None:
            ukf.sigmas_f = predicted_sigma_points

        height = measurement[3]
        std_R = [0.05 * height, 0.05 * height, 1e-1, 0.1 * height]
        ukf.R = np.diag(np.square(std_R))
        
        ukf.update(measurement[:4])
        return ukf.x, ukf.P

    def gating_distance(self, mean, covariance, measurements, height, predicted_sigma_points=None, only_position=False):
        mean_proj, cov_proj = self.project(mean, covariance, height)
        if only_position:
            mean_proj, cov_proj = mean_proj[:2], cov_proj[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)