# File: UKF-DeepSORT/deep_sort/ukf_tracker.py (hoặc kalman_filter.py)
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

# 1. Khai báo hàm fx (Hàm chuyển trạng thái phi tuyến)
def fx(state, dt):
    x, y, a, h, vx, vy, va, vh = state
    next_x = x + vx * dt
    # ... (nội dung hàm đã viết) ...
    return np.array([next_x, next_y, next_a, next_h, next_vx, next_vy, next_va, next_vh])

# 2. Khai báo hàm hx (Hàm đo lường - Measurement function)
def hx(state):
    # Trả về [x, y, a, h] để đối chiếu với tọa độ từ YOLO
    return state[:4]

# 3. Class quản lý bộ lọc
class UKF_Tracker:
    def __init__(self):
        # Khởi tạo điểm Sigma
        points = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=-1)
        
        # Khởi tạo bộ lọc UKF và truyền trực tiếp hàm fx, hx vào
        self.ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=1.0, fx=fx, hx=hx, points=points)