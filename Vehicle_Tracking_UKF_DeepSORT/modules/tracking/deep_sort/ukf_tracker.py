import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

# ==========================================
# CÁC HÀM TOÁN HỌC CHO BỘ LỌC UKF
# ==========================================

def fx(state, dt):
    """
    Hàm chuyển trạng thái (State Transition Function).
    Mô phỏng sự di chuyển của phương tiện theo mô hình vận tốc không đổi.
    """
    x, y, a, h, vx, vy, va, vh = state
    
    # Dự đoán vị trí mới (vị trí cũ + vận tốc * thời gian)
    next_x = x + vx * dt
    next_y = y + vy * dt
    next_a = a + va * dt
    next_h = h + vh * dt
    
    # Vận tốc giữ nguyên (Mô hình Constant Velocity)
    next_vx = vx
    next_vy = vy
    next_va = va
    next_vh = vh
    
    return np.array([next_x, next_y, next_a, next_h, next_vx, next_vy, next_va, next_vh])

def hx(state):
    """
    Hàm đo lường (Measurement Function).
    Ánh xạ từ không gian trạng thái 8D ra không gian đo lường 4D của camera (YOLO).
    """
    # Chỉ lấy 4 giá trị đầu: [cx, cy, a, h]
    return state[:4]

def get_measurement_noise(height):
    """
    Tính toán ma trận hiệp phương sai nhiễu đo lường R (Measurement Noise Covariance).
    Nhiễu tỷ lệ thuận với kích thước (chiều cao) của bounding box.
    """
    weight_position = 1.0 / 20
    weight_shape = 1.0 / 10
    
    std = [
        weight_position * height, # Nhiễu của tọa độ tâm x
        weight_position * height, # Nhiễu của tọa độ tâm y
        1e-1,                     # Nhiễu của tỷ lệ khung hình (ít đổi)
        weight_shape * height     # Nhiễu của chiều cao
    ]
    return np.diag(np.square(std))


# ==========================================
# LỚP QUẢN LÝ TRACKER
# ==========================================

class UKF_Tracker:
    def __init__(self, bbox=None):
        """
        Khởi tạo bộ lọc UKF.
        bbox: Bounding box khởi tạo ban đầu [cx, cy, a, h] (nếu có)
        """
        # 1. Khởi tạo thuật toán lấy điểm Sigma
        points = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=-1)
        
        # 2. Khởi tạo đối tượng UKF từ thư viện
        self.ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=1.0, fx=fx, hx=hx, points=points)
        
        # 3. Thiết lập trạng thái và nhiễu ban đầu
        if bbox is not None:
            # Gán trạng thái ban đầu [cx, cy, a, h, 0, 0, 0, 0] (vận tốc ban đầu bằng 0)
            self.ukf.x = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0.0, 0.0, 0.0, 0.0])
            # Gán ma trận nhiễu R theo chiều cao thực tế của bounding box
            self.ukf.R = get_measurement_noise(bbox[3])
        else:
            # Gán giá trị mặc định nếu chưa có bbox
            self.ukf.R = get_measurement_noise(10.0)

    def predict(self):
        """
        Bước 1 của chu trình: Dự đoán trạng thái ở frame tiếp theo.
        Thường được gọi khi chuyển sang frame mới trong video.
        """
        self.ukf.predict()

    def update(self, bbox):
        """
        Bước 2 của chu trình: Cập nhật lại trạng thái bằng dữ liệu thực tế từ YOLO.
        bbox: Bounding box thực tế phát hiện được ở frame hiện tại dạng [cx, cy, a, h].
        """
        # Lấy chiều cao h từ bounding box hiện tại
        height = bbox[3]
        
        # Cập nhật ma trận nhiễu đo lường R linh hoạt theo kích thước xe hiện tại
        self.ukf.R = get_measurement_noise(height)
        
        # Đưa dữ liệu đo lường vào bộ lọc để hiệu chỉnh (innovation)
        self.ukf.update(bbox[:4])
        
    def get_state(self):
        """
        Lấy trạng thái hiện tại của xe để vẽ lên màn hình hoặc phục vụ liên kết dữ liệu.
        """
        return self.ukf.x