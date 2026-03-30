import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Thêm đường dẫn gốc của project vào sys.path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import class UKF_Tracker mà chúng ta vừa viết lại
from modules.tracking.deep_sort.ukf_tracker import UKF_Tracker

def generate_dummy_trajectory(num_frames=60):
    """
    Tạo ra một quỹ đạo giả lập: Xe đi thẳng, sau đó bẻ cua gắt 90 độ.
    Trả về:
        - ground_truth: Tọa độ thực tế (x, y) lý tưởng.
        - measurements: Tọa độ (x, y) đã bị cộng thêm nhiễu (giả lập YOLO).
    """
    ground_truth = []
    measurements = []
    
    # Giả lập nhiễu của YOLO (Độ lệch chuẩn khoảng 3 pixel)
    measurement_noise_std = 3.0 
    
    x, y = 10.0, 10.0
    vx, vy = 5.0, 0.0 # Bắt đầu bằng việc đi thẳng theo trục X
    
    for i in range(num_frames):
        # Từ frame 20 đến 30: Bẻ lái gắt 90 độ sang trục Y
        if 20 <= i <= 30:
            vx -= 0.5  # Giảm tốc trục X
            vy += 0.5  # Tăng tốc trục Y
            
        x += vx
        y += vy
        
        ground_truth.append((x, y))
        
        # Thêm nhiễu Gaussian để giả lập Bounding Box bị rung lắc
        noisy_x = x + np.random.normal(0, measurement_noise_std)
        noisy_y = y + np.random.normal(0, measurement_noise_std)
        
        # Measurement format của hệ thống: [x, y, aspect_ratio, height]
        # Giả sử aspect_ratio = 1.5, height = 40 (không đổi)
        measurements.append(np.array([noisy_x, noisy_y, 1.5, 40.0]))
        
    return ground_truth, measurements

def main():
    print("1. Đang tạo dữ liệu giả lập (Ground Truth & Measurements)...")
    ground_truth, measurements = generate_dummy_trajectory(num_frames=60)
    
    print("2. Đang khởi tạo UKF Tracker...")
    tracker = UKF_Tracker()
    
    # Khởi tạo vector trạng thái (mean) và ma trận hiệp phương sai (cov) ở frame đầu tiên
    mean, cov = tracker.initiate(measurements[0])
    
    predicted_trajectory = []
    # Lưu lại vị trí khởi tạo
    predicted_trajectory.append((mean[0], mean[1]))
    
    print("3. Bắt đầu chạy vòng lặp Tracking (Predict -> Update)...")
    # Chạy từ frame thứ 2 trở đi
    for i in range(1, len(measurements)):
        measurement = measurements[i]
        
        # --- BƯỚC 1: PREDICT (Dự đoán) ---
        # Lấy ra sigmas_f để duy trì dòng chảy thông tin (Tránh lỗi Kalman Gain = 0)
        mean, cov, sigmas_f = tracker.predict(mean, cov)
        
        # Lưu lại tọa độ ngay sau khi predict (để xem quán tính nó văng đi đâu)
        # predicted_trajectory.append((mean[0], mean[1])) 
        
        # --- BƯỚC 2: UPDATE (Cập nhật) ---
        # Truyền sigmas_f vào để sửa quỹ đạo dựa trên measurement thực tế
        mean, cov = tracker.update(mean, cov, measurement, predicted_sigma_points=sigmas_f)
        
        # Lưu lại tọa độ sau khi đã được tinh chỉnh (Filtered state)
        predicted_trajectory.append((mean[0], mean[1]))
        
        # In log ra màn hình để kiểm tra (không còn bị dính 1 số như trước)
        print(f"Frame {i:02d}: YOLO = ({measurement[0]:.1f}, {measurement[1]:.1f}) | UKF = ({mean[0]:.1f}, {mean[1]:.1f})")

    print("4. Đang vẽ đồ thị...")
    # Tách tọa độ để vẽ
    gt_x, gt_y = zip(*ground_truth)
    meas_x = [m[0] for m in measurements]
    meas_y = [m[1] for m in measurements]
    pred_x, pred_y = zip(*predicted_trajectory)

    plt.figure(figsize=(10, 8))
    
    # Vẽ quỹ đạo thực tế (Đường chuẩn)
    plt.plot(gt_x, gt_y, label='Thực tế (Ground Truth)', color='green', linewidth=2, alpha=0.6)
    
    # Vẽ các điểm đo lường bị nhiễu (YOLO giả lập)
    plt.scatter(meas_x, meas_y, label='Đo lường (Noisy YOLO)', color='red', marker='x', s=30)
    
    # Vẽ quỹ đạo do UKF lọc ra
    plt.plot(pred_x, pred_y, label='UKF Lọc (Filtered)', color='blue', linestyle='--', linewidth=2)
    
    # Đánh dấu điểm bắt đầu
    plt.scatter(gt_x[0], gt_y[0], color='black', marker='o', s=100, label='Điểm xuất phát')

    plt.title('Đánh giá khả năng bám cua của Unscented Kalman Filter')
    plt.xlabel('Tọa độ X')
    plt.ylabel('Tọa độ Y')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal') # Đảm bảo tỷ lệ trục x và y bằng nhau để nhìn rõ góc cua
    
    plt.show()

if __name__ == "__main__":
    main()