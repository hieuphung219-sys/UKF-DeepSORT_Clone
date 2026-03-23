import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Đường dẫn để nhận diện module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.tracking.deep_sort.ukf_tracker import UKF_Tracker

# [Task 51] Tạo mock data (Xe chạy thẳng rồi bẻ cua gắt)
frames = 50
true_x, true_y = [], []
for i in range(frames):
    if i < 20: # Chạy thẳng
        x, y = 10 * i, 100  
    elif i < 35: # Bẻ cua gắt
        x = 10 * 19 + 5 * (i - 19)
        y = 100 + 10 * (i - 19)
    else: # Chạy thẳng hướng chéo
        x = true_x[-1] + 5
        y = true_y[-1] + 15
    true_x.append(x)
    true_y.append(y)

# Thêm nhiễu để giả lập BBox bắt được từ YOLO (Measurement noise)
np.random.seed(42)
meas_x = [tx + np.random.normal(0, 5) for tx in true_x]
meas_y = [ty + np.random.normal(0, 5) for ty in true_y]

# [Task 52] Cho UKF chạy độc lập
ukf = UKF_Tracker()
mean, cov = ukf.initiate(np.array([meas_x[0], meas_y[0], 1.5, 40])) # [x, y, a, h]
pred_x, pred_y = [mean[0]], [mean[1]]

for i in range(1, frames):
    # Bước Predict
    mean, cov, _ = ukf.predict(mean, cov)
    pred_x.append(mean[0])
    pred_y.append(mean[1])
    
    # Bước Update
    measurement = np.array([meas_x[i], meas_y[i], 1.5, 40])
    mean, cov = ukf.update(mean, cov, measurement)

# [Task 53] Vẽ đồ thị Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(true_x, true_y, 'g-', label='Quỹ đạo thực tế (Ground Truth)', linewidth=2)
plt.scatter(meas_x, meas_y, c='r', marker='x', label='Tọa độ YOLO đo được (Noisy Measurements)')
plt.plot(pred_x, pred_y, 'b--', label='Quỹ đạo UKF dự đoán (UKF Predicted)', linewidth=2)
plt.title('Đánh giá khả năng bám sát quỹ đạo rẽ cua gắt của UKF')
plt.xlabel('Tọa độ X')
plt.ylabel('Tọa độ Y')
plt.legend()
plt.grid(True)
plt.show()

# [Task 54] Hướng dẫn tinh chỉnh: 
# Nhìn vào đồ thị, nếu quỹ đạo xanh dương bám không sát khi xe bẻ lái, 
# hãy tăng tham số 0.2 tại 'std_Q' trong file ukf_tracker.py lên cao hơn nữa.