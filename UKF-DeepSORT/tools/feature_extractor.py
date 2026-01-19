import numpy as np

class FeatureExtractor:
    def __init__(self, model_path):
        self.model_path = model_path
        # Không tải model .tflite lỗi nữa
        print("LƯU Ý: Đang chạy chế độ 'Trí nhớ giả' (SORT) để tránh lỗi FlexDelegate.")
        
        # Kích thước chuẩn mà DeepSORT yêu cầu (128 chiều)
        self.feature_dim = 128

    def extract_feature(self, img):
        # Thay vì tốn thời gian chạy mô hình AI, ta trả về vector ngẫu nhiên hoặc số 0
        # DeepSORT sẽ dựa hoàn toàn vào vị trí (Kalman Filter) để theo dõi
        # Đây là cách biến DeepSORT thành SORT
        return np.ones(self.feature_dim) # Trả về vector toàn số 1