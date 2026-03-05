from ultralytics import YOLO

# Khởi tạo mô hình
model = YOLO("yolov8m.pt")

# Đường dẫn đến video 10 giây của bạn
video_path = "kitti_test_set_10s.mp4" 

print("Đang chạy YOLOv8 qua toàn bộ video. Quá trình này sẽ mất một chút thời gian...")

# Chạy dự đoán và lọc chính xác 6 class bạn đã chọn
# Tham số save=True sẽ tự động render một video mới có vẽ sẵn các hộp Bounding Box
results = model.predict(source=video_path, classes=[0, 1, 2, 3, 5, 7], save=True)

print("Hoàn tất! Video kết quả đã được lưu trong thư mục: runs/detect/predict/")