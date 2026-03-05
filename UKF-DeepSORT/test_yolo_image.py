import cv2
from ultralytics import YOLO

# 1. Khởi tạo mô hình với file weights bạn vừa tải
model = YOLO("yolov8m.pt")

# 2. Đường dẫn đến file video 10 giây của bạn 
# (Hãy thay đổi tên file dưới đây cho khớp với file thực tế của bạn)
video_path = "kitti_test_set_10s.mp4" 
cap = cv2.VideoCapture(video_path)

# Đọc frame đầu tiên của video
ret, frame = cap.read()

if ret:
    print("Đã cắt thành công 1 ảnh tĩnh từ video. Đang tiến hành nhận diện...")
    
    # 3. Chạy YOLO dự đoán trên ảnh tĩnh này
    # Tham số save=True sẽ tự động vẽ Bounding Box và lưu thành file ảnh mới
    # Tham số show=True sẽ hiển thị ảnh kết quả lên màn hình
    results = model.predict(source=frame, save=True, show=True)
    
    print("Hoàn tất! Hãy kiểm tra thư mục 'runs/detect/predict' trong dự án để xem ảnh kết quả.")
else:
    print("Lỗi: Không thể đọc được video. Hãy kiểm tra lại đường dẫn video_path.")

cap.release()
cv2.destroyAllWindows()