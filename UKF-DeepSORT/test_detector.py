import cv2
from vehicle_detector import VehicleDetector

# 1. Khởi tạo class
detector = VehicleDetector(model_path="yolov8m.pt", conf_thresh=0.5)

# 2. Đọc thử video 10s của bạn
cap = cv2.VideoCapture("kitti_test_set_10s.mp4")
ret, frame = cap.read() # Đọc 1 frame đầu tiên

if ret:
    # 3. Đưa frame qua Detector
    bboxes = detector.detect(frame)
    print(f"Phát hiện được {len(bboxes)} đối tượng trong frame này!")
    
    # --- PHẦN THÊM MỚI: VẼ BOUNDING BOX LÊN ẢNH ĐỂ KIỂM CHỨNG ---
    # Từ điển map class_id sang tên gọi (dựa trên class của YOLO/COCO)
    class_names = {0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    
    for box in bboxes:
        # Lấy các giá trị từ mảng
        x, y, w, h, conf, cls_id = box
        
        # Chuyển tọa độ và ID sang số nguyên để vẽ
        x, y, w, h = int(x), int(y), int(w), int(h)
        cls_id = int(cls_id)
        
        # Chọn màu: Ô tô (Xanh lục), Người (Đỏ), Xe đạp (Xanh lơ)
        color = (0, 255, 0)
        if cls_id == 0: color = (0, 0, 255)
        elif cls_id == 1: color = (255, 255, 0)
        
        # Vẽ khung hình chữ nhật
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Ghi tên class và điểm tin cậy lên trên khung
        label = f"{class_names.get(cls_id, 'Unknown')} {conf:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Lưu ảnh kết quả ra file
    output_image = "test_detector_result.jpg"
    cv2.imwrite(output_image, frame)
    print(f"Đã lưu ảnh kiểm chứng vào file '{output_image}'. Hãy mở ảnh lên để xem!")
    # -------------------------------------------------------------

else:
    print("Không đọc được video!")
    
cap.release()