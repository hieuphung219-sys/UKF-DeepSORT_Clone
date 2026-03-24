import cv2
import argparse
import numpy as np

# Import các modules tự xây dựng
from modules.detection.vehicle_detector import VehicleDetector
from modules.reid.feature_extractor import PyTorchFeatureExtractor
from modules.tracking.application_util.preprocessing import extract_image_patches
from modules.tracking.deep_sort.detection import Detection
from modules.tracking.deep_sort.tracker import Tracker
from modules.tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from modules.visualization.visualizer import Visualizer

def run_pipeline(video_path, yolo_weights, reid_weights, output_path):
    print("1. Đang tải mô hình YOLOv8...")
    detector = VehicleDetector(model_path=yolo_weights, conf_thresh=0.5)

    print("2. Đang tải mô hình Re-ID (OSNet/ResNet50)...")
    extractor = PyTorchFeatureExtractor(model_path=reid_weights)

    print("3. Đang khởi tạo UKF-DeepSORT...")
    metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.2, budget=100)
    tracker = Tracker(metric, max_age=30, n_init=3, lambda_weight=0.5)

    print("4. Đang khởi tạo Visualizer...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Fallback nếu video không đọc được FPS
    visualizer = Visualizer(output_path=output_path, fps=fps)

    frame_idx = 0
    print("\n[BẮT ĐẦU CHẠY PIPELINE]")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # --- GIAI ĐOẠN 2: YOLO DETECTION ---
        # detections trả về dạng [x_min, y_min, w, h, conf, class_id]
        raw_detections = detector.detect(frame)
        
        filtered_detections = []
        for det in raw_detections:
            x, y, w, h, conf, cls_id = det
            # Chỉ giữ lại bbox nếu chiều rộng > 20 và chiều cao > 20 pixel (có thể tự chỉnh số này)
            if w > 20 and h > 20: 
                filtered_detections.append(det)

        raw_detections = filtered_detections

        # --- GIAI ĐOẠN 3: RE-ID FEATURE EXTRACTION ---
        # Lọc ra tọa độ bbox để cắt ảnh
        bboxes = [det[:4] for det in raw_detections]
        
        # Cắt và resize bằng hàm extract_image_patches (Letterbox)
        patches = extract_image_patches(frame, bboxes, patch_shape=(256, 128))
        
        deepsort_detections = []
        for i, patch in enumerate(patches):
            # Trích xuất vector đặc trưng 2048 chiều
            feature_vector = extractor.extract_feature(patch)
            
            # Đóng gói thành đối tượng Detection chuẩn của DeepSORT
            bbox = raw_detections[i][:4]
            conf = raw_detections[i][4]
            deepsort_detections.append(Detection(bbox, conf, feature_vector))
            
        # --- GIAI ĐOẠN 5: UKF-DEEPSORT TRACKING ---
        tracker.predict()
        tracker.update(deepsort_detections)
        
        # --- GIAI ĐOẠN 6: VISUALIZATION ---
        vis_frame = visualizer.draw_and_save(frame, tracker.tracks, frame_idx)
        
        # Mở comment 3 dòng dưới đây nếu bạn muốn xem trực tiếp (popup) khi chạy trên local
        # cv2.imshow("UKF-DeepSORT Pipeline", vis_frame)
        # if cv2.waitKey(1) & 0xFF == 27: # Nhấn ESC để thoát sớm
        #     break

    cap.release()
    cv2.destroyAllWindows()
    visualizer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Vehicle Tracking Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến video test 10 giây")
    parser.add_argument("--yolo", type=str, default="yolov8m.pt", help="Đường dẫn file weights YOLO")
    parser.add_argument("--reid", type=str, required=True, help="Đường dẫn file weights Re-ID (.pth)")
    parser.add_argument("--output", type=str, default="output_test.mp4", help="Tên file video đầu ra")
    
    args = parser.parse_args()
    run_pipeline(args.video, args.yolo, args.reid, args.output)