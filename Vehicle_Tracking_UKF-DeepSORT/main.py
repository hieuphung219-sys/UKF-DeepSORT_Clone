import cv2
import argparse

# Import các modules tự xây dựng
from modules.detection.vehicle_detector import VehicleDetector
from modules.reid.feature_extractor import PyTorchFeatureExtractor
from modules.tracking.application_util.preprocessing import extract_image_patches
from modules.tracking.deep_sort.detection import Detection
from modules.tracking.deep_sort.tracker import Tracker
from modules.tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from modules.visualization.visualizer import Visualizer

# Import class CMC vừa tạo
from modules.tracking.application_util.cmc import CameraMotionCompensator 

def run_pipeline(video_path, yolo_weights, reid_weights, output_path, txt_output):
    print("1. Đang tải mô hình YOLOv8...")
    # Lọc nhiễu (Confidence threshold, kích thước) đã được xử lý bên trong VehicleDetector
    detector = VehicleDetector(model_path=yolo_weights, conf_thresh=0.7)

    print("2. Đang tải mô hình Re-ID (VeRi)...")
    extractor = PyTorchFeatureExtractor(model_path=reid_weights)

    print("3. Đang khởi tạo UKF-DeepSORT (CTRV Model)...")
    metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.5, budget=100)
    tracker = Tracker(metric, max_age=30, n_init=3, lambda_weight=0.5)

    print("4. Đang khởi tạo Camera Motion Compensator...")
    cmc = CameraMotionCompensator()

    print("5. Đang khởi tạo Visualizer...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 
    visualizer = Visualizer(output_path=output_path, fps=fps)

    # Khởi tạo file text để ghi kết quả đánh giá (Evaluation)
    f_out = open(txt_output, "w")
    frame_idx = 0

    print("\n[BẮT ĐẦU CHẠY PIPELINE] - Hệ thống End-to-End Tracking")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # --- GIAI ĐOẠN 1: CAMERA MOTION COMPENSATION (Gọn gàng & Thuần OOP) ---
        H = cmc.compute_affine_matrix(frame)
        if H is not None:
            tracker.camera_update(H)
        
        # --- GIAI ĐOẠN 2: YOLO DETECTION ---
        raw_detections = detector.detect(frame)

        # --- GIAI ĐOẠN 3: RE-ID FEATURE EXTRACTION ---
        bboxes = [det[:4] for det in raw_detections]
        patches = extract_image_patches(frame, bboxes, patch_shape=(256, 128))
        
        deepsort_detections = []
        for i, patch in enumerate(patches):
            feature_vector = extractor.extract_feature(patch)
            bbox = raw_detections[i][:4]
            conf = raw_detections[i][4]
            deepsort_detections.append(Detection(bbox, conf, feature_vector))
            
        # --- GIAI ĐOẠN 4: UKF-DEEPSORT TRACKING ---
        tracker.predict()
        tracker.update(deepsort_detections)
        
        # --- GIAI ĐOẠN 5: GHI FILE TEXT & VISUALIZATION ---
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h = track.to_tlwh()
            
            # Ghi ra file .txt chuẩn MOT
            f_out.write(f"{frame_idx},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        
        vis_frame = visualizer.draw_and_save(frame, tracker.tracks, frame_idx)

    # Đóng file text và giải phóng bộ nhớ
    f_out.close()
    cap.release()
    cv2.destroyAllWindows()
    visualizer.release()
    
    # In ra tổng số ID sinh ra để kiểm tra mức độ IDSW
    if hasattr(tracker, '_next_id'):
        print(f"\n[HOÀN TẤT] Tổng số ID độc nhất đã được sinh ra: {tracker._next_id - 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Vehicle Tracking Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến video test")
    parser.add_argument("--yolo", type=str, default="yolov8m.pt", help="Đường dẫn file weights YOLO")
    parser.add_argument("--reid", type=str, required=True, help="Đường dẫn file weights Re-ID (.pth)")
    parser.add_argument("--output", type=str, default="output_test.mp4", help="Tên file video đầu ra")
    parser.add_argument("--txt_output", type=str, default="results.txt", help="Tên file text chứa tọa độ chuẩn MOT")
    
    args = parser.parse_args()
    run_pipeline(args.video, args.yolo, args.reid, args.output, args.txt_output)