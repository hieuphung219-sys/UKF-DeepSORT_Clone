import os

def read_kitti_tracking_annotation(file_path):
    """
    Đọc và phân tích file annotation (ground truth) của tập dữ liệu KITTI Tracking.
    
    Args:
        file_path (str): Đường dẫn tới file .txt chứa ground truth.
        
    Returns:
        list: Một danh sách chứa các dictionary, mỗi dictionary là thông tin của một đối tượng.
    """
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file tại đường dẫn: {file_path}")
        return []

    annotations = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Tách các giá trị trên một dòng dựa vào khoảng trắng
            data = line.strip().split()
            
            # Bỏ qua các dòng trống (nếu có)
            if len(data) == 0:
                continue
            
            # Cấu trúc của KITTI Tracking:
            # 0: frame, 1: track_id, 2: type, 3: truncated, 4: occluded, 5: alpha
            # 6: bbox_left, 7: bbox_top, 8: bbox_right, 9: bbox_bottom
            # 10-12: dimensions, 13-15: location, 16: rotation_y, 17: score (optional)
            
            annotation_data = {
                'frame': int(data[0]),
                'track_id': int(data[1]),
                'type': data[2],
                # Lưu Bounding Box dưới dạng [left, top, right, bottom]
                'bbox': [float(data[6]), float(data[7]), float(data[8]), float(data[9])]
            }
            
            annotations.append(annotation_data)
            
    print(f"Đã đọc thành công {len(annotations)} dòng dữ liệu từ file.")
    return annotations

# --- Hướng dẫn chạy thử script ---
if __name__ == "__main__":
    # Thay thế bằng đường dẫn thực tế đến một file ground truth đã tải về
    # Ví dụ: "data/kitti/training/label_02/0000.txt"
    sample_file_path = "data/kitty_tracking/training/label_02/0000.txt" 
    
    # Chỉ chạy test nếu file tồn tại để tránh lỗi
    if os.path.exists(sample_file_path):
        parsed_data = read_kitti_tracking_annotation(sample_file_path)
        if parsed_data:
            print("Dữ liệu đối tượng đầu tiên:", parsed_data[10])
else:
        # Thêm thông báo lỗi rõ ràng thay vì im lặng
        print(f"LỖI: Không tìm thấy file test tại: {sample_file_path}")
        print(f"Gợi ý: Hãy chắc chắn bạn đã tạo file 'sample_0000.txt' trong thư mục 'data'.")
        print(f"Thư mục hiện tại mà Terminal đang đứng là: {os.getcwd()}")