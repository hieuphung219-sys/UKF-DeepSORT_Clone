def extract_bbox_xywh(annotation_line):
    """
    Trích xuất tọa độ Bounding Box (x, y, w, h) từ một dòng annotation của KITTI Tracking.
    """
    parts = annotation_line.strip().split(' ')
    
    # Đảm bảo dòng dữ liệu hợp lệ (KITTI Tracking thường có ít nhất 10 cột)
    if len(parts) >= 10:
        try:
            x_min = float(parts[6])
            y_min = float(parts[7])
            x_max = float(parts[8])
            y_max = float(parts[9])
            
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min
            
            return (x, y, w, h)
        except ValueError:
            print("Lỗi: Không thể ép kiểu dữ liệu.")
            return None
            
    return None

# --- KHỐI KIỂM TRA ĐỘC LẬP (TESTING) ---
if __name__ == "__main__":
    print("Đang chạy kiểm thử hàm extract_bbox_xywh...")
    
    # 1. Tạo một dòng dữ liệu giả lập (mock data) trích xuất từ KITTI
    # Định dạng: frame track_id type truncated occluded alpha xmin ymin xmax ymax ...
    sample_kitti_line = "0 -1 DontCare -1 -1 -10.000000 219.310000 188.490000 245.500000 218.560000 -1000.000000 -1000.000000 -1000.000000 -10.000000 -1.000000 -1.000000 -1.000000"
    
    # 2. Gọi hàm
    result = extract_bbox_xywh(sample_kitti_line)
    
    # 3. In kết quả và đối chiếu
    print(f"Dòng annotation đầu vào:\n{sample_kitti_line}")
    if result:
        x, y, w, h = result
        print("\n[THÀNH CÔNG] Trích xuất tọa độ:")
        print(f"- x (Góc trái): {x:.2f}")
        print(f"- y (Góc trên): {y:.2f}")
        print(f"- w (Chiều rộng): {w:.2f}")
        print(f"- h (Chiều cao): {h:.2f}")
    else:
        print("\n[THẤT BẠI] Không thể trích xuất dữ liệu.")