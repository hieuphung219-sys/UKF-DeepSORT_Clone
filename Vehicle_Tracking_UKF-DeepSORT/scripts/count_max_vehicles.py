from collections import Counter

def count_max_objects_except_dontcare(annotation_file_path):
    # List lưu trữ frame_id của các đối tượng hợp lệ
    valid_frames = []

    try:
        with open(annotation_file_path, 'r') as file:
            for line in file:
                data = line.strip().split(' ')
                
                # Bỏ qua dòng trống hoặc lỗi định dạng
                if len(data) < 3:
                    continue
                
                frame_id = int(data[0])
                obj_type = data[2] # Lấy nhãn đối tượng
                
                # CHỈ ĐẾM NẾU NHÃN KHÁC 'DontCare'
                if obj_type != 'DontCare':
                    valid_frames.append(frame_id)

        # Đếm số lần xuất hiện của mỗi frame_id
        frame_counts = Counter(valid_frames)

        if not frame_counts:
            print("Không tìm thấy đối tượng hợp lệ nào trong file.")
            return

        # Tìm frame có tổng số đối tượng lớn nhất
        max_frame, max_count = frame_counts.most_common(1)[0]

        print(f"Khung hình đông đúc nhất (đã loại trừ 'DontCare') là frame thứ: {max_frame}")
        print(f"Tổng số lượng đối tượng xuất hiện cùng lúc là: {max_count} đối tượng")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {annotation_file_path}")

# Nhớ kiểm tra lại đường dẫn file cho chính xác với máy của bạn nhé
file_path = "data/kitti_tracking/training/label_02/0000.txt" 
count_max_objects_except_dontcare(file_path)