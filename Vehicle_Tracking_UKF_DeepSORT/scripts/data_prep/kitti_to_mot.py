import os
import shutil
import cv2
import argparse
from tqdm import tqdm

# Mapping class từ chuỗi sang số (theo chuẩn MOTChallenge nhưng mở rộng)
# 1: Pedestrian, 2: Person on vehicle, 3: Car, 4: Bicycle, 5: Motorbike, ...
CLASS_MAP = {
    'Pedestrian': 1,
    'Person': 1,
    'Car': 3,
    'Van': 3,
    'Truck': 3,
    'Cyclist': 4,
    'Tram': 3,
    'Misc': 12,
    'DontCare': 0  # 0 thường bị bỏ qua trong evaluation
}

def convert_kitti_to_mot(kitti_root, output_root):
    # Đường dẫn thư mục gốc của KITTI
    image_dir = os.path.join(kitti_root, 'training', 'image_02')
    label_dir = os.path.join(kitti_root, 'training', 'label_02')

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Lỗi: Không tìm thấy thư mục KITTI tại: {kitti_root}")
        print("Cấu trúc mong đợi: training/image_02 và training/label_02")
        return

    # Lấy danh sách các sequence (0000, 0001, ...)
    sequences = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    
    print(f"Tìm thấy {len(sequences)} sequence KITTI. Bắt đầu convert...")

    for seq in sequences:
        print(f"\n--- Đang xử lý Sequence: {seq} ---")
        
        # 1. Tạo thư mục đích: output_root/KITTI-0000/img1 và gt
        target_seq_name = f"KITTI-{seq}"
        target_seq_dir = os.path.join(output_root, target_seq_name)
        target_img_dir = os.path.join(target_seq_dir, 'img1')
        target_gt_dir = os.path.join(target_seq_dir, 'gt')

        if os.path.exists(target_seq_dir):
            shutil.rmtree(target_seq_dir) # Xóa cũ nếu có để làm mới
        
        os.makedirs(target_img_dir)
        os.makedirs(target_gt_dir)

        # ---------------------------------------------------------
        # 2. XỬ LÝ ẢNH (IMAGES)
        # ---------------------------------------------------------
        src_seq_img_dir = os.path.join(image_dir, seq)
        # Lấy danh sách ảnh .png
        images = sorted([f for f in os.listdir(src_seq_img_dir) if f.endswith('.png')])
        
        print(f"-> Convert {len(images)} ảnh sang .jpg...")
        for img_name in tqdm(images):
            # Tên file gốc: 000000.png
            frame_idx = int(os.path.splitext(img_name)[0])
            
            # Tên file mới: 000001.jpg (MOT bắt đầu từ 1)
            new_img_name = f"{frame_idx + 1:06d}.jpg"
            
            src_path = os.path.join(src_seq_img_dir, img_name)
            dst_path = os.path.join(target_img_dir, new_img_name)

            # Đọc và Lưu lại dưới dạng JPG
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)
            else:
                print(f"Cảnh báo: Không đọc được ảnh {src_path}")

        # ---------------------------------------------------------
        # 3. XỬ LÝ NHÃN (LABELS)
        # ---------------------------------------------------------
        print(f"-> Convert annotation...")
        src_label_file = os.path.join(label_dir, f"{seq}.txt")
        dst_gt_file = os.path.join(target_gt_dir, "gt.txt")
        
        if os.path.exists(src_label_file):
            with open(src_label_file, 'r') as f_in, open(dst_gt_file, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split(' ')
                    
                    # Parse KITTI Format
                    # Col 0: frame (0-based)
                    kitti_frame = int(parts[0])
                    # Col 1: track_id
                    track_id = int(parts[1])
                    # Col 2: type (Car, Pedestrian...)
                    obj_type = parts[2]
                    
                    # Lấy bbox (Col 6,7,8,9 -> x1, y1, x2, y2)
                    x1 = float(parts[6])
                    y1 = float(parts[7])
                    x2 = float(parts[8])
                    y2 = float(parts[9])

                    # --- CONVERT LOGIC ---
                    
                    # 1. Frame: 0-based -> 1-based
                    mot_frame = kitti_frame + 1
                    
                    # 2. BBox: (x1,y1,x2,y2) -> (x,y,w,h)
                    width = x2 - x1
                    height = y2 - y1
                    x_topleft = x1
                    y_topleft = y1
                    
                    # 3. Class Mapping
                    class_id = CLASS_MAP.get(obj_type, 12) # Mặc định là 12 (Misc) nếu không khớp
                    
                    # Bỏ qua DontCare nếu muốn (DeepSORT thường không track DontCare)
                    if class_id == 0:
                        continue

                    # 4. Các trường khác
                    conf = 1
                    visibility = 1

                    # Ghi dòng MOT format: 
                    # frame, id, x, y, w, h, conf, class, vis
                    line_out = f"{mot_frame},{track_id},{x_topleft:.2f},{y_topleft:.2f},{width:.2f},{height:.2f},{conf},{class_id},{visibility}\n"
                    f_out.write(line_out)
        else:
            print(f"Cảnh báo: Không tìm thấy file label {src_label_file}")

        # Tạo file seqinfo.ini giả (optional, giúp một số tool visualization)
        with open(os.path.join(target_seq_dir, 'seqinfo.ini'), 'w') as f:
            f.write("[Sequence]\n")
            f.write(f"name=KITTI-{seq}\n")
            f.write(f"imDir=img1\n")
            f.write(f"frameRate=10\n")
            f.write(f"seqLength={len(images)}\n")
            f.write(f"imWidth=1242\n")
            f.write(f"imHeight=375\n")
            f.write(f"imExt=.jpg\n")

    print(f"\nHoàn tất! Dữ liệu đã được lưu tại: {output_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KITTI Tracking dataset to MOT16 format")
    parser.add_argument("--kitti_root", type=str, required=True, help="Đường dẫn đến folder gốc KITTI (chứa training)")
    parser.add_argument("--output_root", type=str, required=True, help="Đường dẫn thư mục đầu ra")
    
    args = parser.parse_args()
    
    convert_kitti_to_mot(args.kitti_root, args.output_root)