# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick

import numpy as np
import cv2

def extract_image_patches(image, bboxes, patch_shape=(256, 128)):
    """
    Cắt và resize ảnh dùng kỹ thuật Letterbox (chống méo ảnh, bù viền xám).
    patch_shape: (Height, Width)
    """
    img_height, img_width = image.shape[:2]
    patches = []
    
    # Chiều cao và chiều rộng mục tiêu
    target_h, target_w = patch_shape 

    for bbox in bboxes:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(img_width, x + w)
        y_max = min(img_height, y + h)
        
        patch = image[y_min:y_max, x_min:x_max]
        
        if patch.size > 0:
            # 1. Tính toán tỷ lệ thu phóng (Scale) sao cho ảnh không bị méo
            h_orig, w_orig = patch.shape[:2]
            scale = min(target_w / w_orig, target_h / h_orig)
            
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            
            # 2. Resize ảnh gốc theo tỷ lệ vừa tính
            patch_resized = cv2.resize(patch, (new_w, new_h))
            
            # 3. Tạo một bức ảnh nền màu XÁM (128, 128, 128) với kích thước chuẩn 256x128
            letterbox_patch = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
            
            # 4. Tính toán tọa độ để dán bức ảnh đã resize vào chính giữa nền xám
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            # 5. Dán ảnh vào nền xám (Phần còn dư mặc nhiên là màu xám)
            letterbox_patch[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = patch_resized
            
            patches.append(letterbox_patch)
        else:
            print(f"Cảnh báo: Bounding box không hợp lệ {bbox}")

    return patches

if __name__ == "__main__":
    # ĐOẠN CODE TEST NẰM TRONG NÀY, PHẢI LÙI VÀO 1 TAB
    print("--- Chạy thử nghiệm hàm extract_image_patches ---")
    
    # Tạo một bức ảnh giả lập màu đen kích thước 500x500
    dummy_image = np.zeros((500, 500, 3), dtype=np.uint8) 

    # Vẽ một hình chữ nhật màu trắng để giả lập xe cộ
    cv2.rectangle(dummy_image, (100, 100), (200, 250), (255, 255, 255), -1)

    # Giả lập output của Detector [x, y, w, h]
    dummy_bboxes = [
        [100, 100, 100, 150], 
        [-10, 50, 60, 80]     
    ]

    # Gọi hàm
    cropped_patches = extract_image_patches(dummy_image, dummy_bboxes)

    # Kiểm tra kết quả
    print(f"Số lượng ảnh cắt được: {len(cropped_patches)}")
    for i, p in enumerate(cropped_patches):
        print(f"Kích thước patch {i}: {p.shape}")