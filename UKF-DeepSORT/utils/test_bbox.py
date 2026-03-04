import cv2

def draw_green_bounding_box(image_path, x, y, w, h, output_path="output_test.jpg"):
    """
    Đọc ảnh gốc, vẽ bounding box màu xanh lá cây và lưu/hiển thị ảnh.
    """
    # Đọc ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh từ {image_path}. Hãy kiểm tra lại đường dẫn.")
        return

    # Tính toán tọa độ điểm góc dưới cùng bên phải
    # OpenCV yêu cầu điểm góc trên-trái (x_min, y_min) và góc dưới-phải (x_max, y_max)
    x_min, y_min = int(x), int(y)
    x_max = int(x + w)
    y_max = int(y + h)

    # Màu xanh lá cây trong OpenCV tuân theo chuẩn BGR: (Blue=0, Green=255, Red=0)
    color = (0, 255, 0)
    thickness = 2 # Độ dày của đường viền

    # Vẽ hình chữ nhật lên ảnh
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

    # Lưu ảnh để kiểm tra kết quả
    cv2.imwrite(output_path, img)
    print(f"Đã lưu ảnh test thành công tại: {output_path}")

    # (Tùy chọn) Hiển thị ảnh pop-up nếu bạn chạy code trên máy local
    # cv2.imshow("Kiểm tra Bounding Box", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# --- Test thử hàm ---
# Truyền tọa độ x, y, w, h thực tế mà script đọc annotation của bạn in ra
draw_green_bounding_box("000000.png", x=219.31, y=188.49, w=26.19, h=30.07)