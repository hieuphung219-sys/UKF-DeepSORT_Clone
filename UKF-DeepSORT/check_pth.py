import torch

# Thay đường dẫn thực tế tới file .pth của bạn
model_path = "resources/networks/veri_sbs_R50-ibn.pth" 

try:
    # Load trọng số vào RAM (dùng CPU cho an toàn)
    weights = torch.load(model_path, map_location='cpu')
    
    # Đôi khi file .pth bọc trọng số trong một key như 'model' hoặc 'state_dict'
    if 'state_dict' in weights:
        weights = weights['state_dict']
    elif 'model' in weights:
        weights = weights['model']
        
    print(f"Tổng số lớp (layers) trong mô hình: {len(weights.keys())}")
    print("-" * 50)
    print("Tên của 15 lớp đầu tiên để nhận diện cấu trúc:")

    # In ra 15 keys cuối cùng
    for key in list(weights.keys())[-15:]:
        print(key)
        
except Exception as e:
    print("Có lỗi xảy ra:", e)