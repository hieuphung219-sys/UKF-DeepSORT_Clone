import sys
import os
import torch
import torchvision.transforms as T
from PIL import Image

# Chỉ định đường dẫn tới thư mục fast-reid vừa clone
current_dir = os.path.dirname(os.path.abspath(__file__))
fast_reid_path = os.path.join(current_dir, 'fast-reid')
sys.path.insert(0, fast_reid_path)

# Import module cấu trúc mạng từ Fast-ReID
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model

class PyTorchFeatureExtractor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Thiết lập cấu hình chuẩn của sbs_R50-ibn
        self.cfg = get_cfg()
        config_file = os.path.join(fast_reid_path, "configs", "VeRi", "sbs_R50-ibn.yml")
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.DEVICE = self.device
        
        # 2. Khởi tạo kiến trúc mạng (Tự động tạo các lớp IBN, Non-Local...)
        self.model = build_model(self.cfg)
        self.model.eval()
        
        # 3. Nạp trọng số thủ công (Tránh lỗi thư viện)
        weights = torch.load(model_path, map_location=self.device)
        if 'model' in weights:
            weights = weights['model']
        elif 'state_dict' in weights:
            weights = weights['state_dict']
            
        new_state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # 4. Transform chuẩn ImageNet
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    def extract_feature(self, img_patch_letterbox):
        # Chuyển hệ màu từ BGR (OpenCV) sang RGB
        img_rgb = img_patch_letterbox[:, :, ::-1]
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)
        
        # L2 Normalize theo chuẩn để tính Cosine Distance
        feature_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        return feature_norm.cpu().numpy().squeeze()