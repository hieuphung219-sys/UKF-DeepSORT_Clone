import cv2
import numpy as np
from tools.pytorch_feature_extractor import PyTorchFeatureExtractor
from application_util.preprocessing import extract_image_patches

# Khởi tạo với đuôi .pt
extractor = PyTorchFeatureExtractor(model_path="resources/networks/veri_sbs_R50-ibn.pt")

dummy_image = np.zeros((500, 500, 3), dtype=np.uint8) 
cv2.rectangle(dummy_image, (100, 100), (200, 250), (255, 255, 255), -1)

bbox_xe = [[100, 100, 100, 150]]

patches = extract_image_patches(dummy_image, bbox_xe, patch_shape=(256, 128))

vector_dac_trung = extractor.extract_feature(patches[0])

print("Kích thước Vector:", vector_dac_trung.shape)