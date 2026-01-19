import cv2
import numpy as np
import tensorflow as tf

class Detector:
    def __init__(self, model_path, min_confidence=0.3, min_height=0):
        self.min_confidence = min_confidence
        self.min_height = min_height
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Sắp xếp output để đảm bảo lấy đúng thứ tự
        self.output_details.sort(key=lambda x: x['index'])

        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        self.float_input = (self.input_details[0]['dtype'] == np.float32)

        # --- BỘ LỌC ID (QUAN TRỌNG) ---
        # Theo chuẩn COCO labelmap (dựa trên file starter model):
        # 0: ???
        # 1: person
        # 2: bicycle (Xe đạp)
        # 3: car (Ô tô)
        # 4: motorcycle (Xe máy)
        # 6: bus (Xe buýt)
        # 8: truck (Xe tải)
        # Chúng ta chỉ cho phép các ID này đi qua.
        self.valid_classes = [2, 3, 4, 6, 8]

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Xử lý đầu vào (Hỗ trợ cả model Float và Int)
        if self.float_input:
            input_data = (np.float32(input_data) - 127.5) / 127.5
        else:
            # Model Quantized (của bạn) dùng uint8, không cần chia gì cả
            pass 

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Lấy kết quả
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] 
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        detections = []

        for i in range(len(scores)):
            # --- CHIẾN THUẬT MỚI ---
            # 1. Kiểm tra ID lớp trước (Chỉ lấy xe)
            object_id = int(classes[i])
            if object_id not in self.valid_classes:
                continue # Bỏ qua ngay nếu không phải xe (Giúp loại bỏ khung to vô nghĩa)

            # 2. Ngưỡng tin cậy (Hạ xuống thấp một chút để bắt xe xa)
            # Bạn có thể chỉnh self.min_confidence trong file raspi_deepsort.py xuống 0.2 hoặc 0.25
            if (scores[i] > self.min_confidence) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                
                w = xmax - xmin
                h = ymax - ymin
                
                # Loại bỏ các khung hình quá to bất thường (VD: chiếm > 80% màn hình)
                if (w * h) > (imW * imH * 0.8):
                    continue

                if h > self.min_height:
                    detections.append([np.array([xmin, ymin, w, h]), scores[i]])

        return detections