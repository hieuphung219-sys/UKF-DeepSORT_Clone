import tensorflow as tf
import numpy as np

# Đường dẫn đến file model của bạn
model_path = "resources/networks/mobilenet_v2_coco.tflite"

print(f"--- ĐANG KIỂM TRA FILE: {model_path} ---")
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    print("\n1. THÔNG TIN ĐẦU VÀO (INPUT):")
    input_details = interpreter.get_input_details()
    for i, detail in enumerate(input_details):
        print(f"   Input {i}: Index={detail['index']}, Shape={detail['shape']}, Type={detail['dtype']}")

    print("\n2. THÔNG TIN ĐẦU RA (OUTPUT):")
    output_details = interpreter.get_output_details()
    for i, detail in enumerate(output_details):
        print(f"   Output {i}: Index={detail['index']}, Name={detail['name']}, Shape={detail['shape']}")

except Exception as e:
    print("\nLỖI: Không tìm thấy file hoặc file bị hỏng.")
    print(e)