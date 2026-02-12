import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
import time
import logging

# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ReID_Pipeline")

# =============================================================================
# 2. UTILS & PROFILING MODULE
# =============================================================================

class Profiler:
    """Class hỗ trợ đo thời gian thực thi cho mục đích viết báo cáo/paper."""
    def __init__(self):
        self.stats = {
            'io_time': 0.0,       # Thời gian đọc ảnh
            'preproc_time': 0.0,  # Thời gian cắt/resize ảnh
            'inference_time': 0.0,# Thời gian chạy model
            'total_frames': 0
        }
        self.start_t = 0

    def tic(self):
        self.start_t = time.perf_counter()

    def toc(self, key):
        duration = time.perf_counter() - self.start_t
        self.stats[key] += duration
        return duration

    def increment_frame(self, count=1):
        self.stats['total_frames'] += count

    def report(self):
        total = self.stats['total_frames']
        if total == 0: return
        
        io_avg = (self.stats['io_time'] / total) * 1000
        pre_avg = (self.stats['preproc_time'] / total) * 1000
        inf_avg = (self.stats['inference_time'] / total) * 1000
        total_time = self.stats['io_time'] + self.stats['preproc_time'] + self.stats['inference_time']
        fps = total / total_time if total_time > 0 else 0

        logger.info("-" * 40)
        logger.info(f"PERFORMANCE REPORT (Processed {total} crops)")
        logger.info(f" > I/O Time       : {io_avg:.2f} ms/crop")
        logger.info(f" > Pre-proc Time  : {pre_avg:.2f} ms/crop")
        logger.info(f" > Inference Time : {inf_avg:.2f} ms/crop")
        logger.info(f" > Throughput     : {fps:.2f} crops/sec")
        logger.info("-" * 40)

def extract_image_patch(image, bbox, patch_shape):
    """Trích xuất và resize vùng ảnh (Pre-processing)."""
    bbox = np.array(bbox)
    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int32)

    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

# =============================================================================
# 3. REID MODULE (Feature Extractor)
# =============================================================================

class BaseEncoder:
    def __init__(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name
        self.image_shape = None
        self.feature_dim = None

    def normalize(self, features):
        """
        🚀 L2 Normalization: Bước cực kỳ quan trọng cho DeepSORT.
        Đưa các vector về độ dài đơn vị để so sánh Cosine Distance chính xác.
        """
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        # Thêm epsilon để tránh chia cho 0
        return features / (norm + 1e-10)

class TFLiteImageEncoder(BaseEncoder):
    """Engine cho file .tflite (TensorFlow 2.x / Edge)"""
    def __init__(self, checkpoint_filename, input_name="images", output_name="features"):
        super().__init__(input_name, output_name)
        try:
            self.interpreter = tf.lite.Interpreter(model_path=checkpoint_filename)
        except Exception as e:
            logger.critical(f"Failed to load TFLite model: {e}")
            raise e
            
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.image_shape = self.input_details[0]['shape'][1:]
        self.feature_dim = self.output_details[0]['shape'][1]
        logger.info(f"Initialized TFLite Engine. Input: {self.image_shape}, Output: {self.feature_dim}")

    def __call__(self, data_x):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)

        input_dtype = self.input_details[0]['dtype']

        for i, img in enumerate(data_x):
            input_data = np.expand_dims(img, axis=0)

            # 🔥 Ép đúng kiểu theo model yêu cầu
            if input_data.dtype != input_dtype:
                input_data = input_data.astype(input_dtype)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details[0]['index'])
            out[i] = result[0]

        return self.normalize(out)


class TF1ImageEncoder(BaseEncoder):
    """Engine cho file .pb (TensorFlow 1.x / PC)"""
    def __init__(self, checkpoint_filename, input_name="images", output_name="features"):
        super().__init__(input_name, output_name)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session()
            with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(file_handle.read())
            tf.import_graph_def(graph_def, name="net")
            
            self.input_var = self.graph.get_tensor_by_name("net/%s:0" % input_name)
            self.output_var = self.graph.get_tensor_by_name("net/%s:0" % output_name)
            
            self.feature_dim = self.output_var.get_shape().as_list()[-1]
            self.image_shape = self.input_var.get_shape().as_list()[1:]
        logger.info(f"Initialized TF1 Frozen Graph Engine. Input: {self.image_shape}, Output: {self.feature_dim}")

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        
        # Batch processing helper inside class
        data_len = len(data_x)
        num_batches = int(data_len / batch_size)
        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            out[s:e] = self.session.run(self.output_var, feed_dict={self.input_var: data_x[s:e]})
        if e < data_len:
            out[e:] = self.session.run(self.output_var, feed_dict={self.input_var: data_x[e:]})
            
        return self.normalize(out)

def create_box_encoder(model_filename, batch_size=32):
    if model_filename.endswith(".tflite"):
        encoder_engine = TFLiteImageEncoder(model_filename)
    else:
        encoder_engine = TF1ImageEncoder(model_filename)
        
    image_shape = encoder_engine.image_shape

    def encoder(image, boxes, profiler=None):
        image_patches = []
        if profiler: profiler.tic()
        
        # Pre-processing Loop
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        
        if profiler: profiler.toc('preproc_time')

        # Inference
        if profiler: profiler.tic()
        features = encoder_engine(image_patches, batch_size) if hasattr(encoder_engine, 'session') else encoder_engine(image_patches)
        if profiler: profiler.toc('inference_time')
        
        return features

    return encoder

# =============================================================================
# 4. DATASET MODULE (MOT Format Loader)
# =============================================================================

class MOTDatasetLoader:
    def __init__(self, mot_dir, detection_mode='gt'):
        self.mot_dir = mot_dir
        self.mode = detection_mode
        self.sequences = [s for s in os.listdir(mot_dir) if os.path.isdir(os.path.join(mot_dir, s))]
        logger.info(f"Dataset loaded. Found {len(self.sequences)} sequences.")

    def get_loader(self, sequence):
        """Generator trả về dữ liệu từng frame của 1 sequence."""
        sequence_dir = os.path.join(self.mot_dir, sequence)
        image_dir = os.path.join(sequence_dir, "img1")
        
        # Determine Detection File
        det_file = os.path.join(sequence_dir, "gt/gt.txt") if self.mode == 'gt' else os.path.join(sequence_dir, "det/det.txt")
        
        if not os.path.exists(det_file):
            logger.warning(f"Detection file not found: {det_file}")
            return None

        # Load Detections
        detections = np.loadtxt(det_file, delimiter=',')
        
        # Image Look-up table
        img_files = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))
        }

        min_frame = int(detections[:, 0].min())
        max_frame = int(detections[:, 0].max())
        
        logger.info(f"Sequence: {sequence} | Frames: {min_frame}-{max_frame} | Mode: {self.mode.upper()}")

        for frame_idx in range(min_frame, max_frame + 1):
            if frame_idx not in img_files: continue
            
            # Get boxes for current frame
            mask = detections[:, 0].astype(np.int32) == frame_idx
            rows = detections[mask]
            
            if len(rows) == 0: continue
            
            # Return info needed for processing
            yield img_files[frame_idx], rows

# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def run_pipeline(args):
    profiler = Profiler()
    dataset = MOTDatasetLoader(args.mot_dir, args.detection_mode)
    encoder = create_box_encoder(args.model, batch_size=32)
    
    output_dir = args.output_dir if args.output_dir else args.mot_dir
    os.makedirs(output_dir, exist_ok=True)

    for seq in dataset.sequences:
        loader = dataset.get_loader(seq)
        if loader is None: continue
        
        detections_out = []
        
        for img_path, rows in loader:
            # Measure I/O
            profiler.tic()
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            profiler.toc('io_time')
            
            if image is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            # Extract Features (bao gồm Preproc + Inference + Normalize)
            # Cột 2:6 là x, y, w, h
            features = encoder(image, rows[:, 2:6].copy(), profiler)
            
            # Update Stats
            profiler.increment_frame(len(rows))

            # Concatenate features
            detections_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]

        # Save Result
        if len(detections_out) > 0:
            out_file = os.path.join(output_dir, seq, "detections.npy")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.save(out_file, np.asarray(detections_out), allow_pickle=False)
            logger.info(f"Saved {len(detections_out)} detections to {out_file}")
        else:
            logger.warning(f"No detections generated for {seq}")

    # Final Report
    profiler.report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSORT Feature Generation Pipeline")
    parser.add_argument("--model", default="resources/networks/mars-small128.pb", help="Path to model (.pb or .tflite)")
    parser.add_argument("--mot_dir", required=True, help="Path to MOT dataset root")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--detection_mode", default="gt", choices=["gt", "det"], help="Source: gt or det")
    
    args = parser.parse_args()
    
    # Kiểm tra model path
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        exit(1)
        
    run_pipeline(args)