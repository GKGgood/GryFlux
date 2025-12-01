import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import numpy as np
import cv2

# ================= 配置区域 =================
ONNX_PATH = '/workspace/zyp/GryFlux/data/model/yolox_s.onnx'
ENGINE_PATH = '/workspace/zyp/GryFlux/data/model/yolox_s_int8.engine'
CACHE_FILE = 'yolox_calibration.cache'
CALIB_IMG_DIR = '/workspace/zyp/GryFlux/data/dataset'  # ⚠️ 改为您存放校准图片的文件夹路径
INPUT_SHAPE = (1, 3, 640, 640) # YOLOX-S 默认通常是 640x640，请根据实际情况修改
BATCH_SIZE = 1
CALIB_COUNT = 21 # 用于校准的图片数量，通常 100-500 张足够
# ===========================================

class DataLoader:
    def __init__(self, batch_size, img_dir, input_shape):
        self.index = 0
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(self.img_list) == 0:
            raise FileNotFoundError(f"在 {img_dir} 中没有找到图片！")
        self.calibration_data = np.zeros((self.batch_size, *input_shape[1:]), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index + self.batch_size > len(self.img_list):
            return None # 数据读完了

        # 加载一批图片
        for i in range(self.batch_size):
            img_path = self.img_list[self.index + i]
            img = cv2.imread(img_path)
            
            # ⚠️⚠️⚠️ 关键点：这里的预处理必须和您推理/训练时完全一致！ ⚠️⚠️⚠️
            # YOLOX 通常需要 Letterbox (保持长宽比缩放)
            # 为了简化，这里演示直接 Resize。如果精度太差，请替换为 Letterbox 代码。
            img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
            
            # 归一化和维度转换 (HWC -> CHW)
            # YOLOX 原版通常不除以 255 (保持 0-255 范围)，但要看您导出的 ONNX 里面是否包含了预处理
            # 假设 ONNX 输入需要 float32 且布局为 NCHW
            input_data = img.astype(np.float32)
            input_data = input_data.transpose((2, 0, 1)) # HWC to CHW
            
            self.calibration_data[i] = input_data

        self.index += self.batch_size
        # 返回连续的内存数组（必须是 contiguous 的）
        return np.ascontiguousarray(self.calibration_data)

class ENTROPY_CALIBRATOR(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size, img_dir, input_shape):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data_loader = DataLoader(batch_size, img_dir, input_shape)
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        # TensorRT 调用此函数来获取输入数据
        try:
            batch_data = self.data_loader.next_batch()
            if batch_data is None and self.data_loader.index == 0:
                 print("没有找到校准图片！")
                 return None
            if batch_data is None:
                return None
            
            # 将数据拷贝到 GPU
            cuda.memcpy_htod(self.d_input, batch_data)
            return [int(self.d_input)]
        except Exception as e:
            print(f"Error in get_batch: {e}")
            return None

    def read_calibration_cache(self):
        # 如果缓存文件存在，直接读取
        if os.path.exists(self.cache_file):
            print(f"发现现有缓存文件: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        # 校准完成后，TensorRT 会调用此函数保存缓存
        print(f"正在写入缓存文件: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    # 1. 解析 ONNX
    print(f"正在解析 ONNX: {ONNX_PATH}")
    with open(ONNX_PATH, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 2. 配置 INT8
    config.set_flag(trt.BuilderFlag.INT8)
    # 如果某些层不支持 INT8，允许回退到 FP16/FP32
    config.set_flag(trt.BuilderFlag.FP16) 
    
    # 3. 设置校准器
    calibrator = ENTROPY_CALIBRATOR(CACHE_FILE, BATCH_SIZE, CALIB_IMG_DIR, INPUT_SHAPE)
    config.int8_calibrator = calibrator

    # 4. 设置内存池 (例如 2GB)
    # TensorRT 8.5+ 使用 set_memory_pool_limit, 旧版本用 max_workspace_size
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) 
    except:
        config.max_workspace_size = 2 << 30

    # 5. 构建 Engine
    print("开始构建 INT8 Engine (这需要一些时间，正在进行校准)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine:
        print(f"Engine 构建成功！保存到: {ENGINE_PATH}")
        with open(ENGINE_PATH, "wb") as f:
            f.write(serialized_engine)
        print(f"校准表已保存到: {CACHE_FILE}")
    else:
        print("Engine 构建失败。")

if __name__ == "__main__":
    build_int8_engine()