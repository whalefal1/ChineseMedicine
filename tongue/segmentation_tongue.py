import cv2
import numpy as np
from tongue.tongue_segmentation.segmentation import viscera_split

# seg_tongue: 输入原始舌体图像和分割模型，输出五脏区域mask
# img_path: 舌体原图路径
# model: 已加载的分割模型（需实现predict方法，返回二值mask）
# 返回: 五脏区域mask字典

def seg_tongue(img_path, model):
    # 读取原图
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    # 预测舌体mask
    mask = model.predict(img)  # 需保证输出为单通道二值mask
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8) * 255
    # 区域划分
    regions = viscera_split(mask)
    return regions
