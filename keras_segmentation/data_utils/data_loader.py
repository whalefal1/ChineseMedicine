import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import random
import itertools

# 设置随机种子
def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可重现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 自定义异常类
class DataLoadError(Exception):
    """数据加载异常类"""
    pass

def get_pairs_from_paths(images_path: str, labels_path: str) -> List[Tuple[str, str]]:
    """
    获取图像和标签的配对路径
    
    Args:
        images_path: 图像目录路径
        labels_path: 标签目录路径
    
    Returns:
        包含(图像路径, 标签路径)元组的列表
    
    Raises:
        DataLoadError: 当图像和标签不匹配时抛出异常
    """
    image_files = []
    label_files = []
    
    # 支持的图像格式
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    
    try:
        # 获取所有图像文件
        for filename in os.listdir(images_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_formats:
                image_files.append((
                    os.path.splitext(filename)[0],
                    ext,
                    os.path.join(images_path, filename)
                ))
        
        # 获取所有标签文件
        for filename in os.listdir(labels_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_formats:
                label_files.append((
                    os.path.splitext(filename)[0],
                    os.path.join(labels_path, filename)
                ))
    except FileNotFoundError as e:
        raise DataLoadError(f"数据路径不存在: {e}")
    
    # 匹配图像和标签
    pairs = []
    for img_name, img_ext, img_path in image_files:
        label_path = None
        for label_name, label_full_path in label_files:
            if img_name == label_name:
                label_path = label_full_path
                break
        
        if label_path is None:
            raise DataLoadError(f"找不到图像 {img_name}{img_ext} 对应的标签文件")
        
        pairs.append((img_path, label_path))
    
    return pairs

def get_image_array(
    image_input: str,
    width: int,
    height: int,
    imgNorm: str = "sub_mean",
    ordering: str = 'channels_first'
) -> np.ndarray:
    """
    读取并标准化处理图像
    
    Args:
        image_input: 图像路径或numpy数组
        width: 目标宽度
        height: 目标高度
        imgNorm: 标准化方式，可选 "sub_mean"、"sub_and_divide"、"divide"
        ordering: 通道顺序，可选 'channels_first' 或 'channels_last'
    
    Returns:
        标准化后的图像数组
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input

    if img is None:
        raise DataLoadError(f"无法读取图像: {image_input}")
    
    # 调整图像大小
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    
    # 标准化处理
    if imgNorm == "sub_and_divide":
        img = img / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = img - np.mean(img)
    elif imgNorm == "divide":
        img = img / 255.0
    
    # 调整通道顺序
    if ordering == 'channels_first':
        img = img.transpose((2, 0, 1))
    
    return img

def get_segmentation_array(
    image_input: str,
    width: int,
    height: int,
    no_reshape: bool = False
) -> np.ndarray:
    """
    读取并处理标签图像
    
    Args:
        image_input: 标签图像路径或numpy数组
        width: 目标宽度
        height: 目标高度
        no_reshape: 是否保持原始维度
    
    Returns:
        处理后的标签数组
    """
    if isinstance(image_input, str):
        seg_labels = cv2.imread(image_input, 0)
    else:
        seg_labels = image_input

    if seg_labels is None:
        raise DataLoadError(f"无法读取标签图像: {image_input}")

    # 调整大小
    seg_labels = cv2.resize(seg_labels, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # 二值化处理
    seg_labels[seg_labels > 1] = 1
    
    if not no_reshape:
        seg_labels = np.expand_dims(seg_labels, axis=0)
    
    return seg_labels.astype(np.float32)

class TongueDataset(Dataset):
    """舌诊数据集类"""
    
    def __init__(
        self,
        images_path: str,
        labels_path: str,
        width: int = 224,
        height: int = 224,
        image_norm: str = "sub_mean"
    ):
        """
        初始化数据集
        
        Args:
            images_path: 图像目录路径
            labels_path: 标签目录路径
            width: 图像宽度
            height: 图像高度
            image_norm: 图像标准化方式
        """
        self.pairs = get_pairs_from_paths(images_path, labels_path)
        self.width = width
        self.height = height
        self.image_norm = image_norm

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.pairs[idx]
        
        # 读取和处理图像
        img = get_image_array(
            img_path,
            self.width,
            self.height,
            imgNorm=self.image_norm
        )
        
        # 读取和处理标签
        label = get_segmentation_array(
            label_path,
            self.width,
            self.height
        )
        
        return torch.from_numpy(img), torch.from_numpy(label).long()

def get_data_loader(
    images_path: str,
    labels_path: str,
    batch_size: int = 32,
    width: int = 224,
    height: int = 224,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        images_path: 图像目录路径
        labels_path: 标签目录路径
        batch_size: 批次大小
        width: 图像宽度
        height: 图像高度
        shuffle: 是否打乱数据
        num_workers: 数据加载的线程数
    
    Returns:
        PyTorch数据加载器
    """
    dataset = TongueDataset(
        images_path=images_path,
        labels_path=labels_path,
        width=width,
        height=height
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 