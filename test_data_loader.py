import unittest
import os
import torch
import numpy as np
from keras_segmentation.data_utils.data_loader import (
    set_seed,
    DataLoadError,
    get_pairs_from_paths,
    get_image_array,
    get_segmentation_array,
    TongueDataset,
    get_data_loader
)

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.train_img_path = 'tongue_data/tongue_data/tongue_data/train_img'
        cls.train_label_path = 'tongue_data/tongue_data/tongue_data/train_label'
        cls.test_img_path = 'tongue_data/tongue_data/tongue_data/test_img'
        cls.test_label_path = 'tongue_data/tongue_data/tongue_data/test_label'
        
        # 设置随机种子
        set_seed(42)

    def test_get_pairs_from_paths(self):
        """测试图像和标签配对功能"""
        pairs = get_pairs_from_paths(self.train_img_path, self.train_label_path)
        
        # 检查返回的pairs是否为列表
        self.assertIsInstance(pairs, list)
        
        # 检查是否有配对数据
        self.assertGreater(len(pairs), 0)
        
        # 检查每个配对的格式
        for img_path, label_path in pairs:
            self.assertTrue(os.path.exists(img_path))
            self.assertTrue(os.path.exists(label_path))
            
            # 检查文件名匹配（不包括路径和扩展名）
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_name = os.path.splitext(os.path.basename(label_path))[0]
            self.assertEqual(img_name, label_name)

    def test_get_image_array(self):
        """测试图像处理功能"""
        # 获取一个测试图像路径
        test_img_path = os.path.join(self.train_img_path, os.listdir(self.train_img_path)[0])
        
        # 测试不同的标准化方式
        for norm_type in ["sub_mean", "sub_and_divide", "divide"]:
            img_array = get_image_array(
                test_img_path,
                width=224,
                height=224,
                imgNorm=norm_type
            )
            
            # 检查数组形状和类型
            self.assertEqual(img_array.shape, (3, 224, 224))
            self.assertEqual(img_array.dtype, np.float32)
            
            # 检查数值范围
            if norm_type == "sub_and_divide":
                self.assertTrue(np.all(img_array >= -1) and np.all(img_array <= 1))
            elif norm_type == "divide":
                self.assertTrue(np.all(img_array >= 0) and np.all(img_array <= 1))

    def test_get_segmentation_array(self):
        """测试标签图像处理功能"""
        # 获取一个测试标签图像路径
        test_label_path = os.path.join(self.train_label_path, os.listdir(self.train_label_path)[0])
        
        # 测试标签处理
        label_array = get_segmentation_array(
            test_label_path,
            width=224,
            height=224
        )
        
        # 检查数组形状
        self.assertEqual(label_array.shape, (1, 224, 224))
        
        # 检查值是否为二值
        unique_values = np.unique(label_array)
        self.assertTrue(np.all(np.isin(unique_values, [0, 1])))

    def test_tongue_dataset(self):
        """测试数据集类"""
        dataset = TongueDataset(
            images_path=self.train_img_path,
            labels_path=self.train_label_path,
            width=224,
            height=224
        )
        
        # 检查数据集大小
        self.assertGreater(len(dataset), 0)
        
        # 测试数据获取
        img, label = dataset[0]
        
        # 检查返回的张量类型和形状
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertEqual(label.shape, (1, 224, 224))
        self.assertEqual(label.dtype, torch.long)

    def test_data_loader(self):
        """测试数据加载器"""
        batch_size = 4
        loader = get_data_loader(
            images_path=self.train_img_path,
            labels_path=self.train_label_path,
            batch_size=batch_size,
            width=224,
            height=224
        )
        
        # 获取一个批次
        images, labels = next(iter(loader))
        
        # 检查批次大小和形状
        self.assertEqual(images.shape, (batch_size, 3, 224, 224))
        self.assertEqual(labels.shape, (batch_size, 1, 224, 224))
        
        # 检查数据类型
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)

    def test_error_handling(self):
        """测试错误处理"""
        # 测试不存在的路径
        with self.assertRaises(DataLoadError):
            get_pairs_from_paths("non_existent_path", self.train_label_path)
        
        # 测试无效的图像路径
        with self.assertRaises(DataLoadError):
            get_image_array("invalid_image.jpg", 224, 224)

if __name__ == '__main__':
    unittest.main() 