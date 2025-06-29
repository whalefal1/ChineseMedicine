import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from keras_segmentation.models import get_model
from tqdm import tqdm

class TongueDataset(Dataset):
    """舌体数据集类"""
    def __init__(self, img_dir, label_dir, transform=None):
        """
        初始化数据集
        
        Args:
            img_dir: 图像目录
            label_dir: 标签目录
            transform: 数据增强转换
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 获取所有图像文件
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.bmp'))]
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        # 读取图像和标签
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        
        # 读取并预处理图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        # 读取并预处理标签
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (256, 256))
        label = (label > 0).astype(np.float32)
        label = label[np.newaxis, ...]  # 添加通道维度
        
        # 转换为张量
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        
        return image, label

def test_model():
    """测试模型功能"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = get_model(
        model_name='resnet50_unet',
        n_classes=1,  # 二分类问题使用单通道输出
        input_height=256,
        input_width=256,
        pretrained=True
    )
    model = model.to(device)
    print('模型创建成功')
    
    # 创建数据集和数据加载器
    test_dataset = TongueDataset(
        img_dir='tongue_data/tongue_data/tongue_data/test_img',
        label_dir='tongue_data/tongue_data/tongue_data/test_label'
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f'测试数据集大小: {len(test_dataset)}')
    
    # 评估模式
    model.eval()
    
    # 测试指标
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc='测试进度')):
            try:
                # 将数据移到指定设备
                images = images.to(device)
                labels = labels.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 确保输出尺寸与标签一致
                if outputs.shape != labels.shape:
                    outputs = torch.nn.functional.interpolate(
                        outputs,
                        size=labels.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                
                # 获取预测结果
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                # 计算IoU
                intersection = torch.sum(predictions * labels, dim=(2, 3))
                union = torch.sum((predictions + labels) > 0, dim=(2, 3)).float()
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.mean().item()
                
                # 计算Dice系数
                dice = (2 * intersection + 1e-6) / (torch.sum(predictions, dim=(2, 3)) + torch.sum(labels, dim=(2, 3)) + 1e-6)
                total_dice += dice.mean().item()
                
                # 保存第一个batch的预测结果
                if batch_idx == 0:
                    save_predictions(images, labels, predictions, batch_idx)
                    
            except Exception as e:
                print(f'处理batch {batch_idx}时出错: {str(e)}')
                print(f'输出形状: {outputs.shape}')
                print(f'标签形状: {labels.shape}')
                print(f'预测形状: {predictions.shape}')
                continue
    
    # 计算平均指标
    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    
    print(f'平均IoU: {avg_iou:.4f}')
    print(f'平均Dice系数: {avg_dice:.4f}')

def save_predictions(images, labels, predictions, batch_idx):
    """保存预测结果可视化图像"""
    os.makedirs('test_results', exist_ok=True)
    
    # 转换为NumPy数组
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    for i in range(len(images)):
        # 还原图像
        image = images[i].transpose(1, 2, 0)  # CHW -> HWC
        image = (image * 255).astype(np.uint8)
        
        # 创建可视化图像
        label = (labels[i, 0] * 255).astype(np.uint8)  # 取第一个通道
        pred = (predictions[i, 0] * 255).astype(np.uint8)  # 取第一个通道
        
        # 保存结果
        cv2.imwrite(f'test_results/image_{batch_idx}_{i}.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'test_results/label_{batch_idx}_{i}.jpg', label)
        cv2.imwrite(f'test_results/pred_{batch_idx}_{i}.jpg', pred)

def visualize_segmentation(image, seg):
    """
    将分割结果叠加到原图上
    """
    if image.max() > 1:
        image = image / 255.0
    seg_color = get_colored_segmentation_image(seg)
    # resize seg_color to match image size
    if seg_color.shape[:2] != image.shape[:2]:
        seg_color = cv2.resize(seg_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = (image * 0.7 + seg_color / 255.0 * 0.3) * 255
    return vis.astype(np.uint8)

if __name__ == '__main__':
    test_model() 