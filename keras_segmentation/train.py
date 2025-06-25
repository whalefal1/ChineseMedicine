import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data_utils.data_loader import TongueDataset
from .models.all_models import get_model

def find_latest_checkpoint(checkpoints_path):
    """查找最新的检查点"""
    if not os.path.exists(checkpoints_path):
        return None
        
    checkpoints = [f for f in os.listdir(checkpoints_path) if f.endswith('.pth')]
    if not checkpoints:
        return None
        
    epochs = [int(re.search(r'(\d+)', f).group(1)) for f in checkpoints]
    latest_epoch = max(epochs)
    
    return os.path.join(checkpoints_path, f'model_epoch_{latest_epoch}.pth')

def train(
    model_name,
    train_images,
    train_annotations,
    val_images,
    val_annotations,
    epochs=5,
    batch_size=2,
    checkpoints_path='checkpoints'
):
    """
    训练模型
    
    Args:
        model_name: 模型名称
        train_images: 训练图像目录
        train_annotations: 训练标签目录
        val_images: 验证图像目录
        val_annotations: 验证标签目录
        epochs: 训练轮次
        batch_size: 批大小
        checkpoints_path: 检查点保存路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型
    model = get_model(
        model_name=model_name,
        n_classes=1,
        input_height=256,
        input_width=256,
        pretrained=True
    ).to(device)
    
    # 保存配置文件
    config = {
        'model_name': model_name,
        'n_classes': 1,
        'input_height': 256,
        'input_width': 256,
        'output_height': 256,
        'output_width': 256
    }
    os.makedirs(checkpoints_path, exist_ok=True)
    config_path = os.path.join(checkpoints_path, "resunet_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"模型配置已保存至: {config_path}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 加载最新的检查点
    latest_checkpoint = find_latest_checkpoint(checkpoints_path)
    if latest_checkpoint:
        print(f'加载检查点: {latest_checkpoint}')
        model.load_state_dict(torch.load(latest_checkpoint))
    
    # 创建数据加载器
    train_dataset = TongueDataset(
        images_path=train_images, 
        labels_path=train_annotations,
        width=256,
        height=256,
        image_norm="divide"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TongueDataset(
        images_path=val_images, 
        labels_path=val_annotations,
        width=256,
        height=256,
        image_norm="divide"
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 开始训练
    for epoch in range(1, epochs + 1):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}') as pbar:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # 确保输出尺寸与标签一致
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs,
                        size=labels.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )

                loss = criterion(outputs, labels.type_as(outputs))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
                
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                # 确保输出尺寸与标签一致
                if outputs.shape[-2:] != labels.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs,
                        size=labels.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                loss = criterion(outputs, labels.type_as(outputs))
                
                val_loss += loss.item() * images.size(0)
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 保存检查点
        os.makedirs(checkpoints_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoints_path, f'model_epoch_{epoch}.pth'))
        
    print('训练完成') 