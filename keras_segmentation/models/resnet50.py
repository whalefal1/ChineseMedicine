import torch
import torch.nn as nn
import torchvision.models as models

def resnet50_encoder(pretrained=True):
    """
    构建ResNet50编码器
    
    Args:
        pretrained: 是否使用预训练权重
    Returns:
        编码器模型和各阶段输出通道数
    """
    resnet = models.resnet50(pretrained=pretrained)
    
    # 获取各阶段的特征图
    stages = []
    stages.append(nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu))  # stage 1
    stages.append(nn.Sequential(resnet.maxpool, resnet.layer1))          # stage 2
    stages.append(resnet.layer2)                                         # stage 3
    stages.append(resnet.layer3)                                         # stage 4
    stages.append(resnet.layer4)                                         # stage 5
    
    # 各阶段输出通道数
    channels = [64, 256, 512, 1024, 2048]
    
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.stages = nn.ModuleList(stages)
            
        def forward(self, x):
            features = []
            for stage in self.stages:
                x = stage(x)
                features.append(x)
            return features
            
    return Encoder(), channels 