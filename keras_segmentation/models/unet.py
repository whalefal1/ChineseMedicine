import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import get_segmentation_model
from .resnet50 import resnet50_encoder

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResUNet(nn.Module):
    def __init__(self, encoder_channels, n_classes=2):
        super(ResUNet, self).__init__()
        self.encoder_channels = encoder_channels
        
        # 解码器
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(encoder_channels[4], encoder_channels[3], 256),
            DecoderBlock(256, encoder_channels[2], 128),
            DecoderBlock(128, encoder_channels[1], 64),
            DecoderBlock(64, encoder_channels[0], 32),
            DecoderBlock(32, 0, 16)
        ])
        
        # 最后的分类层
        self.final_conv = nn.Conv2d(16, n_classes, kernel_size=1)
        
    def forward(self, features):
        x = features[-1]  # 从编码器的最后一层开始
        skips = features[:-1][::-1]  # 反转跳跃连接特征
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            
        x = self.final_conv(x)
        return x

def resnet50_unet(n_classes=2, input_height=None, input_width=None, pretrained=True):
    """
    构建基于ResNet50的UNet模型
    
    Args:
        n_classes: 类别数
        input_height: 输入图像高度
        input_width: 输入图像宽度
        pretrained: 是否使用预训练权重
    Returns:
        Res-UNet模型
    """
    # 获取基础分割模型类
    base_model = get_segmentation_model(input_height=input_height, input_width=input_width)
    
    # 构建完整的Res-UNet模型
    class ResUNetModel(base_model):
        def __init__(self):
            super(ResUNetModel, self).__init__()
            self.encoder, channels = resnet50_encoder(pretrained=pretrained)
            self.decoder = ResUNet(channels, n_classes=n_classes)
            
        def forward(self, x):
            features = self.encoder(x)
            x = self.decoder(features)
            return F.interpolate(x, size=x.shape[2:], mode='bilinear', align_corners=True)
            
    return ResUNetModel() 