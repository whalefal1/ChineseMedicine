import torch
import torch.nn as nn

def get_segmentation_model(input_height=None, input_width=None):
    """
    定义分割模型的基类
    
    Args:
        input_height: 输入图像高度
        input_width: 输入图像宽度
    """
    class SegmentationModel(nn.Module):
        def __init__(self):
            super(SegmentationModel, self).__init__()
            self.input_height = input_height
            self.input_width = input_width
            self.n_classes = 2  # 二分类：舌体区域和非舌体区域
            
        def forward(self, x):
            raise NotImplementedError
            
        def predict(self, x):
            """
            预测函数
            
            Args:
                x: 输入图像张量
            Returns:
                预测结果
            """
            self.eval()
            with torch.no_grad():
                out = self.forward(x)
                return torch.sigmoid(out) > 0.5
                
    return SegmentationModel 