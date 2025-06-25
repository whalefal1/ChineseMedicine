from .unet import resnet50_unet

def get_model(model_name, n_classes=2, input_height=None, input_width=None, pretrained=True):
    """
    根据模型名称获取对应的模型实例
    
    Args:
        model_name: 模型名称
        n_classes: 类别数
        input_height: 输入图像高度
        input_width: 输入图像宽度
        pretrained: 是否使用预训练权重
    Returns:
        模型实例
    """
    model_map = {
        'resnet50_unet': resnet50_unet
    }
    
    if model_name not in model_map:
        raise ValueError(f'不支持的模型名称: {model_name}')
        
    return model_map[model_name](
        n_classes=n_classes,
        input_height=input_height,
        input_width=input_width,
        pretrained=pretrained
    ) 