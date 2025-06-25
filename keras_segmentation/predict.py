import os
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
from .models.all_models import get_model
from .data_utils.data_loader import get_image_array

def model_from_checkpoint_path(checkpoints_path):
    """
    加载最新权重和配置，返回模型
    """
    config_path = os.path.join(checkpoints_path, 'resunet_config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'未找到模型配置文件: {config_path}')
    with open(config_path, 'r') as f:
        config = json.load(f)
    # 查找最新权重
    ckpts = [f for f in os.listdir(checkpoints_path) if f.endswith('.pth')]
    if not ckpts:
        raise FileNotFoundError('未找到模型权重文件')
    epochs = [int(''.join(filter(str.isdigit, f))) for f in ckpts]
    latest_idx = np.argmax(epochs)
    latest_ckpt = os.path.join(checkpoints_path, ckpts[latest_idx])
    # 加载模型
    model = get_model(
        model_name=config['model_name'],
        n_classes=config['n_classes'],
        input_height=config['input_height'],
        input_width=config['input_width'],
        pretrained=False
    )
    model.load_state_dict(torch.load(latest_ckpt, map_location='cpu'))
    model.eval()
    return model, config

def get_colored_segmentation_image(seg, color=(0, 0, 255)):
    """
    将二值分割标签转为彩色图像，舌体区域高亮
    """
    seg = seg.squeeze()
    color_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    color_img[seg > 0.5] = color
    return color_img

def visualize_segmentation(image, seg):
    """
    将分割结果叠加到原图上
    """
    if image.max() > 1:
        image = image / 255.0
    seg_color = get_colored_segmentation_image(seg)
    if seg_color.shape[:2] != image.shape[:2]:
        seg_color = cv2.resize(seg_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = (image * 0.7 + seg_color / 255.0 * 0.3) * 255
    return vis.astype(np.uint8)

def predict(model, config, image_path, save_path=None):
    """
    对单张图片进行分割预测并可视化
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_arr = get_image_array(image_path, config['input_width'], config['input_height'], imgNorm="divide")
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()
    with torch.no_grad():
        output = model(img_tensor)
        if output.shape[-2:] != img_tensor.shape[-2:]:
            output = torch.nn.functional.interpolate(output, size=img_tensor.shape[-2:], mode='bilinear', align_corners=True)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        seg = (pred > 0.5).astype(np.uint8)
    vis = visualize_segmentation(image_rgb, seg)
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return seg, vis

def evaluate(model, config, img_dir, label_dir):
    """
    计算mIoU等分割指标
    """
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.bmp', '.png'))]
    ious = []
    for fname in tqdm(img_files, desc='评估中'):
        img_path = os.path.join(img_dir, fname)
        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(label_path):
            continue
        seg, _ = predict(model, config, img_path)
        label = cv2.imread(label_path, 0)
        label = cv2.resize(label, (config['input_width'], config['input_height']), interpolation=cv2.INTER_NEAREST)
        label = (label > 0).astype(np.uint8)
        intersection = np.logical_and(seg, label).sum()
        union = np.logical_or(seg, label).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)
    miou = np.mean(ious) if ious else 0
    print(f"mIoU: {miou:.4f}")
    return miou 