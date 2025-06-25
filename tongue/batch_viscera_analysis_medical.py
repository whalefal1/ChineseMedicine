import os
import cv2
import numpy as np
from tongue.tongue_segmentation.viscera_split_medical import viscera_split_medical

VISCERA_COLORS = {
    '心肺': (255, 0, 0),   # 红
    '脾':   (0, 255, 0),   # 绿
    '肾':   (0, 0, 255),   # 蓝
    '左肝': (255, 255, 0), # 黄
    '右肝': (255, 0, 255), # 紫
}

def visualize_regions(mask, regions):
    color_img = np.stack([mask]*3, axis=-1)
    color_img = (color_img > 0).astype(np.uint8) * 80  # 基础灰色
    for name, region in regions.items():
        color = VISCERA_COLORS.get(name, (255,255,255))
        for c in range(3):
            color_img[:,:,c][region>0] = color[c]
    return color_img

def batch_viscera_analysis_medical(prediction_dir, out_dir_mask, out_dir_vis):
    os.makedirs(out_dir_mask, exist_ok=True)
    os.makedirs(out_dir_vis, exist_ok=True)
    for filename in os.listdir(prediction_dir):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.bmp')):
            continue
        mask_path = os.path.join(prediction_dir, filename)
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"无法读取: {mask_path}")
            continue
        mask_bin = (mask > 0).astype('uint8') * 255
        regions = viscera_split_medical(mask_bin)
        if regions is None:
            print(f"分区失败: {filename}")
            continue
        # 保存五区mask
        for name, region in regions.items():
            save_path = os.path.join(out_dir_mask, f"{os.path.splitext(filename)[0]}_{name}.png")
            cv2.imwrite(save_path, region)
        # 可视化叠加
        vis_img = visualize_regions(mask_bin, regions)
        vis_save_path = os.path.join(out_dir_vis, filename)
        cv2.imwrite(vis_save_path, vis_img)
        print(f"已处理: {filename}")

if __name__ == '__main__':
    prediction_dir = 'prediction_bin'
    out_dir_mask = 'viscera_masks_medical'
    out_dir_vis = 'viscera_visualization_medical'
    batch_viscera_analysis_medical(prediction_dir, out_dir_mask, out_dir_vis) 