import cv2
import numpy as np
import os

def calcuAera(img_path, mask_path):
    # 读取原图和mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    if img is None or mask is None:
        return None, None, None
    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 整体亮度
    mean_brightness = np.mean(gray)
    # 舌体mask面积比例
    mask_bin = (mask > 0).astype(np.uint8)
    tongue_area = np.sum(mask_bin)
    total_area = mask_bin.shape[0] * mask_bin.shape[1]
    area_ratio = tongue_area / total_area
    # 舌体区域亮度
    if tongue_area > 0:
        tongue_brightness = np.sum(gray * mask_bin) / tongue_area
    else:
        tongue_brightness = 0
    return mean_brightness, tongue_brightness, area_ratio

def haveTongue(img_path, mask_path, min_ratio=0.01, max_ratio=0.5):
    mean_brightness, tongue_brightness, area_ratio = calcuAera(img_path, mask_path)
    if mean_brightness is None:
        return False, '图像或mask读取失败'
    if mean_brightness < 50:
        return False, '图像整体过暗'
    if mean_brightness > 200:
        return False, '图像整体过亮'
    if tongue_brightness < 50:
        return False, '舌体区域过暗'
    if tongue_brightness > 200:
        return False, '舌体区域过亮'
    if area_ratio < min_ratio:
        return False, '未检测到舌体或舌体面积过小'
    if area_ratio > max_ratio:
        return False, '舌体面积过大或不完整'
    return True, '质量合格'

def batch_check_quality(img_dir, mask_dir, min_ratio=0.01, max_ratio=0.5, result_file='tongue_quality_check.txt'):
    results = []
    for filename in os.listdir(mask_dir):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.bmp')):
            continue
        mask_path = os.path.join(mask_dir, filename)
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            results.append((filename, False, '原图不存在'))
            continue
        ok, reason = haveTongue(img_path, mask_path, min_ratio, max_ratio)
        results.append((filename, ok, reason))
        print(f'{filename}: {reason}')
    # 保存结果
    with open(result_file, 'w', encoding='utf-8') as f:
        for filename, ok, reason in results:
            f.write(f'{filename}\t{"合格" if ok else "不合格"}\t{reason}\n')
    print(f'检测结果已保存到 {result_file}')

if __name__ == '__main__':
    # 假设原图和mask文件名一致
    img_dir = 'tongue_data/tongue_data/test_img'  # 或实际原图目录
    mask_dir = 'prediction_bin'
    batch_check_quality(img_dir, mask_dir) 