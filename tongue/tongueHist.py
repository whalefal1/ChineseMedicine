import cv2
import numpy as np
import os

def getStatistics(hist):
    # 最大值（峰值）
    peak = np.max(hist)
    # 均值
    mean = np.sum(hist * np.arange(len(hist))) / np.sum(hist) if np.sum(hist) > 0 else 0
    # 标准差宽度
    std = np.sqrt(np.sum(hist * (np.arange(len(hist)) - mean) ** 2) / np.sum(hist)) if np.sum(hist) > 0 else 0
    # 截断均值（去除两端各5%）
    total = np.sum(hist)
    cumsum = np.cumsum(hist)
    lower = np.searchsorted(cumsum, total * 0.05)
    upper = np.searchsorted(cumsum, total * 0.95)
    if upper > lower:
        trunc_mean = np.sum(hist[lower:upper] * np.arange(lower, upper)) / np.sum(hist[lower:upper])
    else:
        trunc_mean = mean
    # 最小值
    min_val = np.min(hist)
    # 最大值位置
    peak_pos = np.argmax(hist)
    return peak, std, mean, trunc_mean, min_val, peak_pos

def calcuVec(img, region_mask):
    features = []
    # 色彩空间转换
    color_spaces = [
        ('Lab', cv2.COLOR_BGR2Lab),
        ('HLS', cv2.COLOR_BGR2HLS),
        ('RGB', cv2.COLOR_BGR2RGB)
    ]
    for name, code in color_spaces:
        img_cs = cv2.cvtColor(img, code)
        for ch in range(3):
            channel = img_cs[:,:,ch]
            # 只统计mask区域
            values = channel[region_mask > 0]
            if len(values) == 0:
                hist = np.zeros(256)
            else:
                hist = cv2.calcHist([values], [0], None, [256], [0,256]).flatten()
            # 6个统计量
            peak, std, mean, trunc_mean, min_val, peak_pos = getStatistics(hist)
            features.extend([peak, std, mean, trunc_mean, min_val, peak_pos])
    # 若不足72维，补0
    if len(features) < 72:
        features.extend([0] * (72 - len(features)))
    return features[:72]

def getVec(img_path, mask_paths_dict):
    img = cv2.imread(img_path)
    if img is None:
        return None
    total_mask = None
    region_features = {}
    region_ratios = {}
    total_pixels = 0
    # 先统计总mask
    for region, mask_path in mask_paths_dict.items():
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            continue
        mask_bin = (mask > 0).astype(np.uint8)
        if total_mask is None:
            total_mask = np.zeros_like(mask_bin)
        total_mask = cv2.bitwise_or(total_mask, mask_bin)
    if total_mask is None or np.sum(total_mask) == 0:
        return None
    total_pixels = np.sum(total_mask)
    # 计算每个区域特征和占比
    for region, mask_path in mask_paths_dict.items():
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            continue
        mask_bin = (mask > 0).astype(np.uint8)
        region_pixels = np.sum(mask_bin)
        ratio = region_pixels / total_pixels if total_pixels > 0 else 0
        region_ratios[region] = ratio
        features = calcuVec(img, mask_bin)
        region_features[region] = features
    return region_features, region_ratios

def batch_getVec(img_dir, mask_dir, out_file='tongue_features.txt'):
    # 假设五区mask命名为：原图名_区域名.png
    region_names = ['心肺', '肾', '左肝', '右肝', '脾']
    results = []
    for filename in os.listdir(img_dir):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.bmp')):
            continue
        img_path = os.path.join(img_dir, filename)
        mask_paths_dict = {}
        base = os.path.splitext(filename)[0]
        for region in region_names:
            mask_name = f'{base}_{region}.png'
            mask_path = os.path.join(mask_dir, mask_name)
            mask_paths_dict[region] = mask_path
        result = getVec(img_path, mask_paths_dict)
        if result is None:
            print(f"{filename}: mask文件缺失或读取失败，跳过")
            continue
        features, ratios = result
        line = [filename]
        for region in region_names:
            line.append(ratios.get(region, 0))
            feats = features.get(region, [0]*72)
            line.extend(feats)
        results.append(line)
    # 保存
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write('\t'.join(map(str, line)) + '\n')
    print(f'特征值及占比已保存到 {out_file}')

if __name__ == '__main__':
    img_dir = 'tongue_data/tongue_data/test_img'  # 原图目录
    mask_dir = 'viscera_masks_medical'            # 五区mask目录
    batch_getVec(img_dir, mask_dir) 