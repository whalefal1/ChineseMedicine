import os
import cv2
import numpy as np

def extract_tongue_mask(img):
    if len(img.shape) == 2:
        # 灰度图，直接阈值
        _, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        return mask
    elif len(img.shape) == 3:
        # 彩色图，尝试提取紫色区域
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 紫色HSV范围（可根据实际分割色调整）
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        # 若紫色区域太少，尝试提取最大连通域
        if np.sum(mask) < 100:
            # 取所有非黑色区域
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # 形态学操作去噪
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        return mask
    else:
        return None

def main():
    src_dir = 'prediction'
    dst_dir = 'prediction_bin'
    os.makedirs(dst_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.png') or filename.lower().endswith('.bmp')):
            continue
        img_path = os.path.join(src_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'无法读取: {filename}')
            continue
        mask = extract_tongue_mask(img)
        if mask is None or np.sum(mask) == 0:
            print(f'未检测到舌体区域: {filename}')
            continue
        # 保证只有0和255
        mask = (mask > 0).astype(np.uint8) * 255
        out_path = os.path.join(dst_dir, filename)
        cv2.imwrite(out_path, mask)
        print(f'已生成标准二值mask: {filename}')

if __name__ == '__main__':
    main() 