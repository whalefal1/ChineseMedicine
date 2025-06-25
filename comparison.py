import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

TEST_IMG_DIR = 'tongue_data/tongue_data/test_img'
PREDICTION_DIR = 'prediction'
COMPARE_DIR = 'compare'

os.makedirs(COMPARE_DIR, exist_ok=True)

def main():
    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.bmp', '.png'))]
    for fname in img_files:
        img_path = os.path.join(TEST_IMG_DIR, fname)
        pred_path = os.path.join(PREDICTION_DIR, fname)
        if not os.path.exists(pred_path):
            continue
        img = cv2.imread(img_path)
        pred = cv2.imread(pred_path)
        if img is None or pred is None:
            continue
        # 调整预测图像尺寸与原图一致
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]))
        # 拼接
        compare_img = np.concatenate([img, pred], axis=1)
        save_path = os.path.join(COMPARE_DIR, fname)
        cv2.imwrite(save_path, compare_img)

if __name__ == '__main__':
    main() 