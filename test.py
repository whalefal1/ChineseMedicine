import os
from keras_segmentation.predict import model_from_checkpoint_path, predict, evaluate

# 路径配置
CHECKPOINTS_PATH = 'weights'
TEST_IMG_DIR = 'tongue_data/tongue_data/test_img'
TEST_LABEL_DIR = 'tongue_data/tongue_data/test_label'
PREDICTION_DIR = 'prediction'

os.makedirs(PREDICTION_DIR, exist_ok=True)

def main():
    # 加载模型和配置
    model, config = model_from_checkpoint_path(CHECKPOINTS_PATH)
    # 批量预测并保存可视化结果
    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.bmp', '.png'))]
    for fname in img_files:
        img_path = os.path.join(TEST_IMG_DIR, fname)
        save_path = os.path.join(PREDICTION_DIR, fname)
        predict(model, config, img_path, save_path=save_path)
    # 评估mIoU
    evaluate(model, config, TEST_IMG_DIR, TEST_LABEL_DIR)

if __name__ == '__main__':
    main() 