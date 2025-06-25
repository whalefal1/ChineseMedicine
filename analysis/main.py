import os
import numpy as np
from analysis.merge_features import load_weights, merge_feature, scaler_feature, merge_region, health_score

# 修正特征文件路径为项目根目录下
tongue_features_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tongue_features.txt')
OUTPUT_FILE = 'health_scores.txt'
REGION_NAMES = ['心肺', '肾', '左肝', '右肝', '脾']


def Main():
    weights = load_weights()
    results = []
    with open(tongue_features_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            filename = items[0]
            region_ratios = {region: float(items[1 + i * 1]) for i, region in enumerate(REGION_NAMES)}
            region_features = {}
            region_scores = {}
            for i, region in enumerate(REGION_NAMES):
                feats = [float(x) for x in items[1 + 5 + i * 12:1 + 5 + (i + 1) * 12]]
                region_features[region] = feats
                region_scores[region] = merge_feature(feats, weights[region])
            # 区域得分归一化
            norm_scores = scaler_feature(list(region_scores.values()))
            norm_region_scores = {region: norm_scores[i] for i, region in enumerate(REGION_NAMES)}
            # 融合整体得分
            total_score = merge_region(norm_region_scores, region_ratios)
            health = health_score(total_score)
            results.append([filename] + [norm_region_scores[region] for region in REGION_NAMES] + [health])
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('文件名\t心肺\t肾\t左肝\t右肝\t脾\t健康值\n')
        for line in results:
            f.write('\t'.join(map(str, line)) + '\n')
    print(f'健康值得分已保存到 {OUTPUT_FILE}')

if __name__ == '__main__':
    Main() 