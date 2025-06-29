import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_weights(csv_path=None):
    import pandas as pd
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tongue_data', 'tongue_data', 'LS.csv')
    arr = pd.read_csv(csv_path, header=None).values.flatten()
    region_names = ['心肺', '肾', '左肝', '右肝', '脾']
    n_feat = 12  # 每区12维特征
    weights = {region: arr[i*n_feat:(i+1)*n_feat] for i, region in enumerate(region_names)}
    return weights

def merge_feature(features, weights):
    n = min(len(features), len(weights))
    return float(np.sum(np.multiply(features[:n], weights[:n])))

def scaler_feature(scores):
    scores = np.array(scores).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(scores)
    return scaled.flatten()

def merge_region(region_scores, region_ratios):
    total = 0
    for region, score in region_scores.items():
        total += score * region_ratios.get(region, 0)
    return total

def health_score(score):
    # 归一化到[-1, 1]
    return score * 2 - 1 