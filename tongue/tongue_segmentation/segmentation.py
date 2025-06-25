import cv2
import numpy as np

# 获取舌头的最小正外接矩形
# 输入: mask（二值化舌体分割图）
# 输出: rect（最小正外接矩形的四个顶点坐标）
def get_tongue(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

# 二阶贝塞尔曲线
# 输入: p0, p1, p2 三个点
# 输出: 曲线上的点集

def Bezier2(p0, p1, p2, num=100):
    t = np.linspace(0, 1, num)
    curve = (1 - t)[:, None] ** 2 * p0 + 2 * (1 - t)[:, None] * t[:, None] * p1 + t[:, None] ** 2 * p2
    return curve.astype(np.int32)

# 基于最小外接矩形的五区分割
# 输入: mask（二值化舌体分割图）
# 输出: 区域字典，key为脏器名，value为区域mask

def viscera_split(mask):
    h, w = mask.shape
    box = get_tongue(mask)
    if box is None:
        return None
    # 计算最小外接矩形的宽高和角度
    rect = cv2.minAreaRect(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width == 0 or height == 0:
        return None
    # 目标点（拉正后矩形的四个角）
    dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    # box顺序调整为左上、右上、右下、左下
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    src_pts = order_points(box)
    # 仿射变换拉正
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    mask_warp = cv2.warpPerspective(mask, M, (width, height))
    # 在拉正后的矩形内分区
    regions = {}
    # 医学常规比例
    tip_h = int(height * 0.18)      # 舌尖（心肺）约18%
    mid_h = int(height * 0.32)      # 舌中（脾）约32%
    root_h = height - tip_h - mid_h # 舌根（肾）剩余
    # 舌尖（心肺）
    region_tip = np.zeros_like(mask_warp)
    region_tip[:tip_h, :] = mask_warp[:tip_h, :]
    # 舌中（脾）
    region_mid = np.zeros_like(mask_warp)
    region_mid[tip_h:tip_h+mid_h, :] = mask_warp[tip_h:tip_h+mid_h, :]
    # 舌根（肾）
    region_root = np.zeros_like(mask_warp)
    region_root[tip_h+mid_h:, :] = mask_warp[tip_h+mid_h:, :]
    # 左肝
    region_left = np.zeros_like(mask_warp)
    region_left[:, :width//2] = mask_warp[:, :width//2]
    # 右肝
    region_right = np.zeros_like(mask_warp)
    region_right[:, width//2:] = mask_warp[:, width//2:]
    # 逆变换回原图
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    def inv_warp(region):
        return cv2.warpPerspective(region, Minv, (w, h))
    regions['心肺'] = inv_warp(region_tip)
    regions['脾'] = inv_warp(region_mid)
    regions['肾'] = inv_warp(region_root)
    regions['左肝'] = inv_warp(region_left)
    regions['右肝'] = inv_warp(region_right)
    # 保证只在舌体mask范围内
    for k in regions:
        regions[k] = cv2.bitwise_and(regions[k], mask)
    return regions 