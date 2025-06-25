import cv2
import numpy as np

def viscera_split_medical(mask):
    h, w = mask.shape
    # 找到mask的最小外接矩形并仿射拉正
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
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
    # 五区分割（医学标准）
    regions = {}
    # 舌尖（心肺）
    tip_h = int(height * 0.18)
    region_tip = np.zeros_like(mask_warp)
    cv2.ellipse(region_tip, (width//2, height-tip_h//2), (width//2-2, tip_h), 0, 0, 180, 255, -1)
    region_tip = cv2.bitwise_and(region_tip, mask_warp)
    # 舌根（肾）
    root_h = int(height * 0.22)
    region_root = np.zeros_like(mask_warp)
    cv2.ellipse(region_root, (width//2, root_h//2), (width//2-2, root_h), 0, 180, 360, 255, -1)
    region_root = cv2.bitwise_and(region_root, mask_warp)
    # 舌边（左肝/右肝）
    region_left = np.zeros_like(mask_warp)
    region_right = np.zeros_like(mask_warp)
    poly_left = np.array([
        [0, int(height*0.22)],
        [int(width*0.18), int(height*0.45)],
        [int(width*0.18), int(height*0.80)],
        [0, int(height*0.80)]
    ], dtype=np.int32)
    poly_right = np.array([
        [width-1, int(height*0.22)],
        [width-1-int(width*0.18), int(height*0.45)],
        [width-1-int(width*0.18), int(height*0.80)],
        [width-1, int(height*0.80)]
    ], dtype=np.int32)
    cv2.fillPoly(region_left, [poly_left], 255)
    cv2.fillPoly(region_right, [poly_right], 255)
    region_left = cv2.bitwise_and(region_left, mask_warp)
    region_right = cv2.bitwise_and(region_right, mask_warp)
    # 舌中（脾胃）
    region_mid = mask_warp.copy()
    for r in [region_tip, region_root, region_left, region_right]:
        region_mid = cv2.subtract(region_mid, r)
    # 逆变换回原图
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    def inv_warp(region):
        return cv2.warpPerspective(region, Minv, (w, h))
    regions['心肺'] = inv_warp(region_tip)
    regions['肾'] = inv_warp(region_root)
    regions['左肝'] = inv_warp(region_left)
    regions['右肝'] = inv_warp(region_right)
    regions['脾'] = inv_warp(region_mid)
    # 保证只在舌体mask范围内
    for k in regions:
        regions[k] = cv2.bitwise_and(regions[k], mask)
    return regions 