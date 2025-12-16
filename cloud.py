#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在 GF-1 四波段影像中，只模拟“3~4片”独立的大片云。
主要做法：
1) 使用高阈值 + Perlin 噪声，初始云区更少
2) 大核形态学 (先闭后开)，让云合并成几大团并去除小碎片
3) 连通域：先剔除小面积，再只保留最大 N 个

依赖：numpy, opencv-python, gdal, noise
"""

import numpy as np
import cv2
from osgeo import gdal
import noise


# ========== 1. 读写影像函数 ==========

def read_gf1_4band(filepath):
    ds = gdal.Open(filepath)
    if ds is None:
        raise FileNotFoundError(f"无法打开影像: {filepath}")
    if ds.RasterCount < 4:
        raise ValueError("波段数不足4，无法作为GF-1四波段处理。")

    bands = []
    for i in range(4):
        arr = ds.GetRasterBand(i + 1).ReadAsArray()
        bands.append(arr)
    img_4band = np.dstack(bands).astype(np.float32)

    proj = ds.GetProjection()
    geo = ds.GetGeoTransform()
    ds = None
    return img_4band, proj, geo


def write_gf1_4band(filepath, image_4band, projection, geotransform):
    h, w, c = image_4band.shape
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(filepath, w, h, c, gdal.GDT_UInt16)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)

    clipped = np.clip(image_4band, 0, 10000).astype(np.uint16)
    for i in range(c):
        out_ds.GetRasterBand(i + 1).WriteArray(clipped[:, :, i])

    out_ds = None


# ========== 2. 生成 Perlin 噪声遮罩 ==========

def generate_perlin_mask(H, W, scale=80.0, octaves=3, threshold=0.7):
    """
    使用 Perlin 噪声获取 (H,W) 的灰度，再用 threshold 生成 0/1 遮罩。
    threshold 越高 -> 云越少。
    """
    perlin_data = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            x = j / scale
            y = i / scale
            val = 0.0
            freq = 1.0
            amp = 1.0
            persistence = 0.5
            for _ in range(octaves):
                v = noise.pnoise2(x * freq, y * freq)
                val += v * amp
                freq *= 2.0
                amp *= persistence
            # 映射到[0,1]
            perlin_data[i, j] = (val + 1.0) / 2.0

    mask = (perlin_data > threshold).astype(np.float32)
    return mask


# ========== 3. 形态学+连通域处理，只保留少数云 ==========

def remove_small_blobs(mask, min_area=5000):
    """
    连通域分析，剔除面积 < min_area 的云块。
    mask: (H,W), 0/1
    返回新的mask
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask

    # stats[i, cv2.CC_STAT_AREA] 表示第i个连通域的面积(i=0是背景)
    new_mask = np.zeros_like(mask, dtype=np.float32)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            new_mask[labels == i] = 1.0
    return new_mask


def keep_top_n_clouds(mask, top_n=3):
    """
    在二值mask里只保留面积最大的 top_n 个连通域，其余去掉。
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
    idxs = np.argsort(areas)[::-1]
    new_mask = np.zeros_like(mask, dtype=np.float32)
    for i in range(min(top_n, len(idxs))):
        lbl = idxs[i] + 1
        new_mask[labels == lbl] = 1.0
    return new_mask


# ========== 4. 综合流程函数 ==========

def simulate_few_big_clouds(
        image_4band,
        perlin_scale=80.0,
        perlin_octaves=3,
        perlin_threshold=0.7,
        morph_kernel_close=15,
        morph_kernel_open=7,
        min_area=5000,
        top_n=3,
        cloud_val=(9000, 9000, 9000, 9500)
):
    """
    目标: 最终只剩 3大块云 (top_n=3)
    步骤:
      1) Perlin 噪声 + 高阈值 -> 初步 mask
      2) 大核闭运算(合并云) -> 大核开运算(去掉小零散)
      3) 剔除小面积云块(min_area)
      4) 只保留面积最大的 top_n 云块
      5) 与原图像叠加
    """
    H, W, C = image_4band.shape
    # 1) Perlin 初步 mask
    raw_mask = generate_perlin_mask(H, W, perlin_scale, perlin_octaves, perlin_threshold)

    # 2) 形态学
    mask_u8 = (raw_mask * 255).astype(np.uint8)
    kernel_close = np.ones((morph_kernel_close, morph_kernel_close), np.uint8)
    kernel_open = np.ones((morph_kernel_open, morph_kernel_open), np.uint8)

    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    morph_mask = (opened / 255.0).astype(np.float32)

    # 3) 剔除小面积云
    cleaned_mask = remove_small_blobs(morph_mask, min_area=min_area)

    # 4) 保留最大 top_n 云
    final_mask = keep_top_n_clouds(cleaned_mask, top_n=top_n)

    # 5) 与原图叠加
    mask_3d = np.repeat(final_mask[:, :, np.newaxis], 4, axis=2)
    cloud_val_np = np.array(cloud_val, dtype=np.float32)
    cloud_val_4band = np.ones((H, W, 4), dtype=np.float32) * cloud_val_np

    base_float = image_4band.astype(np.float32)
    cloudy_image = base_float * (1 - mask_3d) + cloud_val_4band * mask_3d
    cloudy_image = np.clip(cloudy_image, 0, 10000)

    return cloudy_image, final_mask


# ========== 5. 主函数: 修改路径与参数后即可运行 ==========

def main():
    # (1) 输入输出影像
    input_path = r"F:/DL/input/20200407_GF.tif"
    output_path = r"F:/DL/input/20200407_GF_simcloud-1.tif"

    # 读取影像
    img_4band, proj, geo = read_gf1_4band(input_path)

    # (2) 参数设置
    perlin_scale = 80.0
    perlin_octaves = 3
    perlin_threshold = 0.7  # 0.7~0.8 云会非常少
    morph_kernel_close = 15
    morph_kernel_open = 7
    min_area = 5000  # 小于5000像素的云块直接丢掉
    top_n = 3  # 最终仅保留3块
    cloud_val = (9000, 9000, 9000, 9500)

    # (3) 执行流程
    cloudy_image, final_mask = simulate_few_big_clouds(
        image_4band=img_4band,
        perlin_scale=perlin_scale,
        perlin_octaves=perlin_octaves,
        perlin_threshold=perlin_threshold,
        morph_kernel_close=morph_kernel_close,
        morph_kernel_open=morph_kernel_open,
        min_area=min_area,
        top_n=top_n,
        cloud_val=cloud_val
    )

    # (4) 写出结果
    write_gf1_4band(output_path, cloudy_image, proj, geo)
    print("云模拟完成，输出:", output_path)


if __name__ == "__main__":
    main()
