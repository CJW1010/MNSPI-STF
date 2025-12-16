# -*- coding: utf-8 -*-
"""
Accuracy metrics for fused imagery — fixed to 4 bands (1..4).
- 仅排除 Alpha / Palette；Undefined 视为可用
- 强制评估第 1–4 波段；任一影像缺失可用的 1..4 波段时给出明确报错
- 若检测到 DN(>1.5)，自动按 DN_max 归一化到 0–1 再评估
- NoData + 合理范围联合掩膜，只在有效像元上算 AD / RMSE / EDGE / LBP
- EDGE: 取边缘强度 Top 10% 像元做 NDSI；LBP：稳定实现（无 round，支持容差）
- 输出 Excel：band1..band4 + bandAverage（与融合影像同目录）
"""

from osgeo import gdal, gdalconst
import numpy as np
import pandas as pd
import scipy.signal as ss
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import os

# =========================
# 全局参数（按需调整）
# =========================
DN_min = 0.0  # 有效像元下界（DN 或反射率）
DN_max = 10000.0  # 若像元最大值>1.5，视为 DN 域（0..10000），用于归一化到 0–1
LBP_TOL = 5e-5  # 0–1 域建议 0~5e-5；若在 DN 域评估需放大到 ~0.3–1.0
EDGE_TOPQ = 0.90  # 边缘强度 Top 分位（0.9=前 10%）
FORCE_BANDS = [1, 2, 3, 4]  # 固定评估 1..4 波段


# =========================
# 读取影像：仅排除 Alpha/Palette（Undefined 视为可用）
# =========================
def GetData_4bands(path, force_bands):
    """
    返回: (W,H,B,GeoTrans,Proj,data[B,H,W], nodata_list, band_ids)
    - 排除 Alpha/Palette；Undefined/Gray/R/G/B/NIR 均视为可用
    - 强制读取 force_bands（如 [1,2,3,4]）
    - 若指定 band 超范围或是 Alpha/Palette，则报错
    """
    ds = gdal.Open(path, gdalconst.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open raster: {path}")

    W, H, B_all = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    gt, prj = ds.GetGeoTransform(), ds.GetProjection()

    # 标记不可用波段（仅 Alpha / Palette）
    unusable = set()
    band_ci = {}
    for b in range(1, B_all + 1):
        ci = ds.GetRasterBand(b).GetColorInterpretation()
        band_ci[b] = ci
        if ci in (gdalconst.GCI_AlphaBand, gdalconst.GCI_PaletteIndex):
            unusable.add(b)

    # 校验并读取
    for b in force_bands:
        if not (1 <= b <= B_all):
            raise ValueError(f"{os.path.basename(path)} 没有第 {b} 波段（总计 {B_all} 波段）。")
        if b in unusable:
            ci_map = {
                gdalconst.GCI_Undefined: "Undefined",
                gdalconst.GCI_GrayIndex: "Gray",
                gdalconst.GCI_PaletteIndex: "Palette",
                gdalconst.GCI_RedBand: "Red",
                gdalconst.GCI_GreenBand: "Green",
                gdalconst.GCI_BlueBand: "Blue",
                gdalconst.GCI_AlphaBand: "Alpha",
                gdalconst.GCI_NIRBand: "NIR",
            }
            ci_name = ci_map.get(band_ci[b], str(band_ci[b]))
            raise ValueError(
                f"{os.path.basename(path)} 的第 {b} 波段 ColorInterp={ci_name}（Alpha/Palette 不可用）。"
                f"请先用 gdal_translate 过滤掉 Alpha/Palette 后再评估。"
            )

    B = len(force_bands)
    data = np.zeros((B, H, W), dtype=np.float32)
    nodata = []
    for i, b in enumerate(force_bands):
        rb = ds.GetRasterBand(b)
        arr = rb.ReadAsArray().astype(np.float32)
        data[i] = arr
        nodata.append(rb.GetNoDataValue())

    return W, H, B, gt, prj, data, nodata, list(force_bands)


# =========================
# 工具函数
# =========================
def unify_to_01(arr, dn_max=DN_max):
    """若数组最大值 > 1.5，视为 DN 域并 /dn_max 映射到 0..1；否则原样返回。"""
    amax = np.nanmax(arr)
    if amax > 1.5:
        return arr / float(dn_max)
    return arr


def build_valid_mask(arr_ref, arr_pred, nd_ref, nd_pred, use_norm=True):
    """
    构建有效像元掩膜（True=无效）：
    - 合并两图的 NoData
    - 检查数值范围（归一化域：0..1；DN 域：DN_min..DN_max）
    - 排除非有限值
    """
    B, H, W = arr_ref.shape
    mask = np.zeros((H, W), dtype=bool)

    for i in range(B):
        if nd_ref[i] is not None:
            mask |= (arr_ref[i] == nd_ref[i])
        if nd_pred[i] is not None:
            mask |= (arr_pred[i] == nd_pred[i])

    if use_norm:
        low, high = 0.0, 1.0
    else:
        low, high = DN_min, DN_max

    mask |= ~np.all((arr_ref >= low) & (arr_ref <= high), axis=0)
    mask |= ~np.all((arr_pred >= low) & (arr_pred <= high), axis=0)
    mask |= ~np.isfinite(np.sum(arr_ref, axis=0))
    mask |= ~np.isfinite(np.sum(arr_pred, axis=0))
    return mask  # True=无效


def roberts_edge(img2d):
    """Roberts 交叉算子，返回与输入同尺寸的边缘强度图。"""
    img2d = img2d.astype(np.float32, copy=False)
    k1 = np.array([[1, 0], [0, -1]], dtype=np.float32)
    k2 = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    g1 = ss.convolve(img2d, k1, mode='valid')  # (H-1, W-1)
    g2 = ss.convolve(img2d, k2, mode='valid')
    out = np.zeros_like(img2d, dtype=np.float32)
    # 左上对齐（也可改 out[1:,1:] = ... 做右下对齐）
    out[:-1, :-1] = np.abs(g1) + np.abs(g2)
    return out


def lbp_8nb(img2d, tol=0.0):
    """
    3x3 8邻域 LBP（顺时针位权：1,2,4,8,16,32,64,128），无 round，支持容差。
    返回 0..255 的 float32 码图。
    """
    img2d = img2d.astype(np.float32, copy=False)
    H, W = img2d.shape
    out = np.zeros_like(img2d, dtype=np.float32)
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    center = img2d[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint16)
    for (dy, dx), w in zip(offsets, weights):
        nbr = img2d[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
        bit = (nbr > (center + tol)).astype(np.uint16)
        code += w * bit
    out[1:-1, 1:-1] = code.astype(np.float32)
    return out  # 0..255


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # 可加：gdal.DontUseExceptions() 来安静掉未来异常策略提示
    root = tk.Tk();
    root.withdraw()
    ref_path = filedialog.askopenfilename(title='open the reference fine-resolution image')
    pred_path = filedialog.askopenfilename(title='open the fused fine-resolution image')

    t0 = datetime.now()

    # 读取两幅影像（强制 1..4 波段；仅排除 Alpha/Palette）
    W1, H1, B1, gt1, prj1, ref_raw, nd_ref, bands_ref = GetData_4bands(ref_path, FORCE_BANDS)
    W2, H2, B2, gt2, prj2, pred_raw, nd_pred, bands_pred = GetData_4bands(pred_path, FORCE_BANDS)

    # 基本检查
    if (W1 != W2) or (H1 != H2):
        raise ValueError(f"Spatial size mismatch: Ref({H1},{W1}) vs Pred({H2},{W2}). 请先严格同栅格对齐。")
    if bands_ref != bands_pred or B1 != 4 or B2 != 4:
        # 正常不会触发；GetData_4bands 已经保证二者都有 1..4 四个可用波段
        raise ValueError("两幅影像的波段集合不一致或不是 4 个。请检查输入。")

    # 统一到 0–1 域（若检测为 DN）
    ref = unify_to_01(ref_raw, dn_max=DN_max)
    pred = unify_to_01(pred_raw, dn_max=DN_max)

    # 有效掩膜（True=无效）
    mask_invalid = build_valid_mask(ref, pred, nd_ref, nd_pred, use_norm=True)
    valid = ~mask_invalid

    # 结果表：band1..band4 + bandAverage；列：AD, RMSE, EDGE, LBP
    results = np.zeros((4 + 1, 4), dtype=np.float64)

    # ===== AD / RMSE =====
    for i in range(4):
        x = ref[i][valid]
        y = pred[i][valid]
        if x.size == 0:
            ad, rmse = 0.0, 0.0
        else:
            d = y - x
            ad = float(np.mean(d))
            rmse = float(np.sqrt(np.mean(d * d)))
        results[i, 0] = ad
        results[i, 1] = rmse

    # ===== EDGE =====
    for i in range(4):
        te = roberts_edge(ref[i])
        pe = roberts_edge(pred[i])
        # 与 roberts 输出左上对齐保持一致，内部再裁一圈更稳
        ti = te[0:-1, 0:-1]
        pi = pe[0:-1, 0:-1]
        vm = mask_invalid[0:-1, 0:-1]
        use = (~vm) & np.isfinite(ti) & np.isfinite(pi)
        if np.count_nonzero(use) < 100:
            results[i, 2] = 0.0
        else:
            x = ti[use].ravel()
            y = pi[use].ravel()
            thr = np.quantile(x, EDGE_TOPQ)
            sel = x >= thr
            if np.count_nonzero(sel) < 50:
                results[i, 2] = 0.0
            else:
                xs = x[sel];
                ys = y[sel]
                ndsi = (ys - xs) / (np.abs(ys + xs) + 1e-5)
                results[i, 2] = float(np.mean(ndsi))

    # ===== LBP =====
    for i in range(4):
        tl = lbp_8nb(ref[i], tol=LBP_TOL) / 255.0
        pl = lbp_8nb(pred[i], tol=LBP_TOL) / 255.0
        # 再去掉一圈，避免边界效应
        ti = tl[1:-1, 1:-1]
        pi = pl[1:-1, 1:-1]
        vm = mask_invalid[1:-1, 1:-1]
        use = (~vm) & np.isfinite(ti) & np.isfinite(pi)
        if np.count_nonzero(use) < 100:
            results[i, 3] = 0.0
        else:
            x = ti[use].ravel()
            y = pi[use].ravel()
            ndsi = (y - x) / (np.abs(y + x) + 1e-5)
            results[i, 3] = float(np.mean(ndsi))

    # 平均行
    results[4, :] = np.mean(results[0:4, :], axis=0)

    # 保存 Excel
    idx = [f"band{x}" for x in range(1, 5)] + ["bandAverage"]
    df = pd.DataFrame(results, index=idx, columns=["AD", "RMSE", "EDGE", "LBP"])
    out_xlsx = os.path.join(os.path.dirname(pred_path),
                            os.path.splitext(os.path.basename(pred_path))[0] + "_accuracy.xlsx")
    df.to_excel(out_xlsx)

    # 打印
    print("accuracy assessment results: AD, RMSE, EDGE, LBP")
    np.set_printoptions(suppress=True)
    print(results)

    t1 = datetime.now()
    print("time used: {} h {} m {} s".format(
        (t1 - t0).seconds // 3600, (t1 - t0).seconds % 3600 // 60, (t1 - t0).seconds % 60
    ))
    print("Saved:", out_xlsx)
