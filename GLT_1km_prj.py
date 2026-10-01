# -*- coding: utf-8 -*-
import numpy as np
import os
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# ==================== 尝试导入tifffile ====================
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("警告: 未安装tifffile，将使用numpy保存为raw格式")
    print("请安装: pip install tifffile")


# ==================== 读写 GeoTIFF ====================
def read_geotiff(filename):
    """使用tifffile读取GeoTIFF文件"""
    if not HAS_TIFFFILE:
        raise ImportError("请安装tifffile: pip install tifffile")
    data = tifffile.imread(filename)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data


def write_geotiff(filename, data, geotransform=None, epsg=4326, dtype=None,
                  nodata=-1):
    """
    使用tifffile写入GeoTIFF，包含地理坐标信息。

    注意：TIFF 的 ResolutionUnit 只接受 1/2/3，不支持 'DEGREE'，
    所以这里不传 resolutionunit，地理信息完全靠 GeoTIFF 的
    ModelPixelScale(33550) / ModelTiepoint(33922) / GeoKeyDirectory(34735) 表达。
    """
    if not HAS_TIFFFILE:
        np.save(filename.replace('.tif', '.npy'), data)
        print(f"  保存为numpy格式: {filename.replace('.tif', '.npy')}")
        return

    data = np.asarray(data)
    if dtype is not None:
        data = data.astype(dtype)

    # 构建 GeoTIFF 标签
    extratags = []
    if geotransform is not None:
        gt = [float(v) for v in geotransform]
        # 模型像素尺度 (x, y, z)
        extratags.append((33550, 12, 3, (gt[1], abs(gt[5]), 0.0), False))
        # 模型 TiePoint (i, j, k, x, y, z)
        extratags.append((33922, 12, 6, (0.0, 0.0, 0.0, gt[0], gt[3], 0.0), False))
        # GeoKeyDirectory
        # 1024: GTModelTypeGeoKey = 2 (ModelTypeGeographic)
        # 1025: GTRasterTypeGeoKey = 1 (RasterPixelIsArea)
        # 2048: GeographicTypeGeoKey = EPSG (通常 4326)
        geo_keys = (
            1, 1, 0, 3,
            1024, 0, 1, 2,
            1025, 0, 1, 1,
            2048, 0, 1, int(epsg)
        )
        extratags.append((34735, 3, len(geo_keys), geo_keys, False))
        # 角度单位: degree (9102)
        extratags.append((34736, 12, 1, (1.0,), False))

    try:
        try:
            import imagecodecs  # noqa
            compression = 'lzw'
        except ImportError:
            compression = None

        tifffile.imwrite(
            filename,
            data,
            photometric='minisblack',
            planarconfig='contig',
            compression=compression,
            extratags=extratags,
            # 关键：不要传 resolutionunit='DEGREE'，TIFF 不支持
        )
        print(f"  已保存GeoTIFF: {filename}")
    except Exception as e:
        print(f"  保存GeoTIFF失败: {e}")
        npy_file = filename.replace('.tif', '.npy')
        np.save(npy_file, data)
        print(f"  已降级保存为numpy格式: {npy_file}")


# ==================== 核心：最邻近插值 ====================
def build_lookup_nearest(lat_data, lon_data, valid_mask,
                         target_lon_grid, target_lat_grid):
    """
    使用 cKDTree 做最邻近查询，生成 目标经纬度 -> 原始行列号 的查找表。

    说明：
        - 用 cKDTree 而不是 griddata(method='nearest')，因为后者内部
          也会先做 Delaunay 剖分，对 9000 万点太慢。
        - cKDTree.query 支持 workers=-1 多核并行。

    Args:
        lat_data, lon_data: 原始经纬度查找表 (2D)
        valid_mask: 有效像元掩码
        target_lon_grid, target_lat_grid: 目标网格 (2D)

    Returns:
        row_out, col_out: 2D int32 数组，无效处为 -1
    """
    nrows, ncols = target_lon_grid.shape
    total = nrows * ncols

    # 有效点
    vi = np.where(valid_mask)
    v_rows = vi[0].astype(np.int32)
    v_cols = vi[1].astype(np.int32)

    # 经度统一到 [0, 360)，避免跨日期变更线断裂
    v_lons_raw = lon_data[valid_mask].astype(np.float64)
    v_lats = lat_data[valid_mask].astype(np.float64)
    v_lons_u = np.where(v_lons_raw < 0, v_lons_raw + 360.0, v_lons_raw)

    n_valid = len(v_lons_u)
    print(f"  有效点数: {n_valid:,}")
    if n_valid < 1:
        raise RuntimeError("有效点太少")

    # 构建 cKDTree
    import time
    src = np.column_stack((v_lons_u, v_lats))
    print(f"  构建 cKDTree（{n_valid:,} 点）...")
    t0 = time.time()
    tree = cKDTree(src)
    print(f"  cKDTree 完成，耗时 {time.time()-t0:.1f}s")
    del src, v_lons_u, v_lats, v_lons_raw
    import gc; gc.collect()

    # 输出（int32）
    row_out = np.full((nrows, ncols), -1, dtype=np.int32)
    col_out = np.full((nrows, ncols), -1, dtype=np.int32)

    # 分块参数
    chunk_pixels = 2_000_000
    chunk_rows = max(1, chunk_pixels // ncols)
    num_chunks = (nrows + chunk_rows - 1) // chunk_rows

    print(f"  目标点数: {total:,}")
    print(f"  分块: {num_chunks} 块 (每块 {chunk_rows} 行, "
          f"约 {chunk_rows*ncols:,} 点/块)")
    print(f"  {'='*70}")

    # 目标 1D 坐标（经度统一到 [0, 360)）
    lon_row = target_lon_grid[0, :]
    lon_row_u = np.where(lon_row < 0, lon_row + 360.0, lon_row)
    lat_col = target_lat_grid[:, 0]

    # 进度条
    def _print_progress(done, total_n, t_start, prefix='  插值'):
        elapsed = time.time() - t_start
        pct = done / total_n * 100
        if done > 0:
            eta = elapsed / done * (total_n - done)
            if eta > 3600:
                eta_str = f"{eta/3600:.1f}h"
            elif eta > 60:
                eta_str = f"{eta/60:.1f}m"
            else:
                eta_str = f"{eta:.0f}s"
        else:
            eta_str = "?"
        bar_len = 30
        filled = int(bar_len * done / total_n)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  {prefix} [{bar}] {pct:5.1f}%  "
              f"({done:,}/{total_n:,})  已用 {elapsed:.0f}s  剩余 ~{eta_str}",
              end='', flush=True)

    t_start = time.time()

    for ci in range(num_chunks):
        r0 = ci * chunk_rows
        r1 = min(r0 + chunk_rows, nrows)

        # 构造本块目标点
        blk_lat = lat_col[r0:r1]
        blk_lon_grid, blk_lat_grid = np.meshgrid(lon_row_u, blk_lat)
        tgt = np.column_stack((blk_lon_grid.ravel(),
                                blk_lat_grid.ravel()))

        # 最邻近查询：k=1，workers=-1 多核并行
        dist, idx = tree.query(tgt, k=1, workers=-1)

        # 通过索引取回原始行列号
        rv = v_rows[idx]
        cv = v_cols[idx]

        # 写回
        row_out[r0:r1, :] = rv.reshape(r1 - r0, ncols)
        col_out[r0:r1, :] = cv.reshape(r1 - r0, ncols)

        del tgt, rv, cv, idx, dist
        del blk_lon_grid, blk_lat_grid, blk_lat

        done = r1 * ncols
        _print_progress(done, total, t_start)

    print(f"\n  {'='*70}")
    print(f"  插值完成，总耗时 {time.time()-t_start:.1f}s")

    del tree, v_rows, v_cols
    gc.collect()

    return row_out, col_out


# ==================== 主流程 ====================
def create_equirectangular_lookup():
    """
    从标称投影的经纬度查找表生成等经纬投影的查找表。
    输出像元值为原始标称查找表的行列号（整数）。
    """

    # ==================== 硬编码路径 ====================
    LAT_TIF = r"D:\FY_Download\FY4B-_DISK_1330E_GEO_NOM_LUT_20220323000000_1000M_V0001\FY4B-_DISK_1330E_GEO_NOM_LUT_20220323000000_1000M_V0001\FY4B_GEO_LUT_1km_Latitude.tif"
    LON_TIF = r"D:\FY_Download\FY4B-_DISK_1330E_GEO_NOM_LUT_20220323000000_1000M_V0001\FY4B-_DISK_1330E_GEO_NOM_LUT_20220323000000_1000M_V0001\FY4B_GEO_LUT_1km_Longitude.tif"
    OUTPUT_DIR = r"D:\FY_Download\FY4B-_DISK_1330E_GEO_NOM_LUT_20220323000000_1000M_V0001\Output2"

    # ==================== 参数 ====================
    RESOLUTION_DEGREE = 0.011
    # [min_lon, max_lon, min_lat, max_lat]
    # min_lon > max_lon 视为跨越国际日期变更线
    TARGET_EXTENT = [20, -140, -90, 90]
    # NoData 填充值
    NODATA = -1

    # ==================== 输出文件名 ====================
    resolution_str = f"{RESOLUTION_DEGREE:.4f}".replace('.', '_')

    if TARGET_EXTENT is not None:
        min_lon, max_lon, min_lat, max_lat = [float(v) for v in TARGET_EXTENT]
        crosses_dateline = min_lon > max_lon
        if crosses_dateline:
            extent_str = (f"Lon{int(min_lon)}_{int(max_lon)}_crossdateline"
                          f"_Lat{int(min_lat)}_{int(max_lat)}")
        else:
            extent_str = (f"Lon{int(min_lon)}_{int(max_lon)}"
                          f"_Lat{int(min_lat)}_{int(max_lat)}")
        OUTPUT_ROW = os.path.join(
            OUTPUT_DIR,
            f"FY4B_Equirect_Row_Nearest_{extent_str}_{resolution_str}deg.tif")
        OUTPUT_COL = os.path.join(
            OUTPUT_DIR,
            f"FY4B_Equirect_Col_Nearest_{extent_str}_{resolution_str}deg.tif")
        stats_file = os.path.join(
            OUTPUT_DIR,
            f"FY4B_Equirect_Stats_Nearest_{extent_str}_{resolution_str}deg.txt")
    else:
        OUTPUT_ROW = os.path.join(
            OUTPUT_DIR,
            f"FY4B_Equirect_Row_Nearest_{resolution_str}deg.tif")
        OUTPUT_COL = os.path.join(
            OUTPUT_DIR,
            f"FY4B_Equirect_Col_Nearest_{resolution_str}deg.tif")
        stats_file = os.path.join(
            OUTPUT_DIR,
            f"FY4B_Equirect_Stats_Nearest_{resolution_str}deg.txt")

    print("=" * 70)
    print("标称坐标系经纬度查找表 → 等经纬投影查找表")
    print("（cKDTree 最邻近插值 + 分块 + 进度显示）")
    print("=" * 70)
    print(f"输入纬度文件: {LAT_TIF}")
    print(f"输入经度文件: {LON_TIF}")
    print(f"输出行号文件: {OUTPUT_ROW}")
    print(f"输出列号文件: {OUTPUT_COL}")
    print(f"目标分辨率: {RESOLUTION_DEGREE:.5f} 度")
    print(f"插值方法: 最邻近（Nearest）")
    if TARGET_EXTENT is not None:
        if crosses_dateline:
            print(f"裁切范围: 经度[{min_lon:.2f}, {max_lon:.2f}] (跨越国际日期变更线)")
        else:
            print(f"裁切范围: 经度[{min_lon:.2f}, {max_lon:.2f}], "
                  f"纬度[{min_lat:.2f}, {max_lat:.2f}]")
    else:
        print("裁切范围: 自动计算（全球范围）")
    print("=" * 70)

    # ---------- 1. 读取 ----------
    print("\n[1/4] 读取原始经纬度查找表...")
    if not os.path.exists(LAT_TIF):
        raise FileNotFoundError(f"找不到文件: {LAT_TIF}")
    if not os.path.exists(LON_TIF):
        raise FileNotFoundError(f"找不到文件: {LON_TIF}")

    lat_data = read_geotiff(LAT_TIF)
    lon_data = read_geotiff(LON_TIF)
    rows, cols = lat_data.shape
    print(f"  原始尺寸: {rows} x {cols}")
    print(f"  纬度范围: [{np.nanmin(lat_data):.4f}, {np.nanmax(lat_data):.4f}]")
    print(f"  经度范围: [{np.nanmin(lon_data):.4f}, {np.nanmax(lon_data):.4f}]")

    # ---------- 2. 有效掩码 ----------
    print("\n[2/4] 构建有效掩码与目标网格...")

    base_valid = np.isfinite(lat_data) & np.isfinite(lon_data)

    if TARGET_EXTENT is not None:
        lon_u = np.where(lon_data < 0, lon_data + 360.0, lon_data)
        if crosses_dateline:
            lon_ok = (lon_u >= min_lon) & (lon_u <= (max_lon + 360.0))
        else:
            lon_ok = (lon_u >= min_lon) & (lon_u <= max_lon)
        lat_ok = (lat_data >= min_lat) & (lat_data <= max_lat)
        valid_mask = base_valid & lon_ok & lat_ok
    else:
        valid_mask = base_valid
        lon_u_valid = np.where(lon_data[base_valid] < 0,
                               lon_data[base_valid] + 360.0,
                               lon_data[base_valid])
        min_lon = float(np.min(lon_u_valid))
        max_lon = float(np.max(lon_u_valid))
        min_lat = float(np.min(lat_data[base_valid]))
        max_lat = float(np.max(lat_data[base_valid]))

    n_valid = int(np.sum(valid_mask))
    print(f"  有效像元: {n_valid:,} / {rows*cols:,} "
          f"({n_valid/(rows*cols)*100:.2f}%)")
    if n_valid < 100:
        raise RuntimeError("有效像元太少，请检查裁切范围或数据")

    # ---------- 目标网格 ----------
    if TARGET_EXTENT is not None and crosses_dateline:
        tgt_lon_min = min_lon
        tgt_lon_max = max_lon + 360.0
    else:
        tgt_lon_min = min_lon
        tgt_lon_max = max_lon

    ncols_t = int(round((tgt_lon_max - tgt_lon_min) / RESOLUTION_DEGREE)) + 1
    nrows_t = int(round((max_lat - min_lat) / RESOLUTION_DEGREE)) + 1

    print(f"  目标网格: {nrows_t} x {ncols_t} = {nrows_t*ncols_t:,} 像元")
    print(f"  目标经度范围(0-360): [{tgt_lon_min:.4f}, {tgt_lon_max:.4f}]")
    print(f"  目标纬度范围: [{min_lat:.4f}, {max_lat:.4f}]")

    lon_1d = np.linspace(tgt_lon_min, tgt_lon_max, ncols_t, dtype=np.float64)
    lat_1d = np.linspace(max_lat, min_lat, nrows_t, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

    # ---------- 3. 最邻近插值 ----------
    print("\n[3/4] 执行最邻近插值...")
    row_out, col_out = build_lookup_nearest(
        lat_data, lon_data, valid_mask,
        lon_grid, lat_grid
    )

    # 释放
    del lat_data, lon_data, lon_grid, lat_grid, lon_1d, lat_1d
    import gc; gc.collect()

    # 统计
    valid_out = (row_out != NODATA) & (col_out != NODATA)
    n_out = int(np.sum(valid_out))
    print(f"\n  成功映射: {n_out:,} / {nrows_t*ncols_t:,} "
          f"({n_out/(nrows_t*ncols_t)*100:.2f}%)")
    if n_out > 0:
        print(f"  行号范围: [{row_out[valid_out].min()}, "
              f"{row_out[valid_out].max()}]")
        print(f"  列号范围: [{col_out[valid_out].min()}, "
              f"{col_out[valid_out].max()}]")

    # ---------- 4. 保存 ----------
    print("\n[4/4] 保存输出...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    geotransform = [
        float(tgt_lon_min),
        float(RESOLUTION_DEGREE),
        0.0,
        float(max_lat),
        0.0,
        float(-RESOLUTION_DEGREE)
    ]

    print(f"  保存行号: {OUTPUT_ROW}")
    write_geotiff(OUTPUT_ROW, row_out, geotransform, epsg=4326,
                  dtype=np.int32, nodata=NODATA)
    print(f"  保存列号: {OUTPUT_COL}")
    write_geotiff(OUTPUT_COL, col_out, geotransform, epsg=4326,
                  dtype=np.int32, nodata=NODATA)

    # 统计文件
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("等经纬投影查找表生成统计（最邻近插值）\n")
        f.write("=" * 70 + "\n")
        f.write(f"输入纬度: {LAT_TIF}\n")
        f.write(f"输入经度: {LON_TIF}\n")
        f.write(f"输出行号: {OUTPUT_ROW}\n")
        f.write(f"输出列号: {OUTPUT_COL}\n")
        f.write(f"分辨率: {RESOLUTION_DEGREE:.5f} 度\n")
        f.write(f"插值方法: 最邻近（cKDTree）\n")
        f.write(f"目标网格: {nrows_t} x {ncols_t}\n")
        f.write(f"有效映射: {n_out:,} ({n_out/(nrows_t*ncols_t)*100:.2f}%)\n")
        if TARGET_EXTENT is not None:
            if crosses_dateline:
                f.write(f"裁切范围: 经度[{min_lon:.2f}, {max_lon:.2f}] "
                        f"(跨日期变更线)\n")
            else:
                f.write(f"裁切范围: 经度[{min_lon:.2f}, {max_lon:.2f}], "
                        f"纬度[{min_lat:.2f}, {max_lat:.2f}]\n")
        if n_out > 0:
            f.write(f"行号范围: [{row_out[valid_out].min()}, "
                    f"{row_out[valid_out].max()}]\n")
            f.write(f"列号范围: [{col_out[valid_out].min()}, "
                    f"{col_out[valid_out].max()}]\n")
        f.write("=" * 70 + "\n")
    print(f"  统计: {stats_file}")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print("说明:")
    print("  - 输出像元值 = 原始标称查找表的 (row, col)，整数")
    print("  - 无效值 = -1")
    print("  - 使用 cKDTree 最邻近查询")
    print("=" * 70)

    return row_out, col_out


if __name__ == "__main__":
    try:
        create_equirectangular_lookup()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()