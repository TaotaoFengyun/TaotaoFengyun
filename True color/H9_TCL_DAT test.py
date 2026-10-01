import numpy as np
from satpy import Scene
from pathlib import Path
import os
import re
import cv2
import bz2
import shutil
from PIL import Image
import warnings
from collections import defaultdict
from pyorbital.astronomy import cos_zen

warnings.filterwarnings('ignore')

# ================= 配置参数 =================
CLEAN_DATA_DIR = r"D:\Satellites\Himawari\Himawari 8 9\H9_AHI_20260829"
OUTPUT_DIR = r"D:\TaotaoFengyun\Himawari9\202618 Saudel"
NIGHTLIGHT_PATH = r"D:\Satellites\Himawari\Himawari 8 9\TIFF\H9_VIIRS 2025_1.png"
MASK_PATH = r"D:\Satellites\Himawari\Himawari 8 9\TIFF\Black.png"
TEMP_DIR = r"D:\Satellites\Himawari\Himawari 8 9\Himawari_Decompressed1"  # 解压临时目录

OUTPUT_SCALE = 1
CALIB_COEFFS = {'B01': 0.00176, 'B02': 0.00215, 'B03': 0.00138, 'B04': 0.00100}

# 图像增强参数
GAIN = 2.3
SATURATION = 2.0
WHITE_POINT = 2.3
EXPOSURE = 2.5
APPLY_LINEAR_CORRECTION = True
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 0.50
CLAHE_TILE_GRID_SIZE = 4
IR_GAMMA = 1.0
NIGHTLIGHT_ALPHA = 1.0  # 夜光透明度（0-1），值越大夜光越明显

# 红外辐射亮度映射参数
IR_RAD_MIN = 0.1
IR_RAD_MAX = 2.0
IR_CONTRAST_STRETCH = True
IR_STRETCH_PERCENT = 0.5

# 大气校正参数（SZA 自适应）
# ATMOSPHERE = 0.0045 - sza * 0.00002
# 有效范围: sza ∈ [0, 180]，大气校正系数 ∈ [0.0045, 0.0009]
ATM_COEFF_A = 0.0040  # 截距 (SZA=0° 时的值)
ATM_COEFF_B = 0.00003  # 斜率
ATM_COEFF_MIN = 0.0005  # 最小大气校正系数
ATM_COEFF_MAX = 0.0040  # 最大大气校正系数


# ================= 工具函数 =================

def decompress_bz2_file(bz2_path, output_dir=None):
    """
    解压bz2文件，返回解压后的文件路径

    参数:
        bz2_path: bz2压缩文件路径
        output_dir: 输出目录，如果为None则使用临时目录

    返回:
        解压后的文件路径
    """
    if output_dir is None:
        output_dir = TEMP_DIR

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否为bz2文件
    bz2_path = Path(bz2_path)
    if not str(bz2_path).lower().endswith('.bz2'):
        # 如果不是bz2文件，直接返回原路径
        return str(bz2_path)

    # 生成输出文件名（去掉.bz2后缀）
    output_filename = bz2_path.stem  # 去掉.bz2后缀
    output_path = os.path.join(output_dir, output_filename)

    # 检查是否已经解压过（如果文件存在且大小匹配）
    if os.path.exists(output_path):
        # 比较文件大小（简单检查，不完整但快速）
        bz2_size = os.path.getsize(bz2_path)
        output_size = os.path.getsize(output_path)
        # 如果解压后的文件大小合理（通常比压缩文件大），认为已解压
        if output_size > bz2_size * 0.5:
            print(f"   📁 文件已解压: {output_path}")
            return output_path

    print(f"   📦 正在解压: {bz2_path.name} -> {output_filename}")

    try:
        # 解压bz2文件
        with bz2.open(bz2_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"   ✅ 解压完成: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
        return output_path

    except Exception as e:
        print(f"   ❌ 解压失败: {e}")
        return str(bz2_path)


def decompress_files(files):
    """
    解压文件列表中的所有bz2文件

    参数:
        files: 文件路径列表

    返回:
        解压后的文件路径列表
    """
    decompressed_files = []

    for f in files:
        f_path = Path(f)
        if f_path.suffix.lower() == '.bz2':
            # 是bz2文件，需要解压
            decompressed_path = decompress_bz2_file(f_path)
            decompressed_files.append(decompressed_path)
        else:
            # 检查文件是否真实存在
            if os.path.exists(f_path):
                decompressed_files.append(str(f_path))
            else:
                # 如果文件不存在，检查是否有对应的bz2文件
                bz2_path = Path(str(f_path) + '.bz2')
                if os.path.exists(bz2_path):
                    decompressed_path = decompress_bz2_file(bz2_path)
                    decompressed_files.append(decompressed_path)
                else:
                    print(f"   ⚠️ 文件不存在: {f_path}")
                    decompressed_files.append(str(f_path))

    return decompressed_files


def find_hsd_files(base_path):
    """
    递归查找所有HSD文件（支持.dat和.dat.bz2）

    参数:
        base_path: 基础路径（文件或目录）

    返回:
        文件路径列表
    """
    files = []
    base_path = Path(base_path)

    if base_path.is_file():
        # 如果是文件，直接添加
        files.append(base_path)
    elif base_path.is_dir():
        # 如果是目录，查找所有HSD文件
        for pattern in ['HS_H09_*.DAT', 'HS_H09_*.DAT.bz2', 'HS_H09_*.dat', 'HS_H09_*.dat.bz2']:
            for f in base_path.glob(pattern):
                files.append(f)
        # 递归查找子目录
        for sub_dir in base_path.iterdir():
            if sub_dir.is_dir():
                files.extend(find_hsd_files(sub_dir))
    else:
        # 如果路径不存在，尝试当作文件查找
        if os.path.exists(str(base_path) + '.bz2'):
            files.append(Path(str(base_path) + '.bz2'))

    return sorted(files)


def clean_temp_files():
    """
    清理临时解压文件（可选）
    """
    if os.path.exists(TEMP_DIR):
        print(f"\n🧹 清理临时文件: {TEMP_DIR}")
        try:
            # 只删除.dat文件
            for f in os.listdir(TEMP_DIR):
                if f.endswith('.dat') or f.endswith('.DAT'):
                    file_path = os.path.join(TEMP_DIR, f)
                    os.remove(file_path)
                    print(f"   已删除: {f}")
            print(f"   ✅ 临时文件清理完成")
        except Exception as e:
            print(f"   ⚠️ 清理临时文件失败: {e}")


# ================= 核心算法 =================

def downsample_array(arr, scale):
    if scale <= 1: return arr
    h, w = arr.shape[-2], arr.shape[-1]
    h_new = (h // scale) * scale
    w_new = (w // scale) * scale
    if arr.ndim == 3:
        cropped = arr[:, :h_new, :w_new]
        return cropped.reshape(3, h_new // scale, scale, w_new // scale, scale).mean(axis=(2, 4))
    else:
        cropped = arr[:h_new, :w_new]
        return cropped.reshape(h_new // scale, scale, w_new // scale, scale).mean(axis=(1, 3))


def apply_band_correction(r, g, b):
    print("✅ 应用波段线性增益校正...")
    r = np.clip(r * 0.98 - 0.0000, 0, None)
    g = np.clip(g * 0.95 - 0.0004, 0, None)
    b = np.clip(b * 0.82 - 0.0003, 0, None)
    return r, g, b


def apply_clahe(rgb_u8):
    print(f"   应用 CLAHE (clipLimit={CLAHE_CLIP_LIMIT}, tileGrid={CLAHE_TILE_GRID_SIZE})...")
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def uncharted2_filmic(x, white_point):
    A, B, C, D, E, F = 0.15, 0.50, 0.40, 0.60, 0.020, 0.30
    W = white_point
    x = np.maximum(x, 0.0)
    num = x * (A * x + C * B) + D * E
    den = x * (A * x + B) + D * F
    curr = num / den - E / F
    white_num = W * (A * W + C * B) + D * E
    white_den = W * (A * W + B) + D * F
    white_scale = white_num / white_den - E / F
    return curr / white_scale


def process_full_disk(bands_dict, sza_data=None):
    """
    处理全圆盘真彩色图像

    参数:
        bands_dict: 包含 B01, B02, B03, B04 波段的字典
        sza_data: 太阳天顶角数据 (2D numpy array)，用于自适应大气校正
    """
    print("1. 正在执行辐射定标...")
    b01 = bands_dict['B01'].astype(np.float32) * CALIB_COEFFS['B01']
    b02 = bands_dict['B02'].astype(np.float32) * CALIB_COEFFS['B02']
    b04 = bands_dict['B04'].astype(np.float32) * CALIB_COEFFS['B04']

    print("2. 正在处理 B03 (降采样 + 定标)...")
    h, w = bands_dict['B03'].shape
    h_new = h - (h % 2)
    w_new = w - (w % 2)
    y_off, x_off = (h - h_new) // 2, (w - w_new) // 2
    b03_1km = bands_dict['B03'][y_off:y_off + h_new, x_off:x_off + w_new].reshape(h_new // 2, 2, w_new // 2, 2).mean(
        axis=(1, 3)) * CALIB_COEFFS['B03']

    print("3. 正在合成真彩色 RGB...")
    r = b03_1km
    g = 0.657069 * b02 + 0.083581 * b04
    b = b01
    rgb = np.stack([r, g, b], axis=0)
    del b01, b02, b04, b03_1km, r, g, b

    if OUTPUT_SCALE > 1:
        print(f"4. 正在降采样至 {OUTPUT_SCALE}km 分辨率...")
        rgb = downsample_array(rgb, OUTPUT_SCALE)

    print("5. 应用波段线性增益校正...")
    if APPLY_LINEAR_CORRECTION:
        r, g, b = rgb[0], rgb[1], rgb[2]
        r, g, b = apply_band_correction(r, g, b)
        rgb = np.stack([r, g, b], axis=0)
        del r, g, b

    # ========== 大气校正（SZA 自适应） ==========
    print("6. 大气校正与全局曝光...")

    # 获取 RGB 的空间维度
    _, h_rgb, w_rgb = rgb.shape

    # 如果提供了 SZA 数据，则计算每个像素的大气校正系数
    if sza_data is not None:
        # 确保 SZA 尺寸与 RGB 匹配
        if sza_data.shape != (h_rgb, w_rgb):
            print(f"   调整 SZA 尺寸: {sza_data.shape} -> {(h_rgb, w_rgb)}")
            sza_resized = np.array(
                Image.fromarray(sza_data.astype(np.float32)).resize((w_rgb, h_rgb), Image.BILINEAR)
            )
        else:
            sza_resized = sza_data

        # 根据 SZA 计算大气校正系数: ATMOSPHERE = 0.0045 - sza * 0.00002
        atm_coeff = ATM_COEFF_A - sza_resized * ATM_COEFF_B
        # 限制大气校正系数范围，防止出现负值或过大值
        atm_coeff = np.clip(atm_coeff, ATM_COEFF_MIN, ATM_COEFF_MAX)

        print(f"   SZA 范围: {sza_resized.min():.1f}° ~ {sza_resized.max():.1f}°")
        print(f"   大气校正系数范围: {atm_coeff.min():.6f} ~ {atm_coeff.max():.6f}")
        print(f"   大气校正系数均值: {atm_coeff.mean():.6f}")
        print(f"   大气校正系数标准差: {atm_coeff.std():.6f}")

        # 为每个波段计算不同的偏移量 (比例 1.0, 2.0, 3.25)
        atm_ratios = np.array([1.0, 2.0, 3.25], dtype=np.float32).reshape(3, 1, 1)
        # 每个波段的偏移量 = atm_coeff * ratio
        offsets = atm_coeff * atm_ratios  # shape: (3, h, w)

        # 分母: (1 - offset)^2
        denominator = (1.0 - offsets) ** 2

        # 大气校正: (rgb - offsets) / denominator
        rgb = (rgb - offsets) / denominator
    else:
        # 如果未提供 SZA，使用默认大气校正系数（SZA=90° 时的值）
        print("   ⚠️ 未提供 SZA 数据，使用默认大气校正系数 (SZA=90°)")
        default_atm = ATM_COEFF_A - 90.0 * ATM_COEFF_B  # 0.0045 - 0.0018 = 0.0027
        atm_ratios = np.array([1.0, 2.0, 3.25], dtype=np.float32).reshape(3, 1, 1)
        offsets = default_atm * atm_ratios
        denominator = (1.0 - offsets) ** 2
        rgb = (rgb - offsets) / denominator
    # ========== 大气校正结束 ==========

    rgb *= GAIN * EXPOSURE

    print("7. 饱和度增强...")
    lum_coeffs = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32).reshape(3, 1, 1)
    luma = np.sum(rgb * lum_coeffs, axis=0, keepdims=True)
    rgb = luma + (rgb - luma) * SATURATION

    print("8. 色调映射 (Uncharted 2)...")
    RGBLin_2_AP0 = np.array(
        [[0.4397010, 0.3829780, 0.1773350], [0.0897923, 0.8134230, 0.0967616], [0.0175440, 0.1115440, 0.8707040]],
        dtype=np.float32)
    rgb_ap0 = np.einsum('cxy, dc -> dxy', rgb, RGBLin_2_AP0)
    rgb_tonemapped = uncharted2_filmic(rgb_ap0, WHITE_POINT)

    AP0_2_RGBLin = np.array(
        [[2.52169, -1.13413, -0.38756], [-0.27648, 1.37272, -0.09624], [-0.01538, -0.15298, 1.16835]], dtype=np.float32)
    rgb_linear = np.einsum('cxy, dc -> dxy', rgb_tonemapped, AP0_2_RGBLin)

    print("9. Gamma 校正与输出...")
    mask = rgb_linear < 0.0031308
    rgb_srgb = np.where(mask, 12.92 * rgb_linear, 1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055)
    rgb_u8 = np.clip(rgb_srgb, 0, 1) * 255.0
    rgb_u8 = rgb_u8.astype(np.uint8)
    result = np.transpose(rgb_u8, (1, 2, 0))

    if USE_CLAHE:
        print("10. 应用 CLAHE 局部对比度增强...")
        result = apply_clahe(result)

    print(f"   输出图像形状: {result.shape}")
    return result


def calculate_sza(scene_obj):
    """基于 B01 的几何信息计算太阳天顶角"""
    print("计算太阳天顶角 (SZA)...")
    try:
        b01_da = scene_obj['B01']
        lons, lats = b01_da.attrs['area'].get_lonlats()
        obs_time = b01_da.attrs['start_time']
        cos_sza = cos_zen(obs_time, lons, lats)
        sza_rad = np.arccos(np.clip(cos_sza, -1, 1))
        sza = np.degrees(sza_rad).astype(np.float32)

        sza = np.nan_to_num(sza, nan=90.0)
        print(f"SZA 范围: {sza.min():.1f}° ~ {sza.max():.1f}°")
        return sza
    except Exception as e:
        print(f"⚠️ SZA 计算失败: {e}")
        print("   使用默认 SZA 值 (90° 过渡区)")
        h, w = 11000, 11000
        sza = np.full((h, w), 90.0, dtype=np.float32)
        return sza


def contrast_stretch(image, min_percent=0.0, max_percent=100.0):
    """对比度拉伸：将指定百分位范围内的数据拉伸到 [0, 1]"""
    clean_image = np.nan_to_num(image, nan=0)
    clean_image[clean_image == -9999] = 0
    valid_data = clean_image[clean_image > 0]

    if len(valid_data) == 0:
        print(f"   ⚠️ 没有有效数据，返回全零数组")
        return np.zeros_like(clean_image, dtype=np.float32)

    p_low = np.percentile(valid_data, min_percent)
    p_high = np.percentile(valid_data, max_percent)

    print(f"   对比度拉伸范围: {p_low:.4f} -> {p_high:.4f}")

    stretched = np.zeros_like(clean_image, dtype=np.float32)
    valid_mask = clean_image > 0
    stretched[valid_mask] = np.clip((clean_image[valid_mask] - p_low) / (p_high - p_low), 0, 1)

    return stretched


def create_night_ir(b13_data, nightlight_img):
    """生成夜间红外云图并叠加彩色夜光"""
    print("开始处理夜间红外模式...")
    TARGET_SIZE = 11000

    if hasattr(b13_data, 'values'):
        b13_data = b13_data.values

    # 处理 NaN 和 -9999
    nan_count = np.sum(np.isnan(b13_data))
    if nan_count > 0:
        print(f"   ⚠️ 发现 {nan_count:,} 个 NaN 值，将强制转换为 0")
        b13_data = np.nan_to_num(b13_data, nan=0.0)

    neg9999_count = np.sum(b13_data == -9999)
    if neg9999_count > 0:
        print(f"   ⚠️ 发现 {neg9999_count:,} 个 -9999 值，将强制转换为 0")
        b13_data[b13_data == -9999] = 0

    valid_data = b13_data[b13_data > 0]
    if len(valid_data) == 0:
        print("⚠️ B13 数据全为无效值 (0/NaN/-9999)，使用模拟数据")
        h, w = 5500, 5500
        x = np.linspace(-3, 3, w)
        y = np.linspace(-3, 3, h)
        X, Y = np.meshgrid(x, y)
        b13_data = 0.5 + 0.4 * np.exp(-((X) ** 2 + (Y) ** 2) * 0.5) + \
                   0.2 * np.sin(X * 2) * np.cos(Y * 2)
        b13_data += 0.05 * np.random.randn(h, w)
        b13_data = np.clip(b13_data, 0.00, 1.5)
        valid_data = b13_data[b13_data > 0]
    else:
        print(f"   B13 有效数据范围: {valid_data.min():.4f} ~ {valid_data.max():.4f}")
        print(f"   B13 有效数据均值: {valid_data.mean():.4f}")
        print(f"   B13 有效数据标准差: {valid_data.std():.4f}")

        zero_count = np.sum(b13_data == 0)
        if zero_count > 0:
            print(f"   📊 值为 0 的像元数: {zero_count:,} (占比 {zero_count / b13_data.size * 100:.2f}%)")

        p_low = np.percentile(valid_data, 0)
        p_high = np.percentile(valid_data, 100)
        print(f"   B13 0% 分位数: {p_low:.4f}, 100% 分位数: {p_high:.4f}")

        global IR_RAD_MIN, IR_RAD_MAX
        IR_RAD_MIN = max(p_low, 0.000)
        IR_RAD_MAX = min(p_high, 20.0)
        print(f"   映射范围: {IR_RAD_MIN:.4f} (黑) ~ {IR_RAD_MAX:.4f} (白)")

    # ================= 新增：归一化 LUT 处理 =================
    print("   应用归一化 LUT 进行稳定映射...")

    # 1. 使用对比度拉伸获取归一化数据
    stretched_data = contrast_stretch(b13_data, min_percent=0.0, max_percent=100)

    # 2. 统计拉伸后的数据分布
    stretched_valid = stretched_data[stretched_data > 0]
    if len(stretched_valid) > 0:
        print(f"   拉伸后有效数据范围: {stretched_valid.min():.4f} ~ {stretched_valid.max():.4f}")
        print(f"   拉伸后有效数据均值: {stretched_valid.mean():.4f}")
        print(f"   拉伸后有效数据标准差: {stretched_valid.std():.4f}")

        # 3. 计算百分位数用于 LUT 映射（使用更稳定的统计量）
        p1 = np.percentile(stretched_valid, 0.01)
        p99 = np.percentile(stretched_valid, 99.99)
        print(f"   1% 分位数: {p1:.4f}, 99% 分位数: {p99:.4f}")

        # 4. 创建归一化 LUT（使用百分位数映射，避免极值影响）
        # 将 [p1, p99] 映射到 [0, 1]，使用分位数截断防止闪烁
        if p99 > p1:
            stretched_data_clipped = np.clip((stretched_data - p1) / (p99 - p1), 0, 1)
        else:
            stretched_data_clipped = stretched_data

        # 5. 应用 Gamma 校正（可选，用于增强暗部细节）
        gamma_corrected = np.power(stretched_data_clipped, 0.8)  # 使用固定 gamma 值
        print(f"   Gamma 校正后范围: {gamma_corrected.min():.4f} ~ {gamma_corrected.max():.4f}")
        print(f"   Gamma 校正后均值: {gamma_corrected.mean():.4f}")

        # 6. 生成最终的归一化数据
        normalized_data = gamma_corrected

        # 7. 统计最终归一化数据分布
        norm_valid = normalized_data[normalized_data > 0]
        if len(norm_valid) > 0:
            print(f"   归一化后有效数据范围: {norm_valid.min():.4f} ~ {norm_valid.max():.4f}")
            print(f"   归一化后有效数据均值: {norm_valid.mean():.4f}")
            print(f"   归一化后有效数据标准差: {norm_valid.std():.4f}")

        # 8. 直接用归一化数据替代原来的 stretched_data
        stretched_data = normalized_data

        print(f"   ✅ 归一化 LUT 应用完成")
    else:
        print("   ⚠️ 拉伸后无有效数据，跳过归一化 LUT")
        stretched_data = np.clip(stretched_data, 0, 1)
    # ================= 归一化 LUT 处理结束 =================

    print(f"   拉伸后范围: {stretched_data.min():.4f} ~ {stretched_data.max():.4f}")
    print(f"   拉伸后均值: {stretched_data.mean():.4f}, 标准差: {stretched_data.std():.4f}")

    if stretched_data.shape[0] != TARGET_SIZE:
        print(
            f"   重采样 B13 数据从 {stretched_data.shape[0]}x{stretched_data.shape[1]} 到 {TARGET_SIZE}x{TARGET_SIZE}...")
        temp_u8 = (stretched_data * 255).astype(np.uint8)
        resized = np.array(
            Image.fromarray(temp_u8).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        ).astype(np.float32) / 255.0
    else:
        resized = stretched_data

    ir_gray = 1.0 - resized
    print(f"   红外灰度范围: {ir_gray.min():.3f} ~ {ir_gray.max():.3f}")
    print(f"   灰度均值: {ir_gray.mean():.3f}, 标准差: {ir_gray.std():.3f}")

    print("   应用 Gamma 校正增强暗部细节...")
    ir_gray = np.power(ir_gray, 1.0 / IR_GAMMA)
    print(f"   Gamma 校正后灰度范围: {ir_gray.min():.3f} ~ {ir_gray.max():.3f}")

    ir_3d = np.stack([ir_gray] * 3, axis=-1).astype(np.float32)

    if nightlight_img is not None:
        # ... 后面的夜光叠加代码保持不变 ...
        print("   加载彩色夜光数据...")
        if len(nightlight_img.shape) == 2:
            # 2维图像（灰度或索引颜色），需要转换为RGB
            print(f"   夜光图像为2维，转换为RGB...")

            # 调整尺寸
            if nightlight_img.shape[0] != TARGET_SIZE or nightlight_img.shape[1] != TARGET_SIZE:
                nightlight_resized = np.array(
                    Image.fromarray(nightlight_img).resize(
                        (TARGET_SIZE, TARGET_SIZE), Image.BILINEAR))
            else:
                nightlight_resized = nightlight_img

            # 归一化
            if nightlight_resized.dtype == np.uint8:
                nl_gray = nightlight_resized.astype(np.float32) / 255.0
            else:
                nl_gray = nightlight_resized.astype(np.float32)
                if nl_gray.max() > 1.0:
                    nl_gray = nl_gray / 255.0

            # 转换为3通道（灰度重复三次）
            nl_3d = np.stack([nl_gray] * 3, axis=-1)
            print(f"   夜光灰度范围: {nl_gray.min():.3f} ~ {nl_gray.max():.3f}")

        elif len(nightlight_img.shape) == 3:
            # 3维图像（已经是RGB）
            print(f"   夜光图像为3维RGB，直接使用...")

            # 调整尺寸
            if nightlight_img.shape[0] != TARGET_SIZE or nightlight_img.shape[1] != TARGET_SIZE:
                nightlight_resized = np.array(
                    Image.fromarray(nightlight_img).resize(
                        (TARGET_SIZE, TARGET_SIZE), Image.BILINEAR))
            else:
                nightlight_resized = nightlight_img

            # 归一化到 [0, 1]
            if nightlight_resized.dtype == np.uint8:
                nl_3d = nightlight_resized.astype(np.float32) / 255.0
            else:
                nl_3d = nightlight_resized.astype(np.float32)
                if nl_3d.max() > 1.0:
                    nl_3d = nl_3d / 255.0

            print(f"   夜光彩色范围: R[{nl_3d[:, :, 0].min():.3f}~{nl_3d[:, :, 0].max():.3f}], "
                  f"G[{nl_3d[:, :, 1].min():.3f}~{nl_3d[:, :, 1].max():.3f}], "
                  f"B[{nl_3d[:, :, 2].min():.3f}~{nl_3d[:, :, 2].max():.3f}]")
        else:
            print(f"   ⚠️ 无法识别的夜光图像维度: {nightlight_img.shape}")
            return ir_3d

        # 使用滤色模式（Screen Mode）叠加彩色夜光
        print("   使用滤色模式 (Screen Mode) 叠加彩色夜光...")

        # 滤色模式：1 - (1 - A) * (1 - B)
        # 对RGB三个通道分别处理
        final_image = np.zeros_like(ir_3d)
        for channel in range(3):
            # 应用滤色模式（夜光强度乘以0.75）
            screen_result = 1.0 - (1.0 - ir_3d[:, :, channel]) * (1.0 - nl_3d[:, :, channel] * 0.75)
            # 混合红外和滤色结果（通过透明度控制夜光强度）
            final_image[:, :, channel] = ir_3d[:, :, channel] * (
                    1 - NIGHTLIGHT_ALPHA) + screen_result * NIGHTLIGHT_ALPHA

        print(f"   叠加后范围: {final_image.min():.3f} ~ {final_image.max():.3f}")

    else:
        print("   未找到夜光数据，使用纯红外模式")
        final_image = ir_3d

    final_image = np.clip(final_image, 0.0, 1.0)

    # 转换为8位
    final_image = (final_image * 255).astype(np.uint8)

    print("✅ 夜间红外模式处理完成")
    return final_image


def blend_day_night(day_rgb, night_ir, sza_data):
    """
    根据 SZA 混合昼夜图像。
    只对真彩色部分创建掩膜，夜间红外作为底层始终显示。

    公式：
    - sza < 75: 真彩色 100% 显示
    - 75 <= sza <= 105: 真彩色透明度从 100% 线性递减到 0
    - sza > 105: 真彩色 0% 显示（完全显示夜间红外）
    """
    print("执行昼夜平滑过渡...")
    h, w = day_rgb.shape[:2]

    # 确保 SZA 尺寸匹配
    if sza_data.shape != (h, w):
        print(f"   调整 SZA 尺寸: {sza_data.shape} -> {(h, w)}")
        sza_resized = np.array(
            Image.fromarray(sza_data.astype(np.float32)).resize((w, h), Image.BILINEAR)
        )
    else:
        sza_resized = sza_data

    # 确保 night_ir 尺寸与 day_rgb 一致
    if night_ir.shape[:2] != (h, w):
        print(f"   调整夜光图尺寸: {night_ir.shape[:2]} -> {(h, w)}")
        night_ir = np.array(Image.fromarray(night_ir).resize((w, h)))

    # --- 只创建真彩色图的掩膜（Alpha） ---
    day_alpha = np.zeros_like(sza_resized, dtype=np.float32)

    # 白天区域 (SZA < 75°)
    day_mask = sza_resized < 65
    day_alpha[day_mask] = 1.0

    # 过渡区域 (75° ≤ SZA ≤ 105°)
    transition_mask = (sza_resized >= 65) & (sza_resized <= 95)
    day_alpha[transition_mask] = 1.0 - (1.0 / 30.0) * (sza_resized[transition_mask] - 65.0)
    day_alpha[transition_mask] = np.clip(day_alpha[transition_mask], 0, 1)

    # 夜间区域 (SZA > 105°) 保持为 0

    # 扩展为 3 通道
    day_alpha_3d = np.stack([day_alpha] * 3, axis=-1)

    # 打印统计信息
    print(f"   真彩色 Alpha 范围: {day_alpha.min():.3f} ~ {day_alpha.max():.3f}")
    print(f"   真彩色 Alpha 均值: {day_alpha.mean():.3f}")
    print(f"   白天区域占比 (Alpha=1): {np.sum(day_alpha == 1) / day_alpha.size * 100:.2f}%")
    print(f"   过渡区域占比 (0<Alpha<1): {np.sum((day_alpha > 0) & (day_alpha < 1)) / day_alpha.size * 100:.2f}%")
    print(f"   夜间区域占比 (Alpha=0): {np.sum(day_alpha == 0) / day_alpha.size * 100:.2f}%")

    # --- 最终融合 ---
    # 夜间红外作为底层，真彩色作为上层
    day_rgb_float = day_rgb.astype(np.float32)
    night_ir_float = night_ir.astype(np.float32)

    # 混合：夜间红外作为底层，真彩色根据 Alpha 叠加
    final = night_ir_float * (1.0 - day_alpha_3d) + day_rgb_float * day_alpha_3d

    final = np.clip(final, 0, 255).astype(np.uint8)

    print(f"   混合后图像范围: {final.min()} ~ {final.max()}")
    return final


def apply_mask_direct(image, mask_path):
    """
    直接叠加黑色PNG掩膜（保留透明通道）

    注意：此方法要求掩膜尺寸与图像尺寸完全一致
    掩膜中的黑色区域(RGB=0,0,0)会覆盖图像
    掩膜中的透明区域(Alpha=0)会透传图像
    掩膜中的彩色区域会覆盖图像

    参数:
        image: 输入图像 (H, W, 3) uint8
        mask_path: 掩膜图像路径

    返回:
        应用掩膜后的图像
    """
    print("\n应用黑色掩膜（直接叠加）...")

    if not os.path.exists(mask_path):
        print(f"⚠️ 掩膜文件不存在: {mask_path}")
        return image

    try:
        # 加载PNG掩膜（保留透明通道）
        mask = Image.open(mask_path)

        # 如果掩膜有透明通道，获取它
        if mask.mode == 'RGBA':
            mask_rgba = np.array(mask)
            mask_rgb = mask_rgba[:, :, :3]  # RGB通道
            mask_alpha = mask_rgba[:, :, 3]  # Alpha通道
            has_alpha = True
        else:
            mask_rgb = np.array(mask.convert('RGB'))
            # 如果没有Alpha通道，使用RGB值来判断
            has_alpha = False
            mask_alpha = None

        print(f"   掩膜模式: {mask.mode}")
        print(f"   掩膜尺寸: {mask_rgb.shape}")
        print(f"   图像尺寸: {image.shape}")

        # 检查尺寸是否一致
        if mask_rgb.shape[:2] != image.shape[:2]:
            print(f"   ⚠️ 掩膜尺寸 ({mask_rgb.shape[:2]}) 与图像尺寸 ({image.shape[:2]}) 不一致!")
            print("   无法直接叠加，需要调整尺寸")
            print("   请确保掩膜尺寸与图像尺寸一致")
            return image

        # 创建结果图像（复制原图）
        result = image.copy()

        # 根据掩膜信息进行叠加
        if has_alpha:
            # 有透明通道：透明区域(Alpha=0)保留原图，其他区域使用掩膜覆盖
            mask_binary = (mask_alpha > 0)  # 非透明区域
            print(f"   非透明像元数: {mask_binary.sum():,} ({mask_binary.sum() / mask_binary.size * 100:.2f}%)")

            # 对于非透明区域，使用掩膜的RGB值
            # 判断哪些像素是黑色
            black_mask = (mask_rgb[:, :, 0] == 0) & (mask_rgb[:, :, 1] == 0) & (mask_rgb[:, :, 2] == 0)
            black_count = black_mask.sum()
            print(f"   黑色像元数: {black_count:,} ({black_count / mask_binary.size * 100:.2f}%)")

            # 将非透明的黑色区域设置为黑色
            result[mask_binary & black_mask] = [0, 0, 0]

            # 对于非透明的非黑色区域，使用掩膜颜色
            non_black_mask = mask_binary & ~black_mask
            if non_black_mask.any():
                result[non_black_mask] = mask_rgb[non_black_mask]

        else:
            # 无透明通道：使用RGB值判断
            # 黑色区域覆盖为黑色
            black_mask = (mask_rgb[:, :, 0] == 0) & (mask_rgb[:, :, 1] == 0) & (mask_rgb[:, :, 2] == 0)
            black_count = black_mask.sum()
            print(f"   黑色像元数: {black_count:,} ({black_count / mask_rgb.size * 100:.2f}%)")

            result[black_mask] = [0, 0, 0]

            # 非黑色区域可以使用掩膜颜色或保留原图
            # 这里保留原图（仅覆盖黑色区域）

        print(f"   ✅ 掩膜应用完成")
        print(f"   覆盖后图像中黑色像元数: {(result == 0).all(axis=2).sum():,}")

        return result

    except Exception as e:
        print(f"❌ 应用掩膜失败: {e}")
        import traceback
        traceback.print_exc()
        return image


def apply_mask_with_transparency(image, mask_path):
    """
    使用透明通道叠加掩膜（更精确的透明叠加）

    使用PIL的Image.alpha_composite进行真正的透明叠加
    """
    print("\n应用掩膜（透明叠加）...")

    if not os.path.exists(mask_path):
        print(f"⚠️ 掩膜文件不存在: {mask_path}")
        return image

    try:
        # 将图像转换为PIL Image
        img_pil = Image.fromarray(image, mode='RGB')

        # 加载掩膜
        mask = Image.open(mask_path)

        # 检查尺寸
        if mask.size != img_pil.size:
            print(f"   ⚠️ 掩膜尺寸 {mask.size} 与图像尺寸 {img_pil.size} 不一致!")
            print("   无法直接叠加，需要调整尺寸")
            return image

        # 如果掩膜没有透明通道，创建一个
        if mask.mode != 'RGBA':
            mask = mask.convert('RGBA')
            # 设置黑色区域为不透明，其他区域为透明
            mask_data = np.array(mask)
            # 检查哪些像素是黑色
            black_mask = (mask_data[:, :, 0] == 0) & (mask_data[:, :, 1] == 0) & (mask_data[:, :, 2] == 0)
            # 黑色区域：Alpha=255（不透明），其他区域：Alpha=0（透明）
            mask_data[:, :, 3] = np.where(black_mask, 255, 0)
            mask = Image.fromarray(mask_data, mode='RGBA')

        # 将图像转换为RGBA以支持透明叠加
        img_rgba = img_pil.convert('RGBA')

        # 使用alpha_composite进行叠加
        # 掩膜会覆盖在图像上，透明区域会透传
        composited = Image.alpha_composite(img_rgba, mask)

        # 转换回RGB
        result = np.array(composited.convert('RGB'))

        print(f"   ✅ 透明叠加完成")
        print(f"   结果图像形状: {result.shape}")

        return result

    except Exception as e:
        print(f"❌ 应用掩膜失败: {e}")
        import traceback
        traceback.print_exc()
        return image


def process_single(files):
    print(f"找到 {len(files)} 个文件, 目标输出分辨率: {OUTPUT_SCALE}km")

    try:
        # 解压所有bz2文件
        print("\n📦 检查并解压bz2文件...")
        decompressed_files = decompress_files(files)

        # 检查是否有文件需要解压
        if len(decompressed_files) != len(files):
            print(f"   已解压 {len(decompressed_files)} 个文件")
        else:
            print(f"   所有文件已准备就绪")

        # 过滤掉不存在的文件
        valid_files = [f for f in decompressed_files if os.path.exists(f)]
        if not valid_files:
            print("❌ 没有有效的文件可供处理")
            return

        if len(valid_files) != len(decompressed_files):
            print(f"   ⚠️ 过滤掉 {len(decompressed_files) - len(valid_files)} 个不存在的文件")

        # 使用解压后的文件
        scn = Scene(reader='ahi_hsd', filenames=valid_files)

        bands_to_load = ['B01', 'B02', 'B03', 'B04', 'B13']
        scn.load(bands_to_load)

        for band in bands_to_load:
            if band not in scn:
                print(f"⚠️ 波段 {band} 加载失败")

        # 1. 计算 SZA（先计算，用于大气校正和昼夜混合）
        sza_data = calculate_sza(scn)

        # 2. 白天真彩色（传递 SZA 数据用于自适应大气校正）
        bands_data = {}
        for k in ['B01', 'B02', 'B03', 'B04']:
            if k in scn:
                bands_data[k] = scn[k].values
            else:
                print(f"⚠️ 波段 {k} 缺失，使用随机数据")
                bands_data[k] = np.random.rand(5500, 5500) * 0.1

        day_rgb = process_full_disk(bands_data, sza_data)

        # 3. 夜间红外
        nightlight_img = None
        if os.path.exists(NIGHTLIGHT_PATH):
            try:
                img = Image.open(NIGHTLIGHT_PATH)
                print(f"夜光图像模式: {img.mode}, 尺寸: {img.size}")
                # 如果是索引颜色模式(P)，转换为RGB
                if img.mode == 'P':
                    img = img.convert('RGB')
                    print("   已将索引颜色模式转换为RGB")
                nightlight_img = np.array(img)
                print(f"彩色夜光图像已加载: {nightlight_img.shape}")
            except Exception as e:
                print(f"⚠️ 加载夜光图像失败: {e}")

        if 'B13' in scn:
            b13_data = scn['B13']
        else:
            print("⚠️ B13 数据缺失，创建模拟数据")
            h, w = 5500, 5500
            x = np.linspace(-3, 3, w)
            y = np.linspace(-3, 3, h)
            X, Y = np.meshgrid(x, y)
            b13_data = 0.5 + 0.4 * np.exp(-((X) ** 2 + (Y) ** 2) * 0.5) + \
                       0.2 * np.sin(X * 2) * np.cos(Y * 2)
            b13_data += 0.05 * np.random.randn(h, w)
            b13_data = np.clip(b13_data, 0.05, 1.5)

        night_ir = create_night_ir(b13_data, nightlight_img)

        # 4. 昼夜混合（使用已计算的 SZA）
        final_rgb = blend_day_night(day_rgb, night_ir, sza_data)

        # 5. 应用黑色掩膜（在最后一步叠加）
        if os.path.exists(MASK_PATH):
            # 方法1：直接叠加（保留RGB和透明通道）
            final_rgb = apply_mask_direct(final_rgb, MASK_PATH)

            # 方法2：使用透明叠加（更精确）
            # final_rgb = apply_mask_with_transparency(final_rgb, MASK_PATH)
        else:
            print(f"⚠️ 掩膜文件不存在，跳过掩膜应用: {MASK_PATH}")

        # 6. 保存
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = scn.start_time.strftime('%Y%m%d_%H%M') if hasattr(scn, 'start_time') else '20260725_0000'
        output_path = os.path.join(OUTPUT_DIR, f"H9_Hybrid_{timestamp}_{OUTPUT_SCALE}km.jpg")
        print(f"\n正在保存至 {output_path} ...")
        Image.fromarray(final_rgb, mode='RGB').save(output_path, quality=100)
        print(f"✅ 完成: {output_path}")

        # # 保存PNG格式（无损）
        # png_path = output_path.replace('.jpg', '.png')
        # Image.fromarray(final_rgb, mode='RGB').save(png_path)
        # print(f"✅ PNG已保存: {png_path}")

        # 可选：清理临时文件
        clean_temp_files()

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


def group_files_by_time(data_dir):
    """根据时间戳对文件进行分组"""
    # 查找所有HSD文件
    all_files = find_hsd_files(data_dir)

    pattern = re.compile(r'HS_H09_(\d{8}_\d{4})_B\d+_')
    groups = defaultdict(list)

    for f in all_files:
        match = pattern.search(str(f))
        if match:
            groups[match.group(1)].append(f)
        else:
            # 如果没有匹配到时间戳，使用文件所在目录名
            groups[str(f.parent)].append(f)

    return dict(groups)


def main():
    data_path = Path(CLEAN_DATA_DIR)

    # 如果路径是文件或目录，直接查找所有HSD文件
    if data_path.is_file() or data_path.is_dir():
        all_files = find_hsd_files(data_path)
        if all_files:
            # 按时间分组
            time_groups = group_files_by_time(data_path)
            if not time_groups:
                # 如果没有分组，将所有文件作为一个组
                print(f"找到 {len(all_files)} 个HSD文件")
                process_single(all_files)
                return
        else:
            print(f"未找到任何 HSD 文件: {data_path}")
            return
    else:
        print(f"路径不存在: {data_path}")
        return

    print(f"共发现 {len(time_groups)} 个时刻")
    for idx, (time_key, files) in enumerate(sorted(time_groups.items()), 1):
        print(f"\n[{idx}/{len(time_groups)}] 正在处理时刻: {time_key}")
        process_single(files)
    print("\n✅ 全部处理完成!")


if __name__ == "__main__":
    main()