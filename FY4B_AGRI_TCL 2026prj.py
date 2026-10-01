"""
FY-4B AGRI卫星数据处理工具
版本: 3.0 (GLT重投影版本)
功能: 使用GLT进行快速重投影、多通道合成真彩色图像、昼夜融合、色调映射
"""

import math
import numpy as np
import h5py
from scipy.ndimage import zoom
from PIL import Image
import os
import warnings
import cv2
import glob
import re
import gc
from satpy import Scene
import logging
from datetime import datetime

# ==================== 尝试导入tifffile ====================
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("警告: 未安装tifffile，将使用OpenCV读取")
    print("建议安装: pip install tifffile")

# ==================== 配置参数（硬编码） ====================
INPUT_FOLDER = r"D:\FY_Download\202626 Surigae\20260923_20260926_20260927"
OUTPUT_FOLDER = r"D:\TaotaoFengyun\FY4_AGRI\202626 Surigae"

# GLT查找表路径（行列号查找表）
GLT_ROW_PATH = r"D:\FY_Download\FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_1000M_V0001\Output1\FY4B_Equirectangular_Row_Lon23_221_Lat-90_90_0_0110deg.tif"
GLT_COL_PATH = r"D:\FY_Download\FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_1000M_V0001\Output1\FY4B_Equirectangular_Col_Lon23_221_Lat-90_90_0_0110deg.tif"
    
# 夜间灯光数据路径
NIGHT_LIGHT_JPG_PATH = r"D:\FY_Download\FY4B-_DISK_1050E_GEO_NOM_LUT_20240227000000_1000M_V0001\FY4B_VIIRS-2025_1050E_1km.png"

# 输出JPG质量
JPG_QUALITY = 95

# ==================== 图像处理参数 ====================
ATMOSPHERE_STRENGTH = 0.095
TONEMAP_WHITE_POINT = 2.0
TONEMAP_GAIN = 1.95
SATURATION_FACTOR = 2.3

USE_CLAHE = True
CLAHE_CLIP_LIMIT = 0.90
CLAHE_TILE_GRID_SIZE = 4

FIXED_NORMALIZER = 1.0
LUT_GAMMA = 0.8
DITHER_NOISE_STD = 0.75

BAND_CORRECTION_R = [0.96*0.99, 0]
BAND_CORRECTION_G = [0.94*0.99, -0.4]
BAND_CORRECTION_B = [0.93*0.99, 8]

# 输出分辨率控制（度），对应GLT的分辨率
OUTPUT_RESOLUTION_DEGREE = 0.011  # 0.011度 ≈ 1.1km，可调整

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fy4b_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 工具函数 ====================

def generate_sqrt_lut(input_max=0.7, output_max=255, gamma=0.8):
    """生成平方根查找表"""
    physical_values = np.linspace(0, input_max, 256)
    stretched = np.power(np.clip(physical_values, 0, input_max), 0.5 / gamma)
    lut = np.clip(stretched / np.max(stretched) * output_max, 0, output_max).astype(np.uint8)
    return lut


def atmospheric_correction(rgb_float, strength=0.025):
    """大气校正（向量化版本）"""
    atm_offsets = strength * np.array([0.7, 1.1, 1.5])
    denominators = (1 - atm_offsets) ** 2
    denominators = np.where(denominators > 0, denominators, 1.0)
    corrected = (rgb_float - atm_offsets) / denominators
    return np.clip(corrected, 0, None)


def ACES_Matrix_Transform(rgb, to_aces=True):
    """ACES色彩空间变换"""
    if to_aces:
        mat = np.array([
            [0.4397010, 0.3829780, 0.1773350],
            [0.0897923, 0.8134230, 0.0967616],
            [0.0175440, 0.1115440, 0.8707040]
        ])
    else:
        mat = np.array([
            [2.52169, -1.13413, -0.38756],
            [-0.27648, 1.37272, -0.09624],
            [-0.01538, -0.15298, 1.16835]
        ])
    return np.einsum('ij,klj->kli', mat, rgb)


def Uncharted2_Filmic_Tonemap(x, white_point=2.5):
    """Uncharted2电影色调映射曲线"""
    A, B, C, D, E, F, W = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30, white_point
    numerator = x * (A * x + C * B) + D * E
    denominator = x * (A * x + B) + D * F
    curve = (numerator / denominator) - (E / F)
    w_numerator = W * (A * W + C * B) + D * E
    w_denominator = W * (A * W + B) + D * F
    white_scale = (w_numerator / w_denominator) - (E / F)
    return curve / white_scale


def filmic_tonemap(rgb_float, white_point=2.5, gain=1.0):
    """完整的Filmic色调映射流程"""
    rgb = rgb_float * gain
    rgb_aces = ACES_Matrix_Transform(rgb, to_aces=True)
    rgb_aces = np.maximum(rgb_aces, 0)
    rgb_tonemapped = Uncharted2_Filmic_Tonemap(rgb_aces, white_point=white_point)
    rgb_out = ACES_Matrix_Transform(rgb_tonemapped, to_aces=False)
    return np.clip(rgb_out, 0, None)


def apply_band_correction(image_uint8, corrections):
    """波段线性校正"""
    channels = cv2.split(image_uint8.astype(np.float32))
    corrected = []
    for ch, (gain, offset) in zip(channels, corrections):
        ch = np.clip(ch * gain + offset, 0, 255).astype(np.uint8)
        corrected.append(ch)
    return cv2.merge(corrected)


def enhance_saturation(rgb_uint8, factor=1.5):
    """增强饱和度"""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255)
    hsv_out = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)


def clean_data(arr):
    """清理无效数据"""
    arr = np.nan_to_num(arr, nan=0)
    arr[arr == 65535] = 0
    return arr


def validate_input_files(file_1km, file_4km, file_geo):
    """验证输入文件路径"""
    for path in [file_1km, file_4km, file_geo]:
        if not os.path.exists(path):
            logger.error(f"文件不存在: {path}")
            return False
        if os.path.getsize(path) == 0:
            logger.error(f"文件为空: {path}")
            return False
    return True


def load_glt(glt_row_path, glt_col_path):
    """
    加载GLT行列号查找表
    优先使用tifffile读取（支持大文件），如果不可用则使用OpenCV
    返回: (row_indices, col_indices, valid_mask)
    """
    logger.info(f"加载GLT查找表...")
    logger.info(f"  行号文件: {glt_row_path}")
    logger.info(f"  列号文件: {glt_col_path}")
    
    # 检查文件是否存在
    if not os.path.exists(glt_row_path):
        raise FileNotFoundError(f"行号文件不存在: {glt_row_path}")
    if not os.path.exists(glt_col_path):
        raise FileNotFoundError(f"列号文件不存在: {glt_col_path}")
    
    # 获取文件大小
    row_size = os.path.getsize(glt_row_path) / (1024**3)  # GB
    col_size = os.path.getsize(glt_col_path) / (1024**3)  # GB
    logger.info(f"  行号文件大小: {row_size:.2f} GB")
    logger.info(f"  列号文件大小: {col_size:.2f} GB")
    
    # 读取行列号查找表
    if HAS_TIFFFILE:
        logger.info("  使用tifffile读取（支持大文件）...")
        try:
            row_glt = tifffile.imread(glt_row_path)
            col_glt = tifffile.imread(glt_col_path)
            logger.info("  tifffile读取成功")
        except Exception as e:
            logger.warning(f"  tifffile读取失败: {e}")
            logger.info("  尝试使用OpenCV读取...")
            row_glt = cv2.imread(glt_row_path, cv2.IMREAD_UNCHANGED)
            col_glt = cv2.imread(glt_col_path, cv2.IMREAD_UNCHANGED)
    else:
        logger.info("  使用OpenCV读取...")
        row_glt = cv2.imread(glt_row_path, cv2.IMREAD_UNCHANGED)
        col_glt = cv2.imread(glt_col_path, cv2.IMREAD_UNCHANGED)
    
    if row_glt is None or col_glt is None:
        raise ValueError("无法读取GLT文件，请检查文件路径和文件完整性")
    
    # 确保数据类型为int32
    if row_glt.dtype != np.int32:
        row_glt = row_glt.astype(np.int32)
    if col_glt.dtype != np.int32:
        col_glt = col_glt.astype(np.int32)
    
    logger.info(f"  GLT尺寸: {row_glt.shape}")
    logger.info(f"  行号范围: [{np.min(row_glt)}, {np.max(row_glt)}]")
    logger.info(f"  列号范围: [{np.min(col_glt)}, {np.max(col_glt)}]")
    
    # 有效掩膜（行列号 >= 0）
    valid_mask = (row_glt >= 0) & (col_glt >= 0)
    valid_count = np.sum(valid_mask)
    total_count = row_glt.size
    logger.info(f"  有效像素: {valid_count:,} ({valid_count/total_count*100:.2f}%)")
    
    return row_glt, col_glt, valid_mask


def apply_glt_to_image(image_data, row_glt, col_glt, valid_mask):
    """
    使用GLT将图像数据重投影到等经纬网格
    image_data: 原始图像数据 (2D或3D)
    row_glt: 行号查找表
    col_glt: 列号查找表
    valid_mask: 有效像素掩膜
    """
    glt_rows, glt_cols = row_glt.shape
    
    # 判断输入是单波段还是多波段
    if image_data.ndim == 2:
        # 单波段
        output = np.zeros((glt_rows, glt_cols), dtype=image_data.dtype)
        src_rows, src_cols = image_data.shape
        
        # 提取有效位置的行列号
        valid_rows = row_glt[valid_mask]
        valid_cols = col_glt[valid_mask]
        
        # 检查行列号是否在有效范围内
        valid_idx = (valid_rows >= 0) & (valid_rows < src_rows) & (valid_cols >= 0) & (valid_cols < src_cols)
        valid_rows = valid_rows[valid_idx]
        valid_cols = valid_cols[valid_idx]
        
        # 获取对应的输出位置
        out_positions = np.where(valid_mask)
        out_rows = out_positions[0][valid_idx]
        out_cols = out_positions[1][valid_idx]
        
        # 使用高级索引进行映射
        output[out_rows, out_cols] = image_data[valid_rows, valid_cols]
        
        return output
        
    elif image_data.ndim == 3:
        # 多波段（RGB）
        output = np.zeros((glt_rows, glt_cols, image_data.shape[2]), dtype=image_data.dtype)
        src_rows, src_cols, bands = image_data.shape
        
        # 提取有效位置的行列号
        valid_rows = row_glt[valid_mask]
        valid_cols = col_glt[valid_mask]
        
        # 检查行列号是否在有效范围内
        valid_idx = (valid_rows >= 0) & (valid_rows < src_rows) & (valid_cols >= 0) & (valid_cols < src_cols)
        valid_rows = valid_rows[valid_idx]
        valid_cols = valid_cols[valid_idx]
        
        # 获取对应的输出位置
        out_positions = np.where(valid_mask)
        out_rows = out_positions[0][valid_idx]
        out_cols = out_positions[1][valid_idx]
        
        # 使用高级索引进行映射（多波段）
        for band_idx in range(bands):
            output[out_rows, out_cols, band_idx] = image_data[valid_rows, valid_cols, band_idx]
        
        return output
    else:
        raise ValueError(f"不支持的数据维度: {image_data.ndim}")


def process_fy4b_disk_glt(file_1km, file_4km, file_geo, glt_row, glt_col, valid_mask, sqrt_lut):
    """
    使用GLT处理FY-4B圆盘数据（先处理图像，最后进行GLT映射）
    """
    logger.info(f"开始处理: {os.path.basename(file_1km)}")

    # 验证输入文件
    if not validate_input_files(file_1km, file_4km, file_geo):
        return False

    try:
        # ===== 1. 加载RGB通道数据 =====
        logger.info("加载C01, C02, C03通道...")
        scn = Scene(filenames=[file_1km], reader='agri_fy4b_l1')
        scn.load(['C01', 'C02', 'C03'], calibration='reflectance')

        c01_raw = clean_data(scn['C01'].values)
        c02_raw = clean_data(scn['C02'].values)
        c03_raw = clean_data(scn['C03'].values)

        # 合成真彩色
        new_red_dn = (c02_raw - 0.14 * c03_raw)
        new_green_dn = (c01_raw * 0.6 + c02_raw * 0.33 + c03_raw * 0.07)
        new_blue_dn = c01_raw

        rgb_dn = np.stack([new_red_dn, new_green_dn, new_blue_dn], axis=-1)
        rgb_dn = np.maximum(rgb_dn, 0)

        # 创建有效掩膜
        valid_mask_original = (c01_raw > 0).astype(np.uint8)

        # 清理临时变量
        del c01_raw, c02_raw, c03_raw, new_red_dn, new_green_dn, new_blue_dn
        gc.collect()

        # ===== 2. 应用平方根变换和缩放 =====
        rgb_sqrt_dn = np.power(rgb_dn.astype(np.float32), 0.5)
        rgb_scaled = rgb_sqrt_dn * 0.090
        rgb_float = rgb_scaled / FIXED_NORMALIZER

        del rgb_dn, rgb_sqrt_dn, rgb_scaled
        gc.collect()

        # ===== 3. 大气校正 =====
        logger.info("应用大气校正...")
        rgb_float = atmospheric_correction(rgb_float, strength=ATMOSPHERE_STRENGTH)

        # ===== 4. Filmic色调映射 =====
        logger.info("应用Filmic色调映射...")
        rgb_tonemapped = filmic_tonemap(rgb_float, white_point=TONEMAP_WHITE_POINT, gain=TONEMAP_GAIN)
        rgb_8bit = np.clip(rgb_tonemapped * 255, 0, 255).astype(np.uint8)

        del rgb_float, rgb_tonemapped
        gc.collect()

        # ===== 5. 波段校正 =====
        logger.info("应用波段线性校正...")
        rgb_8bit = apply_band_correction(rgb_8bit, [BAND_CORRECTION_R, BAND_CORRECTION_G, BAND_CORRECTION_B])

        # ===== 6. CLAHE增强 =====
        if USE_CLAHE:
            logger.info("应用CLAHE增强...")
            lab = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                    tileGridSize=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE))
            l_clahe = clahe.apply(l)
            rgb_8bit = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2RGB)

        # ===== 7. 饱和度增强 =====
        logger.info("增强饱和度...")
        rgb_8bit = enhance_saturation(rgb_8bit, factor=SATURATION_FACTOR)

        # ===== 8. 抖动噪声 =====
        if DITHER_NOISE_STD > 0:
            noise = np.random.normal(0, DITHER_NOISE_STD, rgb_8bit.shape).astype(np.float32)
            rgb_8bit = np.clip(rgb_8bit.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            del noise
            gc.collect()

        # ===== 9. 读取太阳天顶角(SZA) =====
        logger.info("读取太阳天顶角数据...")
        with h5py.File(file_geo, 'r') as f:
            sun_zenith_raw = f['Navigation']['NOMSunZenith'][:]
        
        # 注意：SZA不需要GLT映射，因为它是用于计算昼夜掩膜的
        # 但需要与RGB图像尺寸匹配
        sza_resized = cv2.resize(
            sun_zenith_raw.astype(np.float32),
            (rgb_8bit.shape[1], rgb_8bit.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        del sun_zenith_raw
        gc.collect()

        # ===== 10. 读取红外通道C14 =====
        logger.info("读取红外通道C14...")
        with h5py.File(file_4km, 'r') as f:
            ir_data = f['Data']['NOMChannel14'][:]

        ir_clean = np.where(ir_data == 65535, 0, ir_data).astype(np.float32)
        ir_norm = ir_clean / 4095.0
        
        # 红外数据也需要resize到RGB尺寸
        ir_resized = cv2.resize(
            ir_norm,
            (rgb_8bit.shape[1], rgb_8bit.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # 噪声阈值处理
        noise_threshold = 0.03
        ir_clean_resized = np.where(ir_resized < noise_threshold, 0, ir_resized)

        # 生成夜间图层
        night_layer_gray = (ir_clean_resized * 255).astype(np.uint8)
        night_layer_rgb = cv2.cvtColor(night_layer_gray, cv2.COLOR_GRAY2RGB)

        # 应用Gamma校正
        ir_gamma = 3.5
        lut_ir = np.array([((i / 255.0) ** ir_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        night_layer_rgb = cv2.LUT(night_layer_rgb, lut_ir)

        del ir_data, ir_clean, ir_norm, ir_resized, ir_clean_resized, night_layer_gray
        gc.collect()

        # ===== 11. 融合夜间灯光数据 =====
        if os.path.exists(NIGHT_LIGHT_JPG_PATH):
            logger.info("融合夜间灯光数据...")
            nl_ext = cv2.imread(NIGHT_LIGHT_JPG_PATH)
            nl_ext = cv2.cvtColor(nl_ext, cv2.COLOR_BGR2RGB)
            nl_ext = cv2.resize(nl_ext, (rgb_8bit.shape[1], rgb_8bit.shape[0]), interpolation=cv2.INTER_LINEAR)

            base_n = night_layer_rgb.astype(np.float32) / 255.0
            blend_n = nl_ext.astype(np.float32) / 255.0
            screen_blend = 1.0 - (1.0 - base_n) * (1.0 - blend_n * 0.6)
            night_layer_rgb = np.clip(screen_blend * 255, 0, 255).astype(np.uint8)

            del base_n, blend_n, screen_blend, nl_ext
            gc.collect()

        # ===== 12. 昼夜融合 =====
        logger.info("进行昼夜融合...")
        sza = sza_resized
        day_mask = np.ones_like(sza, dtype=np.float32)
        transition_zone = (sza >= 70) & (sza <= 100)
        day_mask[transition_zone] = (100.0 - sza[transition_zone]) / 30.0
        day_mask[sza > 100] = 0.0

        # 日间部分
        final_image_orig = rgb_8bit.astype(np.float32) / 255.0
        final_image_orig *= day_mask[:, :, np.newaxis]

        # 夜间部分
        night_float = night_layer_rgb.astype(np.float32) / 255.0
        night_weight = 1.0 - day_mask
        night_float *= night_weight[:, :, np.newaxis]

        # 合并
        final_image_orig += night_float
        final_image_orig = np.clip(final_image_orig * 255, 0, 255).astype(np.uint8)

        # 应用有效掩膜（原始图像掩膜）
        mask_3d_orig = np.stack([valid_mask_original] * 3, axis=-1)
        final_image_orig = np.where(mask_3d_orig, final_image_orig, 0)

        del rgb_8bit, night_layer_rgb, day_mask, night_float, night_weight, sza, sza_resized
        del valid_mask_original, mask_3d_orig
        gc.collect()

        # ===== 13. 使用GLT重投影最终图像 =====
        logger.info("使用GLT重投影最终图像...")
        glt_rows, glt_cols = glt_row.shape
        logger.info(f"  输出尺寸: {glt_rows} x {glt_cols}")
        
        # 应用GLT映射
        final_image_glt = apply_glt_to_image(final_image_orig, glt_row, glt_col, valid_mask)
        
        # 清理
        del final_image_orig
        gc.collect()

        # ===== 14. 保存结果 =====
        out_name = os.path.basename(file_1km).replace('.HDF', '_GLT_TRUE_COLOR.jpg')
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        img_out = Image.fromarray(final_image_glt)
        img_out.save(out_path, quality=JPG_QUALITY, optimize=True)
        logger.info(f"成功保存: {out_path}")

        # 清理内存
        del final_image_glt, img_out
        gc.collect()

        return True

    except MemoryError as e:
        logger.error(f"内存不足: {e}")
        gc.collect()
        return False
    except Exception as e:
        logger.error(f"处理出错: {e}", exc_info=True)
        return False


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print(" FY-4B AGRI 处理工具 (GLT重投影版本)")
    print(" ★ 使用GLT进行快速重投影 ★")
    print("=" * 60)
    logger.info(f"输入文件夹: {INPUT_FOLDER}")
    logger.info(f"输出文件夹: {OUTPUT_FOLDER}")
    logger.info(f"GLT行号文件: {GLT_ROW_PATH}")
    logger.info(f"GLT列号文件: {GLT_COL_PATH}")
    logger.info(f"输出分辨率: {OUTPUT_RESOLUTION_DEGREE}度")

    # 加载GLT查找表
    try:
        glt_row, glt_col, valid_mask = load_glt(GLT_ROW_PATH, GLT_COL_PATH)
    except Exception as e:
        logger.error(f"加载GLT失败: {e}")
        exit()

    print("正在生成固定LUT...")
    sqrt_lut = generate_sqrt_lut(input_max=1.0, output_max=255, gamma=LUT_GAMMA)
    logger.info(f"LUT生成完成: input_max=1.0, gamma={LUT_GAMMA}")

    # 查找所有HDF文件
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.HDF"))
    if not files:
        logger.error(f"输入文件夹中未找到任何HDF文件: {INPUT_FOLDER}")
        exit()

    logger.info(f"找到 {len(files)} 个HDF文件")

    # 按时间戳分组
    groups = {}
    for f in files:
        match = re.search(r'(\d{14})', os.path.basename(f))
        if match:
            timestamp = match.group(1)
            if timestamp not in groups:
                groups[timestamp] = {}
            if '1000M' in f and 'GEO' not in f:
                groups[timestamp]['1km'] = f
            elif '4000M' in f and 'GEO' not in f:
                groups[timestamp]['4km'] = f
            elif 'GEO' in f and '4000M' in f:
                groups[timestamp]['geo'] = f

    if not groups:
        logger.error("未找到符合命名规则的有效文件组！")
        exit()

    logger.info(f"发现 {len(groups)} 个有效时间点")

    # 处理每个时间点
    success_count = 0
    for time, paths in sorted(groups.items()):
        file_1km = paths.get('1km')
        file_4km = paths.get('4km')
        file_geo = paths.get('geo')

        if file_1km and file_4km and file_geo:
            logger.info(f"处理时间 {time}...")
            if process_fy4b_disk_glt(file_1km, file_4km, file_geo, glt_row, glt_col, valid_mask, sqrt_lut):
                success_count += 1
        else:
            logger.warning(f"时间 {time} 缺少必要文件，跳过。")
            logger.debug(f"  file_1km: {file_1km}")
            logger.debug(f"  file_4km: {file_4km}")
            logger.debug(f"  file_geo: {file_geo}")

    print("=" * 60)
    logger.info(f"处理完成！成功处理 {success_count}/{len(groups)} 个时间点")
    print("=" * 60)