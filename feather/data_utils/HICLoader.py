# -*- coding: utf-8 -*-

'''
@author: Xu Yan
@file: HICLoader.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import json
import re
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd  # 保留（如无需要可删除）

warnings.filterwarnings('ignore')

# ===================== FPS 参数 =====================
PRESORT_FLAG = True        # 是否预排序点云
COARSESORT_SETTING = 16    # 粗排序设置（保留占位）
PARALLEL_OPTION = True     # 是否启用并行FPS
PARALLEL_M = 32            # 并行分区数
SELECT_DIM = 2             # 预排序维度（0=x,1=y,2=z）
SAVE_COMPUTATION_TWO_AXIS = False  # 是否只在两个未排序轴上计算距离（保留占位）

S73_EXCLUDE_LIST_PATH = os.getenv(
    "PT_HICNET_S73_EXCLUDE_LIST_PATH",
    str((Path(__file__).resolve().parents[2] / "data" / "S73_exclude_list.txt")),
)

# ===================== 车辆标识符映射 =====================
VEHICLE_MAPPING = {
    # 数字 -> 车辆名
    "1": "01-C201",
    "2": "02_EP32",
    "3": "03_JX65",
    "4": "04_S201",
    "5": "05_CX62B",
    "6": "06_CY02C",
    "7": "07_M6",
    "8": "08_S50EVK",
    "9": "09_S73",
    "10": "10_M4",
    "11": "11_FX11",
    "12": "12_ASE",
    "13": "13_X70",
    "14": "14_R11MCE",
    "15": "15XP",
    # 代码 -> 车辆名
    "C201": "01-C201",
    "EP32": "02_EP32",
    "JX65": "03_JX65",
    "S201": "04_S201",
    "CX62B": "05_CX62B",
    "CY02C": "06_CY02C",
    "M6": "07_M6",
    "S50EVK": "08_S50EVK",
    "S73": "09_S73",
    "M4": "10_M4",
    "FX11": "11_FX11",
    "ASE": "12_ASE",
    "X70": "13_X70",
    "R11MCE": "14_R11MCE",
    "XP": "15XP"
}

def extract_vehicle_identifier(file_path):
    """
    从文件路径/文件名中提取车辆标识符。
    规则：
    - all_data{num}.json  -> num -> VEHICLE_MAPPING
    - 文件名中包含车辆代码(C201/M6/...)或完整车辆名则采用
    - 父目录名中若包含车辆代码也可识别
    - 测试/验证集（包含 test/val/validation/eval）默认使用 "04_S201"
    - 否则返回 None（后续使用默认材料表）
    """
    filename = os.path.basename(file_path).replace('.json', '')

    # 模式1: all_data{num}
    match1 = re.search(r'all_data(\d+)', filename)
    if match1:
        vehicle_num = match1.group(1)
        if vehicle_num in VEHICLE_MAPPING:
            return VEHICLE_MAPPING[vehicle_num]

    # 模式2: 匹配 "数字_车辆代码" 格式（如 04_S201）
    match_num_code = re.search(r'(\d+)[_-]?([A-Z]\w*)', filename.upper())
    if match_num_code:
        num_part = match_num_code.group(1)
        code_part = match_num_code.group(2)
        # 尝试用数字部分查找
        if num_part in VEHICLE_MAPPING:
            return VEHICLE_MAPPING[num_part]
        # 尝试用代码部分查找
        if code_part in VEHICLE_MAPPING:
            return VEHICLE_MAPPING[code_part]
    
    # 模式3: 文件名中出现车辆代码（处理下划线和连字符兼容性）
    # 先尝试精确匹配数字（如 "15" -> "15XP"）
    if filename in VEHICLE_MAPPING:
        return VEHICLE_MAPPING[filename]
    
    # 尝试从文件名中提取数字（如 "15_XP" -> "15"）
    match_num = re.search(r'^(\d+)', filename)
    if match_num:
        num_str = match_num.group(1)
        if num_str in VEHICLE_MAPPING:
            return VEHICLE_MAPPING[num_str]
    
    # 尝试匹配车辆代码（忽略下划线和连字符）
    filename_normalized = filename.replace('_', '').replace('-', '').upper()
    for code, vehicle_name in VEHICLE_MAPPING.items():
        if len(code) > 2:
            code_normalized = code.replace('_', '').replace('-', '').upper()
            if code_normalized in filename_normalized or filename_normalized in code_normalized:
                return vehicle_name

    # 模式4: 文件名中出现完整车辆名（处理下划线和连字符）
    for vehicle_name in VEHICLE_MAPPING.values():
        vehicle_code = vehicle_name.split('-')[-1] if '-' in vehicle_name else vehicle_name.split('_')[-1]
        # 处理下划线和连字符的兼容性
        if vehicle_code in filename or vehicle_code.replace('_', '') in filename.replace('_', ''):
            return vehicle_name

    # 模式5: 父目录名（更严格的匹配，避免误匹配）
    parent_folder = os.path.basename(os.path.dirname(file_path))
    if parent_folder:
        # 优先精确匹配
        if parent_folder in VEHICLE_MAPPING:
            print(f"✓ 从父文件夹名 '{parent_folder}' 中识别车辆: {VEHICLE_MAPPING[parent_folder]}")
            return VEHICLE_MAPPING[parent_folder]
        
        # 匹配 "数字_车辆代码" 格式（如 04_S201）
        match_parent_num_code = re.search(r'(\d+)[_-]?([A-Z]\w*)', parent_folder.upper())
        if match_parent_num_code:
            num_part = match_parent_num_code.group(1)
            code_part = match_parent_num_code.group(2)
            # 尝试用数字部分查找
            if num_part in VEHICLE_MAPPING:
                print(f"✓ 从父文件夹名 '{parent_folder}' 中识别车辆: {VEHICLE_MAPPING[num_part]}")
                return VEHICLE_MAPPING[num_part]
            # 尝试用代码部分查找
            if code_part in VEHICLE_MAPPING:
                print(f"✓ 从父文件夹名 '{parent_folder}' 中识别车辆: {VEHICLE_MAPPING[code_part]}")
                return VEHICLE_MAPPING[code_part]
        
        # 避免单个数字的误匹配（如 "1" 在 "JSON1112" 中）
        # 只匹配长度>=2的代码，且要求是完整单词或精确匹配
        for code, vehicle_name in VEHICLE_MAPPING.items():
            if len(code) >= 2:
                # 精确匹配或作为完整单词出现
                if code == parent_folder or (code in parent_folder and len(code) >= 3):
                    print(f"✓ 从父文件夹名 '{parent_folder}' 中识别车辆: {vehicle_name}")
                    return vehicle_name

    # 模式6: 测试/验证文件默认
    filename_lower = filename.lower()
    if any(k in filename_lower for k in ['test', 'val', 'validation', 'eval']):
        print(f"⚠️ 无法从测试集文件名 {filename} 中提取车辆标识符，使用默认车辆: 04_S201")
        return "04_S201"

    print(f"⚠️ 无法从文件名 {filename} 中提取车辆标识符，将使用默认材料查找表")
    return None

# ===================== FPS 采样 =====================
def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return centroids.astype(np.int32)

def farthest_point_sample_improved(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]

    # 1) 预排序
    if PRESORT_FLAG:
        sorted_indices = np.argsort(xyz[:, SELECT_DIM])
        xyz = xyz[sorted_indices]
        point = point[sorted_indices]

    # 2) 分区并行（这里为简化仍串行执行，但保持接口/逻辑）
    if PARALLEL_OPTION and N > PARALLEL_M:
        partition_size = N // PARALLEL_M
        partitions = []
        for i in range(PARALLEL_M):
            s = i * partition_size
            e = (i + 1) * partition_size if i < PARALLEL_M - 1 else N
            partitions.append(xyz[s:e])

        points_per_partition = npoint // PARALLEL_M
        remaining_points = npoint % PARALLEL_M

        centroids = []
        for i, partition in enumerate(partitions):
            current_points = points_per_partition + (1 if i < remaining_points else 0)
            partition_centroids = farthest_point_sample(partition, current_points)
            global_indices = partition_centroids + i * partition_size
            centroids.extend(global_indices)
        centroids = np.array(centroids)
    else:
        centroids = farthest_point_sample(xyz, npoint)

    return centroids.astype(np.int32)

# ===================== 材料查找表 =====================
_global_material_lookup = None
_current_vehicle_name = None
_normalization_params = None  # 存储归一化参数 (min_vals, max_vals)

# 你自己的查找表路径
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MATERIAL_LOOKUP_PATH = os.getenv(
    "PT_HICNET_MATERIAL_LOOKUP_PATH",
    str(_DATA_DIR / "material_lookup_by_vehicle.pkl"),
)
NORMALIZATION_PARAMS_PATH = os.getenv(
    "PT_HICNET_NORMALIZATION_PARAMS_PATH",
    str(_DATA_DIR / "normalization_params.pkl"),
)

def load_normalization_params():
    """
    加载材料属性的归一化参数（Z-Score: mean_vals和std_vals）。
    返回：(mean_vals, std_vals, method) 或 (None, None, None)
    """
    global _normalization_params
    
    if _normalization_params is not None:
        return _normalization_params
    
    try:
        with open(NORMALIZATION_PARAMS_PATH, "rb") as f:
            params = pickle.load(f)
            
            # 检查归一化方法
            method = params.get('method', 'min-max')  # 默认为旧的min-max方法
            
            if method == 'z-score':
                # Z-Score 标准化参数
                mean_vals = params['mean_vals']
                std_vals = params['std_vals']
                _normalization_params = (mean_vals, std_vals, 'z-score')
                print(f"✅ 已加载材料属性归一化参数 (Z-Score标准化)")
            else:
                # 兼容旧的 Min-Max 方法
                min_vals = params['min_vals']
                max_vals = params['max_vals']
                _normalization_params = (min_vals, max_vals, 'min-max')
                print(f"✅ 已加载材料属性归一化参数 (Min-Max标准化)")
            
            return _normalization_params
    except FileNotFoundError:
        print(f"⚠️ 归一化参数文件不存在: {NORMALIZATION_PARAMS_PATH}")
        return None, None, None
    except Exception as e:
        print(f"❌ 加载归一化参数失败: {e}")
        return None, None, None

def normalize_material_properties(mat_props):
    """
    对材料属性进行归一化（使用全局归一化参数）。
    支持两种方法：Z-Score（推荐）和Min-Max（兼容旧模型）
    输入：15维原始材料属性列表或数组
    输出：15维归一化后的材料属性列表
    """
    param1, param2, method = load_normalization_params()
    
    if param1 is None or param2 is None:
        print("⚠️ 无法归一化材料属性：缺少归一化参数，返回原始值")
        return mat_props
    
    normalized = []
    
    if method == 'z-score':
        # Z-Score 标准化：(value - mean) / std
        mean_vals, std_vals = param1, param2
        for i, val in enumerate(mat_props):
            if i < len(mean_vals):
                # Z-Score 公式：不受极值限制，新车材料属性超出训练范围也能正确处理
                norm_val = (val - mean_vals[i]) / std_vals[i]
            else:
                norm_val = 0.0
            normalized.append(norm_val)
    else:
        # Min-Max 标准化（兼容旧方法）：(value - min) / (max - min)
        min_vals, max_vals = param1, param2
        for i, val in enumerate(mat_props):
            if i < len(max_vals):
                if max_vals[i] - min_vals[i] > 0:
                    norm_val = (val - min_vals[i]) / (max_vals[i] - min_vals[i])
                else:
                    norm_val = 0.0
            else:
                norm_val = 0.0
            normalized.append(norm_val)
    
    return normalized

def load_material_lookup_for_vehicle(vehicle_name):
    """
    根据车辆名称加载材料查找表。
    返回：numpy [max_mat_id+1, 15]。若缺失则返回零表。
    """
    global _global_material_lookup, _current_vehicle_name

    # 如果 vehicle_name 是 None（测试时从临时文件名无法识别车辆），返回空表
    # 此时材料属性应该直接从JSON读取，而不需要查表
    if vehicle_name is None:
        print(f"ℹ️  车辆标识符未知，将直接从JSON读取材料属性（不使用查找表）")
        _global_material_lookup = np.zeros((100, 15), dtype=np.float32)
        _current_vehicle_name = None
        return _global_material_lookup

    if vehicle_name == _current_vehicle_name and _global_material_lookup is not None:
        return _global_material_lookup

    try:
        with open(MATERIAL_LOOKUP_PATH, "rb") as f:
            material_lookup_dict = pickle.load(f)

                # ---- 兼容不同命名：空格/下划线/连字符，以及 .xlsx 扩展名 ----
        def _resolve_vehicle_key(name: str, keys):
            cand = set()
            # 原样
            cand.add(name)
            # 空格 ↔ 下划线 / 连字符互换
            cand.add(name.replace(' ', '_'))
            cand.add(name.replace(' ', '-'))
            cand.add(name.replace('_', '-'))
            cand.add(name.replace('-', '_'))
            
            # 特殊处理：数字+字母组合（如 "15XP" ↔ "15_XP"）
            import re as _re
            match = _re.match(r'^(\d+)([A-Za-z]\w*)$', name)
            if match:
                num_part, letter_part = match.groups()
                cand.add(f"{num_part}_{letter_part}")
                cand.add(f"{num_part}-{letter_part}")
            
            # 加上 .xlsx
            base = list(cand)
            for x in base:
                cand.add(x + '.xlsx')
            # 逐个尝试
            for k in cand:
                if k in keys:
                    return k
            return None

        resolved_key = _resolve_vehicle_key(vehicle_name, material_lookup_dict.keys())
        if resolved_key is None:
            print(f"⚠️ 车辆 {vehicle_name} 的材料数据未找到（可用键示例：如含 .xlsx 或下划线）。使用空的材料查找表。")
            _global_material_lookup = np.zeros((100, 15), dtype=np.float32)
            _current_vehicle_name = vehicle_name
            return _global_material_lookup

        # 获取该车辆的材料字典（用匹配到的键）
        vehicle_materials = material_lookup_dict[resolved_key]
        max_mat_id = max(vehicle_materials.keys()) if vehicle_materials else 0
        lookup_table = np.zeros((max_mat_id + 1, 15), dtype=np.float32)
        for mat_id, mat_vec in vehicle_materials.items():
            lookup_table[mat_id] = mat_vec

        _global_material_lookup = lookup_table
        _current_vehicle_name = vehicle_name
        print(f"✅ 已加载车辆 {vehicle_name} 的材料查找表，共 {len(vehicle_materials)} 条")
        return _global_material_lookup

    except FileNotFoundError:
        print(f"❌ 材料查找表文件不存在: {MATERIAL_LOOKUP_PATH}，使用空表")
        _global_material_lookup = np.zeros((100, 15), dtype=np.float32)
        _current_vehicle_name = vehicle_name
        return _global_material_lookup
    except Exception as e:
        print(f"❌ 加载车辆 {vehicle_name} 材料查找表出错: {e}，使用空表")
        _global_material_lookup = np.zeros((100, 15), dtype=np.float32)
        _current_vehicle_name = vehicle_name
        return _global_material_lookup

# ===================== 工具函数 =====================
def _safe_float(v, default=0.0):
    """将可能为字符串/None的数值安全转换为 float。"""
    try:
        if v is None or v == '':
            return default
        return float(v)
    except Exception:
        return default

# ===================== 数据集 =====================
class HICDataLoader(Dataset):
    def __init__(self, root, args, process_data=False, early_fusion=True, normalize_thickness=True):
        self.root = root
        self.args = args
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.early_fusion = early_fusion
        self.normalize_thickness = normalize_thickness

        # 车辆识别
        self.vehicle_name = extract_vehicle_identifier(root)
        print(f"🚗 数据文件 {os.path.basename(root)} 对应车辆: {self.vehicle_name}")

        self.data = None
        self.datapoints = None

    def _normalize_thickness_values(self, thickness):
        if not self.normalize_thickness or thickness.size == 0:
            return thickness
        min_v = float(np.min(thickness))
        max_v = float(np.max(thickness))
        if max_v - min_v < 1e-8:
            return np.zeros_like(thickness, dtype=np.float32)
        return ((thickness - min_v) / (max_v - min_v + 1e-8)).astype(np.float32)

    # === 读取 JSON 并构建样本 ===
    def _load_data(self):
        if self.data is not None:
            return

        # 1) 读取 JSON
        with open(self.root, 'r') as f:
            self.data = json.load(f)

        # 2) 加载材料查找表（按车辆）
        material_lookup_table = load_material_lookup_for_vehicle(self.vehicle_name)

        # 3) 展开样本
        self.datapoints = []
        excluded_samples = []
        for sample_id, point_data in self.data.items():
            # ---------- nearby_nodes ----------
            nodes = []
            thickness = []
            material_props = []

            for node in point_data.get('nearby_nodes', []):
                # 坐标（新/旧 JSON 可能是字符串）
                node_coords = [
                    _safe_float(node.get('X', 0)),
                    _safe_float(node.get('Y', 0)),
                    _safe_float(node.get('Z', 0))
                ]
                nodes.append(node_coords)

                # 厚度
                thickness_value = _safe_float(node.get('Thickness', 0.0), 0.0)
                thickness.append([thickness_value])

                # 材料属性处理：优先从JSON直接读取，否则通过MID查表
                # 检查JSON中是否直接包含15维材料属性字段
                mat_vec = None
                
                # 方式1：尝试直接从JSON节点中读取15维材料属性
                # 支持的字段名：密度RO/密度R0, 杨氏模量E, 泊松比PR, 以及应力应变曲线字段
                if all(key in node for key in ['密度RO', '杨氏模量E', '泊松比PR']):
                    try:
                        mat_vec_raw = [
                            _safe_float(node.get('密度RO', 0.0) or node.get('密度R0', 0.0), 0.0),
                            _safe_float(node.get('杨氏模量E', 0.0), 0.0),
                            _safe_float(node.get('泊松比PR', 0.0), 0.0),
                            # 0.001应力应变曲线 (6个点)
                            _safe_float(node.get('0.001应力应变曲线0', 0.0) or node.get('0.001应力应变曲线_0', 0.0), 0.0),
                            _safe_float(node.get('0.001应力应变曲线0.05', 0.0) or node.get('0.001应力应变曲线_0.05', 0.0), 0.0),
                            _safe_float(node.get('0.001应力应变曲线0.1', 0.0) or node.get('0.001应力应变曲线_0.1', 0.0), 0.0),
                            _safe_float(node.get('0.001应力应变曲线0.15', 0.0) or node.get('0.001应力应变曲线_0.15', 0.0), 0.0),
                            _safe_float(node.get('0.001应力应变曲线0.2', 0.0) or node.get('0.001应力应变曲线_0.2', 0.0), 0.0),
                            _safe_float(node.get('0.001应力应变曲线0.5', 0.0) or node.get('0.001应力应变曲线_0.5', 0.0), 0.0),
                            # 1应力应变曲线 (6个点)
                            _safe_float(node.get('1应力应变曲线0', 0.0) or node.get('1应力应变曲线_0', 0.0), 0.0),
                            _safe_float(node.get('1应力应变曲线0.05', 0.0) or node.get('1应力应变曲线_0.05', 0.0), 0.0),
                            _safe_float(node.get('1应力应变曲线0.1', 0.0) or node.get('1应力应变曲线_0.1', 0.0), 0.0),
                            _safe_float(node.get('1应力应变曲线0.15', 0.0) or node.get('1应力应变曲线_0.15', 0.0), 0.0),
                            _safe_float(node.get('1应力应变曲线0.2', 0.0) or node.get('1应力应变曲线_0.2', 0.0), 0.0),
                            _safe_float(node.get('1应力应变曲线0.5', 0.0) or node.get('1应力应变曲线_0.5', 0.0), 0.0),
                        ]
                        # 关键修复：对从JSON直接读取的原始材料属性进行归一化
                        mat_vec = normalize_material_properties(mat_vec_raw)
                    except Exception:
                        mat_vec = None
                
                # 方式2：如果JSON中没有直接的材料属性，则通过MID查表
                if mat_vec is None:
                    mid_raw = node.get('MID', None)
                    if mid_raw in (None, ''):
                        # 旧 JSON 兜底：MatName（旧字段名）
                        mid_raw = node.get('MatName', None)

                    if mid_raw in (None, ''):
                        mat_vec = [0.0] * 15
                    else:
                        try:
                            # 兼容 "26" / "26.0" / 26
                            mid = int(float(mid_raw))
                            if 0 <= mid < material_lookup_table.shape[0]:
                                mat_vec = material_lookup_table[mid].tolist()
                            else:
                                mat_vec = [0.0] * 15
                        except Exception:
                            mat_vec = [0.0] * 15

                material_props.append(mat_vec)

            # ---------- hic_point ----------
            hic_point = point_data['hic_point']

            # HIC 值：新字段 HIC，兼容旧字段 hic_value
            hic_value_raw = hic_point.get('HIC', None)
            if hic_value_raw in (None, ''):
                hic_value_raw = hic_point.get('hic_value', None)
            if hic_value_raw in (None, ''):
                print(f"警告: 样本 {sample_id} 缺少 HIC 值，跳过")
                continue

            hic_value = _safe_float(hic_value_raw, None)
            if hic_value is None:
                print(f"警告: 样本 {sample_id} HIC 无法解析，跳过")
                continue

            if hic_value == 0:
                excluded_samples.append(sample_id)
                continue

            # 坐标
            hp_x = _safe_float(hic_point.get('X', 0.0), 0.0)
            hp_y = _safe_float(hic_point.get('Y', 0.0), 0.0)
            hp_z = _safe_float(hic_point.get('Z', 0.0), 0.0)

            # 类别/年龄段
            category_raw = hic_point.get('class', 0)
            try:
                category = int(float(category_raw))
            except Exception:
                category = 0

            age_group = hic_point.get('age_group', 'Adult')

            # ---------- 组装样本 ----------
            # 转为 np 数组并清理
            thickness_array = np.nan_to_num(np.array(thickness))
            material_props_array = np.nan_to_num(np.array(material_props))

            self.datapoints.append({
                'point_set': nodes,
                'hic_point': {'x': hp_x, 'y': hp_y, 'z': hp_z},
                'category': category,
                'hic_value': float(hic_value),
                'thickness': thickness_array.tolist(),
                'material_props': material_props_array.tolist(),
                'age_group': age_group
            })

            if len(nodes) == 0:
                print(f"Warning: No nodes found for sample {sample_id}")

        if excluded_samples:
            print(f"已排除 {len(excluded_samples)} 个HIC为0的样本点：{excluded_samples}")

        print(f"Loaded {len(self.datapoints)} samples")

    def __len__(self):
        if self.datapoints is None:
            self._load_data()
        return len(self.datapoints)

    def __getitem__(self, index):
        if self.datapoints is None:
            self._load_data()
        return self._get_item(index)

    def _get_item(self, index):
        """
        返回：point_set, hic_point, category, thickness, material_props, age_group, label
        - point_set: (num_point, 3) float32
        - hic_point: (3,) float32
        - category: int
        - thickness: (num_point, 1) float32
        - material_props: (num_point, 15) float32
        - age_group: int (0=Children, 1=Adult)
        - label: (1,) float32
        """
        # 确保数据已加载
        if self.datapoints is None:
            self._load_data()

        data_point = self.datapoints[index]

        # --- 基础项 ---
        point_set = np.array(data_point['point_set'], dtype=np.float32)        # [N, 3]
        hic_point = np.array([
            data_point['hic_point']['x'],
            data_point['hic_point']['y'],
            data_point['hic_point']['z']
        ], dtype=np.float32)
        hic_value = float(data_point['hic_value'])
        category = int(data_point['category'])
        label = np.array([hic_value], dtype=np.float32)

        # --- 附加特征，先转为 ndarray，再进行形状/维度修正 ---
        thickness = np.array(data_point['thickness'], dtype=np.float32)       # 期望 [N,1] 或 [N,]
        material_props = np.array(data_point['material_props'], dtype=np.float32)  # 期望 [N,15]

        # 处理极端情况：point_set 为空（理论上在 _load_data 已跳过，但双重保险）
        if point_set.size == 0 or point_set.shape[0] == 0:
            raise ValueError(f"Empty point_set for sample index {index} in file {self.root}")

        N = point_set.shape[0]

        # thickness -> 保证为 (N, 1)
        if thickness.ndim == 1:
            thickness = thickness.reshape(-1, 1)
        elif thickness.ndim == 0:
            thickness = np.array([[float(thickness)]], dtype=np.float32)
        # 若 thickness 行数与点数不一致，则通过重复或截断对齐
        if thickness.shape[0] != N:
            if thickness.shape[0] == 0:
                thickness = np.zeros((N, 1), dtype=np.float32)
            else:
                # 重复最后一行或循环重复以匹配长度
                reps = int(np.ceil(N / max(1, thickness.shape[0])))
                thickness = np.tile(thickness, (reps, 1))[:N, :]

        # material_props 处理 -> 保证为 (N, 15)
        if material_props.ndim == 1 and material_props.size == 15:
            material_props = material_props.reshape(1, 15)
        elif material_props.ndim == 0:
            material_props = np.zeros((1, 15), dtype=np.float32)

        if material_props.ndim == 1 and material_props.size == 0:
            material_props = np.zeros((0, 15), dtype=np.float32)

        # 若 material_props 行数与点数不一致，采用重复/截断/填零策略
        if material_props.shape[0] != N:
            if material_props.shape[0] == 0:
                material_props = np.zeros((N, 15), dtype=np.float32)
            else:
                # 重复直到 >= N，然后截断
                reps = int(np.ceil(N / material_props.shape[0]))
                material_props = np.tile(material_props, (reps, 1))[:N, :]

        # 最终核对列数
        if material_props.shape[1] != 15:
            # 若仍然不对则强制回退为 zeros，避免 downstream 崩溃
            material_props = np.zeros((N, 15), dtype=np.float32)

        # --- 年龄组编码 ---
        age_raw = data_point.get('age_group', 'Adult')
        age_group = 0 if str(age_raw).lower().startswith('child') else 1

        # --- 采样 / 填充（统一索引策略） ---
        # N 为当前样本原始点数；目标为 self.npoints
        if N <= 0:
            raise ValueError(f"Invalid point count ({N}) for sample {index} in {self.root}")

        if N >= self.npoints:
            # 足够点：不放回采样或 FPS
            if self.uniform:
                try:
                    indices = farthest_point_sample_improved(point_set, self.npoints)
                except Exception:
                    # FPS 可能出错时降级为随机无放回采样
                    indices = np.random.choice(N, self.npoints, replace=False)
            else:
                indices = np.random.choice(N, self.npoints, replace=False)
        else:
            # 点不足：有放回随机采样补齐
            indices = np.random.choice(N, self.npoints, replace=True)

        # 安全转换并避免越界（保险）
        indices = np.array(indices, dtype=np.int64)
        indices = np.clip(indices, 0, N - 1)

        # 按索引取值
        try:
            point_set = point_set[indices]
            thickness = thickness[indices]
            material_props = material_props[indices]
        except Exception as e:
            # 如果索引操作发生异常，输出详细信息以便调试
            raise RuntimeError(f"Indexing failed in _get_item for sample {index}, N={N}, npoints={self.npoints}, "
                            f"indices_min={indices.min()}, indices_max={indices.max()}, error={e}")

        # 确保返回类型
        point_set = point_set.astype(np.float32)
        thickness = thickness.astype(np.float32)
        material_props = material_props.astype(np.float32)
        hic_point = hic_point.astype(np.float32)
        label = label.astype(np.float32)

        thickness = self._normalize_thickness_values(thickness)
        if self.early_fusion:
            fused_input = np.concatenate([point_set, thickness, material_props], axis=1).astype(np.float32)
            return fused_input, hic_point, category, age_group, label

        return point_set, hic_point, category, thickness, material_props, age_group, label

if __name__ == '__main__':
    import argparse
    import types
    import torch

    # 便捷本地测试
    dummy_args = types.SimpleNamespace(
        num_point=1024,
        use_uniform_sample=True,
        use_normals=False
    )

    # 替换为你的 json 路径测试
    # data = HICDataLoader('/path/to/your/new_format.json', dummy_args, early_fusion=True)
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    # for batch in DataLoader:
    #     fused_input, hic_pt, cat, age, lbl = batch
    #     print("fused_input:", fused_input.shape, "hic:", hic_pt.shape, "label:", lbl.shape)