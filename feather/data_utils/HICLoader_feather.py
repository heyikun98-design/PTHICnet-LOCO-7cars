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

# ===================== S73 样本点排除控制 =====================
EXCLUDE_S73_SAMPLES = True  # 是否排除S73的样本点（设置为False则不排除）
S73_EXCLUDE_LIST_PATH = os.getenv(
    "PT_HICNET_S73_EXCLUDE_LIST_PATH",
    str((Path(__file__).resolve().parents[2] / "data" / "S73_exclude_list.txt")),
)  # S73排除列表文件路径
_s73_exclude_set = None  # 全局变量，存储需要排除的S73样本ID集合

def load_s73_exclude_list():
    """
    加载S73需要排除的样本ID列表。
    返回：样本ID的集合（set），如果文件不存在或加载失败则返回空集合。
    """
    global _s73_exclude_set
    
    if not EXCLUDE_S73_SAMPLES:
        _s73_exclude_set = set()
        return _s73_exclude_set
    
    if _s73_exclude_set is not None:
        return _s73_exclude_set
    
    _s73_exclude_set = set()
    try:
        if os.path.exists(S73_EXCLUDE_LIST_PATH):
            with open(S73_EXCLUDE_LIST_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    sample_id = line.strip()
                    if sample_id:  # 忽略空行
                        _s73_exclude_set.add(sample_id)
            print(f"✅ 已加载S73排除列表，共 {len(_s73_exclude_set)} 个样本ID")
        else:
            print(f"⚠️ S73排除列表文件不存在: {S73_EXCLUDE_LIST_PATH}")
    except Exception as e:
        print(f"❌ 加载S73排除列表失败: {e}")
        _s73_exclude_set = set()
    
    return _s73_exclude_set

# ===================== 车辆标识符映射 =====================
# 车辆代码列表（不带数字前缀）
VEHICLE_CODES = [
    "C201",
    "EP32",
    "JX65",
    "S201",
    "CX62B",
    "CY02C",
    "M6",
    "S50EVK",
    "S73",
    "M4",
    "FX11",
    "ASE",
    "X70",
    "R11MCE",
    "XP"
]

CAR_TO_VEHICLE = {
    "car1": "C201",
    "car2": "EP32",
    "car3": "JX65",
    "car4": "CY02C",
    "car5": "M6",
    "car6": "S50EVK",
    "car7": "FX11",
}


def extract_vehicle_identifier(file_path):
    """
    从文件路径/文件名中提取车辆标识符。
    优先级：CAR_TO_VEHICLE 映射 > 文件名匹配 > 父目录匹配 > 测试集默认
    """
    filename = os.path.basename(file_path).replace('.json', '').replace('.feather', '')
    parent_folder = os.path.basename(os.path.dirname(file_path))

    # 模式0: 检查 car1~car7 映射
    if parent_folder.lower() in CAR_TO_VEHICLE:
        return CAR_TO_VEHICLE[parent_folder.lower()]

    filename_upper = filename.upper()

    # 模式1: 文件名中包含车辆代码
    for code in VEHICLE_CODES:
        code_upper = code.upper()
        filename_normalized = filename_upper.replace('_', '').replace('-', '')
        code_normalized = code_upper.replace('_', '').replace('-', '')
        if code_upper in filename_upper or code_normalized in filename_normalized:
            return code

    # 模式2: 父目录名中包含车辆代码
    if parent_folder:
        parent_upper = parent_folder.upper()
        for code in VEHICLE_CODES:
            code_upper = code.upper()
            parent_normalized = parent_upper.replace('_', '').replace('-', '')
            code_normalized = code_upper.replace('_', '').replace('-', '')
            if code_upper in parent_upper or code_normalized in parent_normalized:
                return code

    # 模式3: 测试/验证文件默认
    filename_lower = filename.lower()
    if any(k in filename_lower for k in ['test', 'val', 'validation', 'eval']):
        return "S201"

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

    # === 读取 JSON/Feather 并构建样本 ===
    def _load_data(self):
        if self.data is not None:
            return

        # 1) 根据文件扩展名选择读取方式
        file_ext = os.path.splitext(self.root)[1].lower()
        
        # 2) 加载材料查找表（按车辆）
        material_lookup_table = load_material_lookup_for_vehicle(self.vehicle_name)
        
        # 2.5) 如果是S73车辆，加载排除列表
        s73_exclude_set = set()
        if self.vehicle_name == "S73":
            s73_exclude_set = load_s73_exclude_list()

        # 3) 展开样本
        self.datapoints = []
        excluded_samples = []  # 记录被排除的样本（HIC值为0）
        excluded_s73_samples = []  # 记录被排除的S73样本
        
        if file_ext == '.feather':
            # 读取 Feather 格式并直接处理（不转换为字典，减少内存占用）
            # print(f"📦 检测到 Feather 格式，使用 pandas 读取: {os.path.basename(self.root)}")
            try:
                df = pd.read_feather(self.root)
                # print(f"📊 Feather 文件包含 {len(df)} 行数据")
                
                # 直接从 DataFrame 处理，避免转换为字典（减少内存占用）
                for idx, row in df.iterrows():
                    file_id = str(row['file_id'])
                    point_data = row['data']
                    # 保持 nearby_nodes 为原始 numpy array，不做 dict 复制，避免内存翻倍
                    self._process_sample(file_id, point_data, material_lookup_table, excluded_samples,
                                        s73_exclude_set, excluded_s73_samples)
                    del point_data  # 显式释放引用
                
                # 释放 DataFrame 以释放内存
                del df
                import gc
                gc.collect()
                
                print(f"✅ 成功读取 Feather 文件，共 {len(self.datapoints)} 个样本")
                # print(f"💡 优化：直接从 DataFrame 处理，避免了 DataFrame → 字典的转换，减少了内存占用")
            except Exception as e:
                print(f"❌ 读取 Feather 文件失败: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            # 读取 JSON 格式（保持原有逻辑）
            # print(f"📄 检测到 JSON 格式，使用 json 读取: {os.path.basename(self.root)}")
            with open(self.root, 'r') as f:
                self.data = json.load(f)
            
            # JSON 格式需要遍历字典
            for sample_id, point_data in self.data.items():
                self._process_sample(sample_id, point_data, material_lookup_table, excluded_samples,
                                    s73_exclude_set, excluded_s73_samples)
        
        # 输出被排除的样本点信息
        if excluded_s73_samples:
            print(f"⚠️ 已排除 {len(excluded_s73_samples)} 个S73样本点: {excluded_s73_samples}")
        if excluded_samples:
            print(f"⚠️ 已排除 {len(excluded_samples)} 个HIC值为0的样本点: {excluded_samples}")
        print(f"Loaded {len(self.datapoints)} samples")
    
    def _process_sample(self, sample_id, point_data, material_lookup_table, excluded_samples, 
                       s73_exclude_set=None, excluded_s73_samples=None):
        """
        处理单个样本，提取节点、厚度、材料属性等信息。
        这个函数被 JSON 和 Feather 两种格式共用。
        
        参数:
            sample_id: 样本ID
            point_data: 样本数据
            material_lookup_table: 材料查找表
            excluded_samples: 记录HIC值为0的排除样本列表
            s73_exclude_set: S73排除样本ID集合
            excluded_s73_samples: 记录被排除的S73样本列表
        """
        # 如果是S73车辆且该样本在排除列表中，则跳过
        if s73_exclude_set is not None and len(s73_exclude_set) > 0:
            if sample_id in s73_exclude_set:
                if excluded_s73_samples is not None:
                    excluded_s73_samples.append(sample_id)
                return  # 跳过该样本
        # ---------- nearby_nodes ----------
        nearby_nodes = point_data.get('nearby_nodes', [])
        N = len(nearby_nodes)
        if N == 0:
            print(f"Warning: No nodes found for sample {sample_id}")
            return

        # 预分配 numpy 数组，避免 Python list 内存膨胀
        nodes = np.empty((N, 3), dtype=np.float32)
        thickness = np.empty((N, 1), dtype=np.float32)

        # 探测节点是否直接携带 15 维材料属性
        first_node = nearby_nodes[0]
        has_direct_mat = all(key in first_node for key in ['密度RO', '杨氏模量E', '泊松比PR'])
        if has_direct_mat:
            material_props = np.empty((N, 15), dtype=np.float32)
        else:
            material_props = np.empty((N, 15), dtype=np.float32)

        for i, node in enumerate(nearby_nodes):
            nodes[i, 0] = _safe_float(node.get('X', 0))
            nodes[i, 1] = _safe_float(node.get('Y', 0))
            nodes[i, 2] = _safe_float(node.get('Z', 0))
            thickness[i, 0] = _safe_float(node.get('Thickness', 0.0), 0.0)

            if has_direct_mat:
                try:
                    mat_vec_raw = [
                        _safe_float(node.get('密度RO', 0.0) or node.get('密度R0', 0.0), 0.0),
                        _safe_float(node.get('杨氏模量E', 0.0), 0.0),
                        _safe_float(node.get('泊松比PR', 0.0), 0.0),
                        _safe_float(node.get('0.001应力应变曲线0', 0.0) or node.get('0.001应力应变曲线_0', 0.0), 0.0),
                        _safe_float(node.get('0.001应力应变曲线0.05', 0.0) or node.get('0.001应力应变曲线_0.05', 0.0), 0.0),
                        _safe_float(node.get('0.001应力应变曲线0.1', 0.0) or node.get('0.001应力应变曲线_0.1', 0.0), 0.0),
                        _safe_float(node.get('0.001应力应变曲线0.15', 0.0) or node.get('0.001应力应变曲线_0.15', 0.0), 0.0),
                        _safe_float(node.get('0.001应力应变曲线0.2', 0.0) or node.get('0.001应力应变曲线_0.2', 0.0), 0.0),
                        _safe_float(node.get('0.001应力应变曲线0.5', 0.0) or node.get('0.001应力应变曲线_0.5', 0.0), 0.0),
                        _safe_float(node.get('1应力应变曲线0', 0.0) or node.get('1应力应变曲线_0', 0.0), 0.0),
                        _safe_float(node.get('1应力应变曲线0.05', 0.0) or node.get('1应力应变曲线_0.05', 0.0), 0.0),
                        _safe_float(node.get('1应力应变曲线0.1', 0.0) or node.get('1应力应变曲线_0.1', 0.0), 0.0),
                        _safe_float(node.get('1应力应变曲线0.15', 0.0) or node.get('1应力应变曲线_0.15', 0.0), 0.0),
                        _safe_float(node.get('1应力应变曲线0.2', 0.0) or node.get('1应力应变曲线_0.2', 0.0), 0.0),
                        _safe_float(node.get('1应力应变曲线0.5', 0.0) or node.get('1应力应变曲线_0.5', 0.0), 0.0),
                    ]
                    material_props[i] = normalize_material_properties(mat_vec_raw)
                except Exception:
                    material_props[i] = 0.0
            else:
                mid_raw = node.get('MID', None)
                if mid_raw in (None, ''):
                    mid_raw = node.get('MatName', None)
                if mid_raw not in (None, ''):
                    try:
                        mid = int(float(mid_raw))
                        if 0 <= mid < material_lookup_table.shape[0]:
                            material_props[i] = material_lookup_table[mid]
                        else:
                            material_props[i] = 0.0
                    except Exception:
                        material_props[i] = 0.0
                else:
                    material_props[i] = 0.0

        # ---------- hic_point ----------
        hic_point = point_data['hic_point']
        sample_info = hic_point.get('sample_info', None)
        if sample_info in (None, ''):
            sample_info = str(sample_id)

        # HIC 值：新字段 HIC，兼容旧字段 hic_value
        hic_value_raw = hic_point.get('HIC', None)
        if hic_value_raw in (None, ''):
            hic_value_raw = hic_point.get('hic_value', None)
        if hic_value_raw in (None, ''):
            print(f"警告: 样本 {sample_id} 缺少 HIC 值，跳过")
            return  # 使用 return 而不是 continue

        hic_value = _safe_float(hic_value_raw, None)
        if hic_value is None:
            print(f"警告: 样本 {sample_id} HIC 无法解析，跳过")
            return  # 使用 return 而不是 continue
        
        # 排除HIC值为0的样本
        if hic_value == 0:
            excluded_samples.append(sample_id)
            return  # 使用 return 而不是 continue

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
        # 清理 NaN
        nodes = np.nan_to_num(nodes)
        thickness = np.nan_to_num(thickness)
        material_props = np.nan_to_num(material_props)

        self.datapoints.append({
            'point_set': nodes,
            'hic_point': {'x': hp_x, 'y': hp_y, 'z': hp_z},
            'category': category,
            'hic_value': float(hic_value),
            'thickness': thickness,
            'material_props': material_props,
            'age_group': age_group,
            'sample_info': sample_info
        })

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