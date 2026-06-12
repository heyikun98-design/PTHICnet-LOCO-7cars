import os
import sys
# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 找到 smxtraining 所在的目录（向上找直到包含 smxtraining 的文件夹）
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
# 把 project 根目录加入 sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import argparse
import json
import pandas as pd
import numpy as np
import torch
import re
import importlib
from tqdm import tqdm
import os
from torch.utils.data import ConcatDataset
from data_utils.HICLoader_feather import HICDataLoader

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_reg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # 新增上下部分数据路径参数（支持 JSON 和 Feather 格式）
    parser.add_argument('--up_test_data', type=str, default='data/test/up_X70.json', help='path to upper part test data (supports .json and .feather)')
    parser.add_argument('--down_test_data', type=str, default='data/test/down_X70.json', help='path to lower part test data (supports .json and .feather)')
    parser.add_argument('--ablation_mode', type=str, default='baseline', choices=['baseline', 'early_fusion_clean', 'pt_hicnet'])
    parser.add_argument('--use_early_fusion', action='store_true', default=False)
    parser.add_argument('--normalize_thickness', action='store_true', default=False)
    return parser.parse_args()

def convert_sample_info_format(coord_str):
    """
    将sample_info从下划线格式转换为点格式
    支持格式：
    - A_1_1 -> A.1.1
    - C_1__1 -> C.1.-1 (双下划线代表负数)
    - A.1.1 -> A.1.1 (已经是点格式，直接返回)
    """
    coord_str = coord_str.strip()
    
    # 如果已经是点格式，直接返回
    if '.' in coord_str and coord_str.count('.') == 2:
        return coord_str
    
    # 如果是下划线格式，进行转换
    if '_' in coord_str:
        # 使用正则表达式提取样本点位，适配下划线格式
        match = re.search(r'([AC])_(\d+)(?:_(\d+)|__(\d+))?', coord_str)
        if match:
            prefix = match.group(1)  # A 或 C
            y_coord = match.group(2)  # y坐标
            # 处理双下划线情况（表示负数）
            x_coord = f"-{match.group(4)}" if match.group(4) else (match.group(3) or '0')
            return f"{prefix}.{y_coord}.{x_coord}"
    
    # 如果都不匹配，返回原始值
    return coord_str

def calculate_accuracy(pred_value, true_value):
    """
    计算预测值和真实值之间的误差相似度
    参考accuracy2.py中的方法
    
    Args:
        pred_value: 预测值
        true_value: 真实值
    
    Returns:
        float: 相似度（0-1之间），越接近1表示预测越准确
    """
    if pd.isna(pred_value) or pd.isna(true_value):
        return None
    if pred_value == 0 and true_value == 0:
        return 1.0
    if pred_value == 0 or true_value == 0:
        return 0.0
    return min(pred_value, true_value) / max(pred_value, true_value)

def calculate_overall_accuracy(predictions_data):
    """
    计算预测数据的整体平均精度
    
    Args:
        predictions_data: 预测数据列表
    
    Returns:
        float: 平均精度
    """
    accuracies = []
    for item in predictions_data:
        accuracy = calculate_accuracy(item['Predicted HIC Value'], item['True HIC Value'])
        if accuracy is not None:
            accuracies.append(accuracy)
    
    if accuracies:
        return np.mean(accuracies)
    else:
        return 0.0

def parse_y_coordinate(sample_info):
    """
    从sample_info中解析y坐标
    
    Args:
        sample_info: 样本信息字符串
    
    Returns:
        int: y坐标值，解析失败返回None
    """
    try:
        # 先进行格式转换
        converted_coord_str = convert_sample_info_format(sample_info)
        parts = converted_coord_str.split('.')
        if len(parts) >= 2:
            return int(float(parts[1]))
        return None
    except:
        return None

def find_boundary_point(up_data_path, down_data_path):
    """
    根据上下部分数据找到分界点（支持 JSON 和 Feather 格式）
    
    Args:
        up_data_path: 上部分数据路径（支持 .json 和 .feather）
        down_data_path: 下部分数据路径（支持 .json 和 .feather）
    
    Returns:
        int: 分界点y坐标值
    """
    up_y_coords = []
    down_y_coords = []
    
    # 读取上部分数据的y坐标
    if up_data_path and os.path.exists(up_data_path):
        file_ext = os.path.splitext(up_data_path)[1].lower()
        if file_ext == '.feather':
            # 读取 Feather 格式
            df = pd.read_feather(up_data_path)
            for idx, row in df.iterrows():
                point_data = row['data']
                if 'hic_point' in point_data and 'sample_info' in point_data['hic_point']:
                    sample_info = point_data['hic_point']['sample_info']
                    y_coord = parse_y_coordinate(sample_info)
                    if y_coord is not None:
                        up_y_coords.append(y_coord)
        else:
            # 读取 JSON 格式
            with open(up_data_path, 'r', encoding='utf-8') as f:
                up_data = json.load(f)
            for key, value in up_data.items():
                sample_info = value["hic_point"]["sample_info"]
                y_coord = parse_y_coordinate(sample_info)
                if y_coord is not None:
                    up_y_coords.append(y_coord)
    
    # 读取下部分数据的y坐标
    if down_data_path and os.path.exists(down_data_path):
        file_ext = os.path.splitext(down_data_path)[1].lower()
        if file_ext == '.feather':
            # 读取 Feather 格式
            df = pd.read_feather(down_data_path)
            for idx, row in df.iterrows():
                point_data = row['data']
                if 'hic_point' in point_data and 'sample_info' in point_data['hic_point']:
                    sample_info = point_data['hic_point']['sample_info']
                    y_coord = parse_y_coordinate(sample_info)
                    if y_coord is not None:
                        down_y_coords.append(y_coord)
        else:
            # 读取 JSON 格式
            with open(down_data_path, 'r', encoding='utf-8') as f:
                down_data = json.load(f)
            for key, value in down_data.items():
                sample_info = value["hic_point"]["sample_info"]
                y_coord = parse_y_coordinate(sample_info)
                if y_coord is not None:
                    down_y_coords.append(y_coord)
    
    if up_y_coords and down_y_coords:
        # 分界点是上部分最小y坐标和下部分最大y坐标之间
        min_up_y = min(up_y_coords)
        max_down_y = max(down_y_coords)
        boundary = (min_up_y + max_down_y) // 2
        print(f"检测到分界点: y={boundary} (上部分y范围: {min(up_y_coords)}-{max(up_y_coords)}, 下部分y范围: {min(down_y_coords)}-{max(down_y_coords)})")
        return boundary
    else:
        print("警告: 无法确定分界点，使用默认值11")
        return 11

def pick_test_files(test_dir: str) -> list:
    """从测试目录查找所有测试文件（支持 JSON 或 Feather）。
    支持两种结构：
    1. 直接包含文件：/test/*.feather
    2. 按车辆组织的目录：/test/ASE/batch_0.feather
    """
    if not os.path.isdir(test_dir):
        # 如果是文件，直接返回
        if os.path.isfile(test_dir) and test_dir.endswith(('.json', '.feather')):
            return [test_dir]
        raise FileNotFoundError(f"Test data path not found: {test_dir}")
    
    test_files = []
    
    # 首先检查是否直接包含文件
    direct_files = [os.path.join(test_dir, f) 
                    for f in os.listdir(test_dir) 
                    if f.endswith(('.json', '.feather')) and os.path.isfile(os.path.join(test_dir, f))]
    
    if direct_files:
        # 方式1：直接包含文件
        test_files = direct_files
    else:
        # 方式2：按车辆组织的目录结构
        # 遍历所有子目录，查找每个子目录下的所有 feather/json 文件
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                # 查找该子目录下的所有 feather/json 文件
                subdir_files = [os.path.join(subdir_path, f) 
                               for f in os.listdir(subdir_path) 
                               if f.endswith(('.json', '.feather')) and os.path.isfile(os.path.join(subdir_path, f))]
                test_files.extend(subdir_files)
    
    if not test_files:
        raise FileNotFoundError(f"No .json or .feather found under {test_dir}")
    
    return test_files

def separate_predictions_by_boundary(predictions_data, boundary):
    """
    根据分界点将预测数据分为上下两部分
    
    Args:
        predictions_data: 预测数据列表
        boundary: 分界点y坐标值
    
    Returns:
        tuple: (上部分数据, 下部分数据)
    """
    up_predictions = []
    down_predictions = []
    
    for item in predictions_data:
        sample_info = item['Sample Info']
        y_coord = parse_y_coordinate(sample_info)
        
        if y_coord is not None:
            if y_coord > boundary:
                up_predictions.append(item)
            else:
                down_predictions.append(item)
        else:
            # 如果无法解析坐标，默认放入下部分
            down_predictions.append(item)
    
    return up_predictions, down_predictions

def test_model(test_data_dir, model_path, args):
    device = torch.device("cpu")
    if args.ablation_mode == "pt_hicnet":
        from models.pt_hicnet import PT_HICnet
        model = PT_HICnet(in_channels=22 if args.use_normals else 19, use_normals=args.use_normals, film_mode="global").to(device)
        args.use_early_fusion = True
    else:
        model_name = "pointnet2_reg_ablation" if args.ablation_mode == "early_fusion_clean" else "pointnet2_reg_att_props"
        model = importlib.import_module(model_name).get_model(
            normal_channel=args.use_normals,
            num_point=args.num_point
        ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 支持目录或文件路径
    if os.path.isdir(test_data_dir):
        # 如果是目录，查找所有测试文件
        test_files = pick_test_files(test_data_dir)
        print(f"找到 {len(test_files)} 个测试文件")
    else:
        # 如果是文件，直接使用
        test_files = [test_data_dir]
    
    # 获取基础名称用于输出文件名
    if os.path.isdir(test_data_dir):
        base_name = os.path.basename(test_data_dir)
    else:
        base_name = os.path.splitext(os.path.basename(test_data_dir))[0]
    
    # 收集所有文件的 sample_info
    all_sample_info_list = []
    all_test_datasets = []
    
    for test_file in test_files:
        file_ext = os.path.splitext(test_file)[1].lower()
        
        if file_ext == '.feather':
            # 读取 Feather 格式，按顺序提取 sample_info
            df = pd.read_feather(test_file)
            for idx, row in df.iterrows():
                point_data = row['data']
                if 'hic_point' in point_data and 'sample_info' in point_data['hic_point']:
                    all_sample_info_list.append(point_data['hic_point']['sample_info'])
                else:
                    all_sample_info_list.append(f"unknown_{idx}")
        else:
            # 读取 JSON 格式，按顺序提取 sample_info
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            for key in sorted(test_data.keys()):  # 按 key 排序以保持一致性
                if 'hic_point' in test_data[key] and 'sample_info' in test_data[key]['hic_point']:
                    all_sample_info_list.append(test_data[key]['hic_point']['sample_info'])
                else:
                    all_sample_info_list.append(f"unknown_{key}")
        
        # 创建数据集
        all_test_datasets.append(
            HICDataLoader(
                root=test_file,
                args=args,
                early_fusion=args.use_early_fusion,
                normalize_thickness=args.normalize_thickness,
            )
        )
    
    # 合并所有数据集
    test_dataset = ConcatDataset(all_test_datasets)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=15, shuffle=False)

    predictions_data = []
    sample_index = 0  # 用于追踪当前样本的索引
    for batch in loader:
        if args.use_early_fusion:
            points, hic_point, category, age_group, target = batch
            thickness, material_props = None, None
        else:
            points, hic_point, category, thickness, material_props, age_group, target = batch
        points = points.to(device)
        hic_point = hic_point.to(device)
        # target = target.to(device)
        category = category.to(device)
        if thickness is not None:
            thickness = thickness.to(device)
        # part_name = part_name.to(device)
        if material_props is not None:
            material_props = material_props.to(device)
        age_group = age_group.to(device)

        points = points.transpose(2, 1)
        if args.use_early_fusion or args.ablation_mode == "pt_hicnet":
            pred, _ = model(points, hic_point, category, age_group)
        else:
            pred, _ = model(points, hic_point, category, thickness, material_props, age_group)
        
        #y_pred = pred.detach().cpu().numpy()
        y_pred = pred
        # y_true = target.detach().cpu().numpy()
        
        # 提取样本点的坐标和预测值
        for i in range(len(y_pred)):
            hic_x = hic_point[i, 0].item()
            hic_y = hic_point[i, 1].item()
            hic_z = hic_point[i, 2].item()
            
            # 获取对应的sample_info（从预加载的列表中获取）
            if sample_index < len(all_sample_info_list):
                sample_info = all_sample_info_list[sample_index]
            else:
                sample_info = f"unknown_{sample_index}"
            
            predictions_data.append({
                'Sample Info': sample_info,
                'True HIC Value': target[i].item(),
                'Predicted HIC Value': y_pred[i].item()
            })
            sample_index += 1  # 更新索引

    # 保存预测结果，文件名与测试数据对应
    df = pd.DataFrame(predictions_data)
    # 修改这里，添加'Sample Info'到columns_order中
    columns_order = ['Sample Info', 'True HIC Value', 'Predicted HIC Value']
    df = df[columns_order]
    output_file = f'predictions_{base_name}_2.xlsx'
    df.to_excel(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")
    
    # 计算整体平均精度
    overall_accuracy = calculate_overall_accuracy(predictions_data)
    print(f"整体平均精度: {overall_accuracy:.3f}")
    
    return predictions_data, output_file

if __name__ == "__main__":
    # 测试数据路径列表（支持 JSON 和 Feather 格式）
    test_data_dirs = [
        r"/home/sl/project/alldata/feather/test/13_X70"
        # 添加更多测试数据路径（支持 .json 和 .feather 格式）...
    ]
    
    # 训练模型路径
    model_path = r"experiments/X70/regression/latest/checkpoints/best_model.pth"
    
    args = parse_args()
    
    # 对每个测试数据进行预测
    for test_data_dir in test_data_dirs:
        print(f"\n处理测试数据: {test_data_dir}")
        predictions_data, output_file = test_model(test_data_dir, model_path, args)
        
        # 如果提供了上下部分数据路径，进行上下分离精度计算
        if args.up_test_data and args.down_test_data:
            print(f"\n" + "="*60)
            print(f"进行上下部分精度分析...")
            print(f"="*60)
            
            # 找到分界点
            boundary = find_boundary_point(args.up_test_data, args.down_test_data)
            
            # 根据分界点分离预测数据
            up_predictions, down_predictions = separate_predictions_by_boundary(predictions_data, boundary)
            
            # 计算各部分精度
            print(f"\n精度统计结果:")
            print(f"-" * 40)
            
            # 上部分精度
            if up_predictions:
                up_accuracy = calculate_overall_accuracy(up_predictions)
                print(f"上部分 (y > {boundary}) 精度: {up_accuracy:.3f} (样本数: {len(up_predictions)})")
            else:
                print(f"上部分 (y > {boundary}) 精度: 无数据")
            
            # 下部分精度
            if down_predictions:
                down_accuracy = calculate_overall_accuracy(down_predictions)
                print(f"下部分 (y <= {boundary}) 精度: {down_accuracy:.3f} (样本数: {len(down_predictions)})")
            else:
                print(f"下部分 (y <= {boundary}) 精度: 无数据")
            
            # 整体精度
            overall_accuracy = calculate_overall_accuracy(predictions_data)
            print(f"整体精度: {overall_accuracy:.3f} (总样本数: {len(predictions_data)})")
            
            print(f"\n" + "="*60)
        else:
            print(f"\n提示: 如需进行上下部分精度分析，请提供 --up_test_data 和 --down_test_data 参数")