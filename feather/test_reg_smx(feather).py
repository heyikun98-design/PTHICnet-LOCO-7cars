import os
import sys
# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 找到 smxtraining 所在的目录（向上找直到包含 smxtraining 的文件夹）
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# 把 project 根目录加入 sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
import pandas as pd
import numpy as np
import torch
import re
import importlib
from tqdm import tqdm
from data_utils.HICLoader_feather import HICDataLoader
MODEL_DIR = os.path.join(CURRENT_DIR, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size in training')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # 新增两个模型路径参数
    parser.add_argument('--model_path_up', type=str, default='experiments/up/checkpoints/best_model.pth', help='model path for upper part (y>11)')
    parser.add_argument('--model_path_down', type=str, default='experiments/down/checkpoints/best_model.pth', help='model path for lower part (y<=11)')
    parser.add_argument('--ablation_mode', type=str, default='baseline', choices=['baseline', 'early_fusion_clean', 'pt_hicnet'])
    parser.add_argument('--use_early_fusion', action='store_true', default=False)
    parser.add_argument('--normalize_thickness', action='store_true', default=False)
    return parser.parse_args()

# 解析y值的函数，参考accuracy1.py
def parse_y(sample_info):
    try:
        return int(float(str(sample_info).split('.')[1]))
    except:
        return None

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

def calculate_average_accuracy(predictions_data):
    """
    计算预测数据的平均精度
    
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

def load_model(model_path, args, device):
    if args.ablation_mode == "pt_hicnet":
        from models.pt_hicnet import PT_HICnet
        model = PT_HICnet(
            in_channels=22 if args.use_normals else 19,
            use_normals=args.use_normals,
            film_mode="global",
        ).to(device)
        args.use_early_fusion = True
    else:
        model_name = "pointnet2_reg_ablation" if args.ablation_mode == "early_fusion_clean" else "pointnet2_reg_att_props"
        model = importlib.import_module(model_name).get_model(
            normal_channel=args.use_normals,
            num_point=args.num_point
        ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def test_model(test_data_path, model_path, part_type, args):
    """
    测试模型并返回预测数据（支持 JSON 和 Feather 格式）
    
    Args:
        test_data_path: 测试数据路径（支持 .json 和 .feather）
        model_path: 模型路径
        part_type: 部位类型 ('UP' 或 'DOWN')
        args: 参数
    
    Returns:
        list: 预测数据列表
    """
    device = torch.device("cpu")
    test_file = os.path.join(test_data_path)
    
    # 支持 JSON 和 Feather 格式
    file_ext = os.path.splitext(test_file)[1].lower()
    sample_info_list = []  # 按顺序存储所有 sample_info
    
    if file_ext == '.feather':
        # 读取 Feather 格式，按顺序提取 sample_info
        df = pd.read_feather(test_file)
        for idx, row in df.iterrows():
            point_data = row['data']
            if 'hic_point' in point_data and 'sample_info' in point_data['hic_point']:
                sample_info_list.append(point_data['hic_point']['sample_info'])
            else:
                sample_info_list.append(f"unknown_{idx}")
    else:
        # 读取 JSON 格式，按顺序提取 sample_info
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        for key in sorted(test_data.keys()):  # 按 key 排序以保持一致性
            if 'hic_point' in test_data[key] and 'sample_info' in test_data[key]['hic_point']:
                sample_info_list.append(test_data[key]['hic_point']['sample_info'])
            else:
                sample_info_list.append(f"unknown_{key}")

    # 加载模型
    model = load_model(model_path, args, device)
    # 构造数据集
    test_dataset = HICDataLoader(
        root=test_data_path,
        args=args,
        early_fusion=args.use_early_fusion,
        normalize_thickness=args.normalize_thickness,
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=15, shuffle=False)
    predictions_data = []
    sample_index = 0
    
    print(f"开始处理 {part_type} 部分数据...")
    
    for batch in loader:
        if args.use_early_fusion:
            points, hic_point, category, age_group, target = batch
            thickness, material_props = None, None
        else:
            points, hic_point, category, thickness, material_props, age_group, target = batch
        points = points.to(device)
        hic_point = hic_point.to(device)
        category = category.to(device)
        if thickness is not None:
            thickness = thickness.to(device)
        if material_props is not None:
            material_props = material_props.to(device)
        age_group = age_group.to(device)
        points = points.transpose(2, 1)
        if args.use_early_fusion or args.ablation_mode == "pt_hicnet":
            pred, _ = model(points, hic_point, category, age_group)
        else:
            pred, _ = model(points, hic_point, category, thickness, material_props, age_group)
        y_pred = pred
        for i in range(len(y_pred)):
            # 从预加载的列表中获取 sample_info
            if sample_index < len(sample_info_list):
                sample_info = sample_info_list[sample_index]
            else:
                sample_info = f"unknown_{sample_index}"
            
            predictions_data.append({
                'Part Type': part_type,  # 添加部位标识
                'Sample Info': sample_info,
                'True HIC Value': target[i].item(),
                'Predicted HIC Value': y_pred[i].item()
            })
            sample_index += 1
    
    print(f"{part_type} 部分处理完成，共 {len(predictions_data)} 个样本")
    return predictions_data

if __name__ == "__main__":
    # 直接指定 up 和 down 测试集路径（支持 .json 和 .feather 格式）
    up_test_data = "data/test/up_R11MCE.json"
    down_test_data = "data/test/down_R11MEC.json"
    args = parse_args()
    
    # 收集所有预测数据
    all_predictions = []
    
    print(f"\nProcessing UP test data: {up_test_data}")
    up_predictions = test_model(up_test_data, args.model_path_up, 'UP', args)
    all_predictions.extend(up_predictions)
    
    print(f"\nProcessing DOWN test data: {down_test_data}")
    down_predictions = test_model(down_test_data, args.model_path_down, 'DOWN', args)
    all_predictions.extend(down_predictions)
    
    # 计算各部分的平均精度
    print(f"\n" + "="*60)
    print(f"精度评估结果")
    print(f"="*60)
    
    # UP 部分精度
    up_accuracy = calculate_average_accuracy(up_predictions)
    print(f"\nUP 部分预测精度 ({len(up_predictions)} 个样本):")
    print(f"  平均精度: {up_accuracy:.3f}")
    
    # DOWN 部分精度
    down_accuracy = calculate_average_accuracy(down_predictions)
    print(f"\nDOWN 部分预测精度 ({len(down_predictions)} 个样本):")
    print(f"  平均精度: {down_accuracy:.3f}")
    
    # 整体精度
    overall_accuracy = calculate_average_accuracy(all_predictions)
    print(f"\n整体预测精度 ({len(all_predictions)} 个样本):")
    print(f"  平均精度: {overall_accuracy:.3f}")
    
    print(f"\n" + "="*60)
    
    # 合并所有预测结果并保存到单个xlsx文件
    print(f"\n合并预测结果...")
    df = pd.DataFrame(all_predictions)
    columns_order = ['Part Type', 'Sample Info', 'True HIC Value', 'Predicted HIC Value']
    df = df[columns_order]
    
    # 生成输出文件名
    output_file = 'predictions_combined_R11MCE.xlsx'
    df.to_excel(output_file, index=False)
    
    print(f"\n所有预测结果已保存到 {output_file}")
    print(f"总共处理了 {len(all_predictions)} 个样本")
    print(f"  - UP 部分: {len(up_predictions)} 个样本")
    print(f"  - DOWN 部分: {len(down_predictions)} 个样本")