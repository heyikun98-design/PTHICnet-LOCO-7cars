import os
import json
import numpy as np
import pandas as pd

# 文件夹路径
data_dir = '/home/sl/project/alldata/feather/通风盖板up/train'

# 获取所有json和feather文件，递归
json_files = [
    os.path.join(root, f)
    for root, _, fs in os.walk(data_dir)
    for f in fs if f.endswith('.json')
    ]

feather_files = [
    os.path.join(root, f)
    for root, _, fs in os.walk(data_dir)
    for f in fs if f.endswith('.feather')
    ]

files = json_files + feather_files
print(f"开始处理{len(json_files)}个JSON文件和{len(feather_files)}个Feather文件，共{len(files)}个文件")

# 收集所有样本的HIC值
all_HICs = []
excluded_samples = []

for file in files:
    # file已经是完整路径，不需要再拼接data_dir
    file_path = file
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"处理文件: {file_path} (格式: {file_ext})")
    
    try:
        if file_ext == '.feather':
            # 处理 Feather 格式
            df = pd.read_feather(file_path)
            
            # 直接从 DataFrame 处理
            for idx, row in df.iterrows():
                file_id = str(row['file_id'])  # 转换为字符串，保持与原JSON格式一致
                point_data = row['data']  # 直接使用 row['data']
                
                # 确保 nearby_nodes 是列表格式（feather 中可能是 numpy array）
                if 'nearby_nodes' in point_data:
                    nearby_nodes = point_data['nearby_nodes']
                    if isinstance(nearby_nodes, np.ndarray):
                        # 将 numpy array 转换为列表（每个元素是字典）
                        point_data['nearby_nodes'] = [dict(node) for node in nearby_nodes]
                    elif not isinstance(nearby_nodes, list):
                        # 如果不是列表也不是数组，尝试转换
                        point_data['nearby_nodes'] = list(nearby_nodes) if nearby_nodes else []
                
                # 提取 HIC 值
                if isinstance(point_data, dict) and 'hic_point' in point_data:
                    hic_point_data = point_data['hic_point']
                    
                    # 检查hic_point中是否有HIC或hic_value字段
                    if isinstance(hic_point_data, dict):
                        # 优先使用 HIC 字段，兼容 hic_value 字段
                        hic_value_raw = hic_point_data.get('HIC', None)
                        if hic_value_raw in (None, ''):
                            hic_value_raw = hic_point_data.get('hic_value', None)
                        
                        if hic_value_raw is not None and hic_value_raw != '':
                            try:
                                HIC = float(hic_value_raw)
                                
                                # 排除HIC值为0的样本
                                if HIC == 0:
                                    excluded_samples.append((file_path, file_id))
                                    continue
                                
                                all_HICs.append(HIC)
                                print(f"样本 {file_id} 的HIC值: {HIC}")
                                
                            except (ValueError, TypeError) as e:
                                print(f"警告: 样本 {file_id} HIC 值无法解析: {hic_value_raw}, 错误: {e}")
                                continue
            
            # 释放 DataFrame 以释放内存
            del df
            import gc
            gc.collect()
            
        else:
            # 处理 JSON 格式（保持原有逻辑）
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 针对你的JSON格式特别处理
                if isinstance(data, dict):
                    # 遍历第一层字典(ID -> 数据)
                    for sample_id, sample_data in data.items():
                        # 检查是否有hic_point字段
                        if isinstance(sample_data, dict) and 'hic_point' in sample_data:
                            hic_point_data = sample_data['hic_point']
                            
                            # 检查hic_point中是否有HIC或hic_value字段
                            if isinstance(hic_point_data, dict):
                                # 优先使用 HIC 字段，兼容 hic_value 字段
                                hic_value_raw = hic_point_data.get('HIC', None)
                                if hic_value_raw in (None, ''):
                                    hic_value_raw = hic_point_data.get('hic_value', None)
                                
                                if hic_value_raw is not None and hic_value_raw != '':
                                    try:
                                        HIC = float(hic_value_raw)
                                        
                                        # 排除HIC值为0的样本
                                        if HIC == 0:
                                            excluded_samples.append((file_path, sample_id))
                                            continue
                                        
                                        all_HICs.append(HIC)
                                        print(f"样本 {sample_id} 的HIC值: {HIC}")
                                    except (ValueError, TypeError) as e:
                                        print(f"警告: 样本 {sample_id} HIC 值无法解析: {hic_value_raw}, 错误: {e}")
                                        continue
            
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()

print(f"收集到 {len(all_HICs)} 个HIC值")
if excluded_samples:
    print(f"已排除 {len(excluded_samples)} 个HIC为0的样本点：")
    for file, sample_id in excluded_samples:
        print(f"  ——文件：{file} ， 样本ID：{sample_id}")

# 如果找到了值，保存为CSV
if all_HICs:
    y_df = pd.DataFrame({'y': all_HICs})
    output_path = '/home/sl/project/attn_props_sjzq/y_train/y_train(7up).csv'
    y_df.to_csv(output_path, index=False)
    print(f'已生成y_train.csv文件，包含{len(all_HICs)}个HIC值')
    
    # 打印统计信息
    print(f"HIC值统计: 最小值={min(all_HICs)}, 最大值={max(all_HICs)}, 平均值={sum(all_HICs)/len(all_HICs)}")
else:
    print("未找到任何HIC值，请检查JSON结构")
