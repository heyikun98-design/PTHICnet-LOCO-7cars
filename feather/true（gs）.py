import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.colors import LinearSegmentedColormap

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

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

def get_color_and_score(value):
    """根据HIC值返回对应的颜色"""
    if value < 650:
        return 'green'
    elif 650 <= value < 1000:
        return 'yellow'
    elif 1000 <= value < 1350:
        return 'orange'
    elif 1350 <= value < 1700:
        return 'brown'
    else:
        return 'red'

def process_multiple_excel_data(file_paths):
    # 创建统计矩阵
    count_matrix = np.zeros((16, 21))  # 记录每个位置的数据点数量
    sum_matrix = np.zeros((16, 21))    # 记录每个位置的HIC真实值总和
    squared_sum_matrix = np.zeros((16, 21))  # 记录每个位置的HIC真实值平方和
    
    # 记录实际的x坐标范围
    min_x, max_x = float('inf'), float('-inf')
    
    # 遍历所有Excel文件
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            # 解析坐标字符串 (例如: "A.12.-5")
            coord_str = row['Sample Info'].strip()  # 确保去除空白字符
            # 先进行格式转换
            converted_coord_str = convert_sample_info_format(coord_str)
            parts = converted_coord_str.split('.')
            if len(parts) == 3:  # 确保坐标格式正确
                y = int(float(parts[1]))  # 获取y坐标 (12)
                x = int(float(parts[2]))  # 获取x坐标 (-5)
                x_idx = 10 - x  # 转换x坐标到矩阵索引
                
                # 更新x坐标范围
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                
                value = row['True HIC Value']  # 修改为使用真实值
                if pd.notna(value):  # 确保值不是NaN
                    # 更新统计数据
                    count_matrix[y][x_idx] += 1
                    sum_matrix[y][x_idx] += value
                    squared_sum_matrix[y][x_idx] += value * value
    
    # 计算均值矩阵
    mean_matrix = np.zeros((16, 21))
    # 计算方差矩阵
    variance_matrix = np.zeros((16, 21))
    
    # 计算均值和标准差
    for i in range(16):
        for j in range(21):
            if count_matrix[i][j] > 0:
                mean = sum_matrix[i][j] / count_matrix[i][j]
                mean_matrix[i][j] = mean
                
                # 重新计算标准差
                variance = 0
                for file_path in file_paths:
                    df = pd.read_excel(file_path)
                    for _, row in df.iterrows():
                        coord_str = row['Sample Info'].strip()
                        # 先进行格式转换
                        converted_coord_str = convert_sample_info_format(coord_str)
                        parts = converted_coord_str.split('.')
                        if len(parts) == 3:
                            y = int(float(parts[1]))
                            x = int(float(parts[2]))
                            x_idx = 10 - x
                            
                            if y == i and x_idx == j:
                                value = row['True HIC Value']  # 修改为使用真实值
                                if pd.notna(value):
                                    variance += (value - mean) ** 2
                
                variance_matrix[i][j] = np.sqrt(variance / count_matrix[i][j])
    
    # 返回矩阵和x坐标范围
    x_range = (min_x, max_x) if min_x != float('inf') else (-10, 10)
    return count_matrix, mean_matrix, variance_matrix, x_range

def plot_single_heatmap(matrix, title, filename, x_range=(-10, 10), is_accuracy=False, is_count=False, is_variance=False):
    # 创建新的图形，调整比例更接近参考图
    plt.figure(figsize=(12, 10))
    
    # 获取矩阵大小
    y_size, x_size = matrix.shape
    
    # 根据x_range计算实际需要显示的列范围
    min_x, max_x = x_range
    x_start_idx = 10 - max_x  # 最大x对应最小索引
    x_end_idx = 10 - min_x    # 最小x对应最大索引
    
    # 创建底色为白色的背景
    plt.fill_between([-0.5, x_size-0.5], -0.5, y_size-0.5, color='white')
    
    # 根据不同类型的图选择不同的颜色映射方式
    if is_count:
        # 数量统计使用从浅绿到深红的渐变
        colors = ['#00FF00', '#40FF00', '#80FF00', '#C0FF00', 
                 '#FFFF00', '#FFC000', '#FF8000', '#FF4000', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('custom_green_to_red', colors)
        use_custom_colors = False
    elif is_variance:
        # 标准差统计使用从浅绿到深红的渐变
        colors = ['#00FF00', '#FFFF00', '#FFA500', '#FF0000']  # 绿-黄-橙-红
        cmap = LinearSegmentedColormap.from_list('custom_std_colors', colors)
        use_custom_colors = False
    else:
        # 均值图使用固定的颜色区间
        use_custom_colors = True
    
    # 获取非零值的最大和最小值用于颜色映射
    non_zero = matrix[matrix != 0]
    if len(non_zero) > 0:
        if is_variance:
            # 对于标准差图，固定最大最小值范围，使颜色映射一致
            vmin, vmax = 0, 5000  # 设置一个合适的范围
        else:
            vmin, vmax = non_zero.min(), non_zero.max()
    else:
        vmin, vmax = 0, 1
    
    # 绘制每个单元格
    for i in range(y_size):
        for j in range(x_size):
            # 只绘制在x范围内的列
            if j < x_start_idx or j > x_end_idx:
                continue
                
            value = matrix[i][j]
            if value != 0:  # 只处理有值的单元格
                if use_custom_colors:
                    # 均值图使用固定的颜色区间
                    if value < 650:
                        color = 'green'
                    elif 650 <= value < 1000:
                        color = 'yellow'
                    elif 1000 <= value < 1350:
                        color = 'orange'
                    elif 1350 <= value < 1700:
                        color = 'brown'
                    else:
                        color = 'red'
                else:
                    # 其他图使用渐变色
                    norm_value = (value - vmin) / (vmax - vmin)
                    color = cmap(norm_value)
                
                # 绘制矩形
                plt.fill([j-0.5, j+0.5, j+0.5, j-0.5], 
                        [i-0.5, i-0.5, i+0.5, i+0.5],
                        color=color)
                
                # 添加数值文本，增大字体
                if is_count:
                    text = f'{value:.0f}'
                    font_size = 12
                elif is_variance:
                    text = f'{value:.1f}'  # 保持一位小数
                    font_size = 11
                else:
                    text = f'{value:.0f}'
                    font_size = 12
                plt.text(j, i, text, ha='center', va='center',
                        color='black', fontsize=font_size, fontweight='bold')
    
    # 先设置坐标轴（只显示有数据的列）
    plt.xlim(x_start_idx-0.5, x_end_idx+0.5)
    plt.ylim(-0.5, y_size-0.5)
    
    # 绘制网格线
    for i in range(y_size + 1):
        plt.axhline(y=i-0.5, color='black', linewidth=0.5)
    for j in range(x_start_idx, x_end_idx + 2):
        plt.axvline(x=j-0.5, color='black', linewidth=0.5)
    
    # 修改坐标轴标签
    x_ticks = range(x_start_idx, x_end_idx + 1)
    x_labels = range(max_x, min_x - 1, -1)
    plt.xticks(x_ticks, x_labels, fontsize=12)
    
    y_ticks = range(y_size)
    plt.yticks(y_ticks, range(y_size), fontsize=12)
    
    # 不显示标题
    # plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    # 添加边框
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_statistical_heatmap(count_matrix, mean_matrix, variance_matrix, x_range, save_dir):
    # 绘制数据点数量分布图
    plot_single_heatmap(count_matrix, 'HIC真实值数量分布图', 
                       os.path.join(save_dir, 'count_heatmap.png'),
                       x_range=x_range,
                       is_count=True)
    
    # 绘制均值分布图
    plot_single_heatmap(mean_matrix, 'HIC True Mean Distrubution', 
                       os.path.join(save_dir, 'true_mean_heatmap.png'),
                       x_range=x_range)
    
    # 绘制标准差分布图
    plot_single_heatmap(variance_matrix, 'HIC真实值标准差分布图',
                       os.path.join(save_dir, 'std_heatmap.png'),
                       x_range=x_range,
                       is_variance=True)

def main():
    # 获取所有Excel文件路径
    folder_path = input("请输入包含Excel文件的文件夹路径: ")
    excel_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.endswith('.xlsx') or f.endswith('.xls')]
    # 1. 创建 acc1 子目录（如果没有就创建）
    save_dir = os.path.join(folder_path, "true")
    os.makedirs(save_dir, exist_ok=True)  # 自动递归创建，无则创建，有则忽略
    
    try:
        # 处理多个Excel文件的数据
        count_matrix, mean_matrix, variance_matrix, x_range = process_multiple_excel_data(excel_files)
        
        # 绘制统计图表
        plot_statistical_heatmap(count_matrix, mean_matrix, variance_matrix, x_range, save_dir)
        
        print(f"统计图表已保存至目录: {save_dir}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()