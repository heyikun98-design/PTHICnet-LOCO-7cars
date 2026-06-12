import re
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def parse_training_log(log_content):
    """
    解析训练日志，提取epoch和对应的MSE值
    """
    epochs = []
    mse_values = []
    
    # 使用正则表达式匹配 "Current Instance MSE: 数值" 的模式
    pattern = r'Current Instance MSE:\s*([\d.]+)'
    matches = re.findall(pattern, log_content)
    
    # 为每个MSE值分配epoch编号（从1开始）
    for i, mse_str in enumerate(matches):
        epochs.append(i + 1)
        mse_values.append(float(mse_str))
    
    return epochs, mse_values

def plot_loss_curve(epochs, mse_values, log_file_path):
    """
    绘制并保存损失曲线
    """
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, mse_values, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)
    plt.title(f'Training Loss Curve\n({os.path.basename(log_file_path)})', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴为科学计数法显示
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 添加最小值标注
    min_idx = np.argmin(mse_values)
    min_epoch = epochs[min_idx]
    min_mse = mse_values[min_idx]
    plt.annotate(f'Min MSE: {min_mse:.2f}\nEpoch: {min_epoch}', 
                xy=(min_epoch, min_mse), 
                xytext=(min_epoch + len(epochs)//10, min_mse + max(mse_values)//10),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # 获取文件夹路径并生成保存文件名
    folder_path = os.path.dirname(log_file_path)
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(folder_path, f"(acc){base_name}_loss_curve_{timestamp}.png")
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print(f"\n训练统计信息:")
    print(f"日志文件: {log_file_path}")
    print(f"总epoch数: {len(epochs)}")
    print(f"初始MSE: {mse_values[0]:.2f}")
    print(f"最终MSE: {mse_values[-1]:.2f}")
    print(f"最小MSE: {min_mse:.2f} (Epoch {min_epoch})")
    print(f"MSE改善: {((mse_values[0] - min_mse) / mse_values[0] * 100):.2f}%")

def read_log_file(file_path):
    """
    读取日志文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    except PermissionError:
        print(f"错误: 没有权限读取文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    """
    主程序 - 直接指定文件夹路径并处理
    """
    print("训练损失曲线绘制工具")
    print("=" * 50)
    
    # 直接指定日志文件夹路径
    folder_path = r"/home/sl/project/attn_props_sjzq/log/X70/feather0.83/2026-01-04_14-18/logs"  # 修改此路径为您的实际日志文件夹路径
    
    # 检查是否为有效目录
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个有效的文件夹")
        return
    
    print(f"正在处理文件夹: {folder_path}")
    
    # 列出文件夹中的文件，仅处理 .txt 和 .log 文件
    log_files = [f for f in os.listdir(folder_path) if f.endswith(('.txt', '.log'))]
    if not log_files:
        print(f"文件夹 '{folder_path}' 中没有找到 .txt 或 .log 文件")
        return
    
    # 逐个处理日志文件
    for log_file in log_files:
        file_path = os.path.join(folder_path, log_file)
        print(f"正在读取文件: {file_path}")
        
        # 读取日志内容
        log_content = read_log_file(file_path)
        if log_content is None:
            continue
        
        # 解析日志
        epochs, mse_values = parse_training_log(log_content)
        
        if not epochs or not mse_values:
            print(f"文件 '{log_file}' 未找到MSE数据，请检查日志文件格式")
            print("确保文件中包含 'Current Instance MSE: 数值' 格式的行")
            continue
        
        print(f"成功解析到 {len(epochs)} 个epoch的数据")
        
        # 绘制并保存损失曲线
        try:
            plot_loss_curve(epochs, mse_values, file_path)
        except Exception as e:
            print(f"绘制图表时出错: {e}")
    
    print("程序结束")

if __name__ == "__main__":
    main()