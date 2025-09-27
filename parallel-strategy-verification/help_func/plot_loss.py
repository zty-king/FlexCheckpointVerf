#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从两个训练日志中提取loss数据并在同一张图中绘制曲线图
用法:
  python -m plot_loss.py <log_file_1> <log_file_2> <output_path_or_dir> [image_name] [label1] [label2]
说明:
  - 若提供 image_name，则第3个参数视为输出目录，最终保存为 <output_dir>/<image_name>
  - 若未提供 image_name：
      * 若第3个参数带有后缀名(如 .png/.jpg)，则视为完整输出路径
      * 否则视为目录，使用默认文件名 loss_compare.png
  - 若提供 label1、label2，则作为两条曲线的图例名称；未提供则默认使用日志文件名
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def extract_loss_from_log(log_file_path):
    """
    从日志文件中提取loss数据
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        global_steps: 全局步数列表
        losses: loss值列表
    """
    global_steps = []
    losses = []
    
    # 匹配loss和global_step的正则表达式
    pattern = r'loss: ([\d.]+),.*?global_step: (\d+)'
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                step = int(match.group(2))
                losses.append(loss)
                global_steps.append(step)
    
    return global_steps, losses


def plot_two_loss_curves(gs1, ls1, label1, gs2, ls2, label2, save_path):
    """
    在同一张图中绘制两条loss曲线并保存
    """
    plt.figure(figsize=(12, 8))
    
    # 第一条曲线（蓝）
    plt.plot(gs1, ls1, 'b-', linewidth=2, marker='o', markersize=3, label=label1)
    
    # 第二条曲线（红）
    plt.plot(gs2, ls2, 'r-', linewidth=2, marker='s', markersize=3, label=label2)
    
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    
    # 使用图片名作为title的一部分
    image_name = Path(save_path).stem  # 获取文件名（不含扩展名）
    plt.title(f'{image_name} train loss', fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 设置坐标轴范围
    max_step = 0
    if gs1:
        max_step = max(max_step, max(gs1))
    if gs2:
        max_step = max(max_step, max(gs2))
    if max_step > 0:
        plt.xlim(0, max_step + 1)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    # 不在这里调用 show()，以便命令行批量运行


def main():
    # 读取命令行参数
    if not (len(sys.argv) in (4, 5, 6, 7)):
        print("用法: python -m plot_loss.py <log_file_1> <log_file_2> <output_path_or_dir> [image_name] [label1] [label2]")
        sys.exit(1)
    
    log1 = sys.argv[1]
    log2 = sys.argv[2]
    out_arg = sys.argv[3]
    img_name = sys.argv[4] if len(sys.argv) >= 5 else None
    label1_cli = sys.argv[5] if len(sys.argv) >= 6 else None
    label2_cli = sys.argv[6] if len(sys.argv) >= 7 else None

    # 解析输出路径
    out_path_obj = Path(out_arg)
    if img_name:  # 指定了图片名，则将第三个参数视为目录
        save_path = out_path_obj / img_name
    else:
        # 未指定图片名：若第三个参数有后缀名视为完整文件路径，否则用默认名
        if out_path_obj.suffix:
            save_path = out_path_obj
        else:
            save_path = out_path_obj / "loss_compare.png"
    
    # 检查文件是否存在
    if not Path(log1).exists():
        print(f"错误: 找不到日志文件 {log1}")
        sys.exit(1)
    if not Path(log2).exists():
        print(f"错误: 找不到日志文件 {log2}")
        sys.exit(1)
    
    print("正在从日志文件中提取loss数据...")
    gs1, ls1 = extract_loss_from_log(log1)
    gs2, ls2 = extract_loss_from_log(log2)
    
    if not gs1:
        print(f"错误: {log1} 中没有找到loss数据")
        sys.exit(1)
    if not gs2:
        print(f"错误: {log2} 中没有找到loss数据")
        sys.exit(1)
    
    print(f"{Path(log1).name}: 提取 {len(gs1)} 个数据点")
    print(f"{Path(log2).name}: 提取 {len(gs2)} 个数据点")
    
    # 曲线标签：优先使用命令行提供的名称，否则用文件名
    label1 = label1_cli if label1_cli else Path(log1).name
    label2 = label2_cli if label2_cli else Path(log2).name
    
    plot_two_loss_curves(gs1, ls1, label1, gs2, ls2, label2, str(save_path))


if __name__ == "__main__":
    main() 