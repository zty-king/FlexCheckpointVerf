#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较两个文件夹中同名的 .distcp 文件的 state dict 的 MD5 值
用法:
  python compare_checkpoints.py <folder1> <folder2>
"""

import os
import paddle
import glob
from pathlib import Path
from typing import Dict, List
import sys
import numpy as np


def get_distcp_files(folder_path: str) -> List[str]:
    """
    获取文件夹中以 .distcp 结尾的文件
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return []
    
    pattern = str(folder / "*.distcp")
    return glob.glob(pattern)


def flatten_state_dict(state_dict: Dict, prefix: str = "") -> Dict[str, any]:
    """
    展平嵌套的 state dict
    """
    flattened = {}
    for key, value in state_dict.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_state_dict(value, new_key))
        else:
            flattened[new_key] = value
    return flattened


def get_md5_sum(tensor) -> str:
    """
    获取 tensor 的 MD5 值
    """
    try:
        # 若为 numpy.ndarray，先转为 paddle.Tensor
        if isinstance(tensor, np.ndarray):
            tensor = paddle.to_tensor(tensor)
        # 其他非 Tensor 类型直接返回 None
        if not isinstance(tensor, paddle.Tensor):
            return None
        return tensor._md5sum()
    except Exception:
        return None


def compare_state_dicts(state_dict1: Dict, state_dict2: Dict, file1_name: str, file2_name: str) -> Dict:
    """
    比较两个 state dict 的 MD5 值
    """
    flat_dict1 = flatten_state_dict(state_dict1)
    flat_dict2 = flatten_state_dict(state_dict2)
    all_keys = set(flat_dict1.keys()) | set(flat_dict2.keys())
    results = {
        "file1": file1_name,
        "file2": file2_name,
        "total_keys": len(all_keys),
        "keys_only_in_file1": [],
        "keys_only_in_file2": [],
        "matching_keys": [],
        "different_md5_keys": [],
        "same_md5_keys": [],
        "md5_comparison": {}
    }
    for key in all_keys:
        if key not in flat_dict1:
            results["keys_only_in_file2"].append(key)
        elif key not in flat_dict2:
            results["keys_only_in_file1"].append(key)
        else:
            results["matching_keys"].append(key)
            md5_1 = get_md5_sum(flat_dict1[key])
            md5_2 = get_md5_sum(flat_dict2[key])
            results["md5_comparison"][key] = {
                "file1_md5": md5_1,
                "file2_md5": md5_2,
                "same": md5_1 == md5_2
            }
            if md5_1 == md5_2:
                results["same_md5_keys"].append(key)
            else:
                results["different_md5_keys"].append(key)
    return results


def print_comparison_results(results: Dict):
    """
    打印比较结果（简洁版）
    """
    print(f"比较: {os.path.basename(results['file1'])} vs {os.path.basename(results['file2'])}")
    print(
        f"总键数: {results['total_keys']}, 匹配: {len(results['matching_keys'])}, "
        f"相同MD5: {len(results['same_md5_keys'])}, 不同MD5: {len(results['different_md5_keys'])}"
    )
    if results['total_keys'] > 0:
        similarity = len(results['same_md5_keys']) / results['total_keys'] * 100
        print(f"相似度: {similarity:.2f}%")


def main():
    # 读取命令行参数
    if len(sys.argv) != 3:
        print("用法: python compare_checkpoints.py <folder1> <folder2>")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    if not Path(folder1).exists():
        print(f"错误: 文件夹不存在 -> {folder1}")
        sys.exit(1)
    if not Path(folder2).exists():
        print(f"错误: 文件夹不存在 -> {folder2}")
        sys.exit(1)

    print(f"比较文件夹: {folder1} vs {folder2}")
    
    # 仅比较 .distcp 文件
    distcp1 = get_distcp_files(folder1)
    distcp2 = get_distcp_files(folder2)
    
    all_similarities = []  # 存储所有文件的相似度
    
    if distcp1 and distcp2:
        distcp1_names = {os.path.basename(f): f for f in distcp1}
        distcp2_names = {os.path.basename(f): f for f in distcp2}
        common_distcp = set(distcp1_names.keys()) & set(distcp2_names.keys())
        if not common_distcp:
            print("两个目录间没有同名的 .distcp 文件可比较")
            print("结论: MD5匹配失败")
            sys.exit(0)
        
        for filename in sorted(common_distcp):
            file1_path = distcp1_names[filename]
            file2_path = distcp2_names[filename]
            try:
                state_dict1 = paddle.load(file1_path, return_numpy=True)
                state_dict2 = paddle.load(file2_path, return_numpy=True)
                results = compare_state_dicts(state_dict1, state_dict2, file1_path, file2_path)
                print_comparison_results(results)
                
                # 计算当前文件的相似度
                if results['total_keys'] > 0:
                    similarity = len(results['same_md5_keys']) / results['total_keys'] * 100
                    all_similarities.append(similarity)
                else:
                    # 对于空文件（总键数为0），认为相似度为100%
                    all_similarities.append(100.0)
                    
            except Exception as e:
                print(f"比较 {filename} 时出错: {e}")
                all_similarities.append(0)  # 出错时认为相似度为0
        
        # 打印总体结论
        print("\n" + "="*50)
        if all_similarities and all(sim == 100.0 for sim in all_similarities):
            print("结论: MD5匹配通过 ✓")
        else:
            print("结论: MD5匹配失败 ✗")
        print("="*50)
        
    else:
        print("未找到可比较的 .distcp 文件")
        print("结论: MD5匹配失败")


if __name__ == "__main__":
    main()