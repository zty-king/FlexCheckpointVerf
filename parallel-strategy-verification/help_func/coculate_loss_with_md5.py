#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从两个训练日志中提取信息，比较：
1) 优先比较 loss_md5：前一个log的最后一个 vs 后一个log的第一个
   - 若二者都存在且相等 -> 输出 "md5对齐"
   - 若不相等或缺失 -> 回退比较loss
2) 回退比较 loss：前一个log的最后一个loss 与 后一个log的第一个loss 的差值精度
   - 若完全相等 -> 输出 0
   - 否则输出 1E±n（例如差0.01 -> 1E-2）

用法:
  python -m coculate_loss_with_md5.py <log_file_1> <log_file_2>
"""

import re
import numpy as np
from pathlib import Path
import sys


LOSS_PATTERN = re.compile(r"loss:\s*([\d.]+),.*?global_step:\s*(\d+)")
# 尝试匹配形如 "loss_md5: abcdef..." 或 "loss md5: ..." 的行
LOSS_MD5_PATTERN = re.compile(r"loss[_\s]?md5:\s*([0-9a-fA-F]{8,})")


def extract_from_log(log_file_path):
    """
    从日志文件中提取：
    - global_steps: list[int]
    - losses: list[float]
    - loss_md5s: list[str]  (出现顺序)
    """
    global_steps = []
    losses = []
    loss_md5s = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = LOSS_PATTERN.search(line)
            if m:
                loss = float(m.group(1))
                step = int(m.group(2))
                losses.append(loss)
                global_steps.append(step)
            mm = LOSS_MD5_PATTERN.search(line)
            if mm:
                loss_md5s.append(mm.group(1))

    return global_steps, losses, loss_md5s


def main():
    if len(sys.argv) != 3:
        print("用法: python -m coculate_loss_with_md5.py <log_file_1> <log_file_2>")
        sys.exit(1)

    log1 = sys.argv[1]
    log2 = sys.argv[2]

    if not Path(log1).exists():
        print(f"错误: 找不到日志文件 {log1}")
        sys.exit(1)
    if not Path(log2).exists():
        print(f"错误: 找不到日志文件 {log2}")
        sys.exit(1)

    _, ls1, md5s1 = extract_from_log(log1)
    _, ls2, md5s2 = extract_from_log(log2)

    # 1) 优先比较 loss_md5（log1最后一个 vs log2第一个）
    md5_last_of_log1 = md5s1[-1] if md5s1 else None
    md5_first_of_log2 = md5s2[0] if md5s2 else None

    if md5_last_of_log1 is not None and md5_first_of_log2 is not None:
        if md5_last_of_log1 == md5_first_of_log2:
            print("md5对齐")
            return
        # 若不相等则回退到比较 loss

    # 2) 回退比较 loss（log1最后一个 vs log2第一个）
    if not ls1:
        print(f"错误: {log1} 中没有找到loss数据")
        sys.exit(1)
    if not ls2:
        print(f"错误: {log2} 中没有找到loss数据")
        sys.exit(1)

    v1 = ls1[-1]
    v2 = ls2[0]
    diff = abs(v1 - v2)

    if diff == 0.0 or np.isclose(diff, 0.0, atol=1e-15):
        print(0)
        return

    exp = int(np.floor(np.log10(diff)))
    print(f"1E{exp}")


if __name__ == "__main__":
    main()
