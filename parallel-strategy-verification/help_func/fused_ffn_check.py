#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理目录下所有 .distcp：
- 找出名称包含 "gate_up_fused" 的共同 key
- 对每个共同 key：
  - 读取各文件对应的 Tensor，沿最后一维切成两半（前半、后半）
  - 按文件顺序拼接所有前半，再拼接所有后半（均沿最后一维）
  - 分别输出两个拼接结果的形状和 md5
- 输出写入目录下 gate_ffn_grouped_md5.log

用法：
  python gate_ffn_group.py <dir_contains_distcp>
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import glob

import numpy as np
import paddle


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat = {}
    for k, v in d.items():
        nk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, nk))
        else:
            flat[nk] = v
    return flat


def to_tensor(value: Any) -> paddle.Tensor:
    if isinstance(value, paddle.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return paddle.to_tensor(value)
    if isinstance(value, (int, float, bool, np.number)):
        return paddle.to_tensor(value)
    if hasattr(value, "numpy"):
        try:
            return paddle.to_tensor(value.numpy())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            arr = np.array(value.tolist())
            return paddle.to_tensor(arr)
        except Exception:
            pass
    return paddle.to_tensor(value)


def tensor_md5(t: paddle.Tensor) -> str:
    try:
        return t._md5sum()
    except Exception:
        return "<md5-unavailable>"


def format_tensor_line(t: paddle.Tensor) -> str:
    try:
        shape = list(t.shape)
        dtype = str(t.dtype)
        if not dtype.startswith("paddle."):
            dtype = f"paddle.{dtype}"
        md5 = tensor_md5(t)
        return f"Tensor(shape={shape}, dtype={dtype}, md5={md5})"
    except Exception:
        return f"Tensor(shape=?, dtype=?, md5=<md5-unavailable>)"


def load_flat_state(distcp_path: Path) -> Dict[str, Any]:
    state = paddle.load(str(distcp_path), return_numpy=True)
    if isinstance(state, dict):
        return flatten_dict(state)
    return {"<root>": state}


def split_half_last_dim(t: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    arr = t.numpy()
    last = arr.shape[-1]
    mid = last // 2
    # 前半 [0:mid], 后半 [mid:]
    first = arr[..., :mid]
    second = arr[..., mid:]
    return paddle.to_tensor(first), paddle.to_tensor(second)


def process_dir(dir_path: Path) -> Path:
    pattern = os.path.join(str(dir_path), "*.distcp")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("目录下没有 .distcp 文件")

    flat_states: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        flat_states.append((os.path.basename(fp), load_flat_state(Path(fp))))

    # 共同 keys 交集，仅包含 gate_up_fused
    common_keys = None
    for _, st in flat_states:
        ks_set = {k for k in st.keys() if "gate_up_fused" in k}
        common_keys = ks_set if common_keys is None else (common_keys & ks_set)
    common_keys = sorted(common_keys) if common_keys else []

    out_path = dir_path / "gate_ffn_grouped_md5.log"
    with open(out_path, "w", encoding="utf-8") as f:
        for key in common_keys:
            f.write(f"[{key}]\n")
            # 收集各文件的前半与后半
            first_halves: List[paddle.Tensor] = []
            second_halves: List[paddle.Tensor] = []
            for _, st in flat_states:
                t = to_tensor(st[key])
                first, second = split_half_last_dim(t)
                first_halves.append(first)
                second_halves.append(second)
            # 沿最后一维拼接
            try:
                first_cat = paddle.concat(first_halves, axis=-1)
                second_cat = paddle.concat(second_halves, axis=-1)
            except Exception as e:
                f.write(f"<concat-error>: {e}\n\n")
                continue
            # 输出两行
            f.write(format_tensor_line(first_cat) + "\n")
            f.write(format_tensor_line(second_cat) + "\n\n")
    return out_path


def main():
    if len(sys.argv) != 2:
        print("用法: python gate_ffn_group.py <dir_contains_distcp>")
        sys.exit(1)

    dir_path = Path(sys.argv[1])
    if not dir_path.exists() or not dir_path.is_dir():
        print("错误: 第一个参数是目录路径")
        sys.exit(1)

    try:
        out_log = process_dir(dir_path)
    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)
    print(f"生成日志: {out_log}")


if __name__ == "__main__":
    main()
