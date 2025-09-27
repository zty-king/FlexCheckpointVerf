#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录对齐切分并输出分段 MD5（仅模式C）：
  - 对目录下所有 .distcp，找出共同的 key；
  - 对每个共同 key，沿最后一维切成 nums 份；
  - 然后“按文件顺序”依次输出三段：前 heads 份、接着 ks 份、接着 vs 份（要求 heads+ks+vs=nums）；
  - 仅处理名称包含 "qkv" 的共同 key；
  - 每个 chunk 输出一行，格式：
      Tensor(shape=[...], dtype=paddle.float32, md5=xxxxxxxx)
  - 所有输出写入目录下 grouped_md5.log

用法：
  python print_distcp.py <dir_contains_distcp> <nums> <heads> <ks> <vs>
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
    try:
        return paddle.to_tensor(value)
    except Exception:
        return paddle.to_tensor(str(value))


def tensor_md5(t: paddle.Tensor) -> str:
    try:
        return t._md5sum()
    except Exception:
        return "<md5-unavailable>"


def format_chunk_line(t: paddle.Tensor) -> str:
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


def split_last_dim_to_chunks(t: paddle.Tensor, num_chunks: int) -> List[paddle.Tensor]:
    arr = t.numpy()
    chunks = np.array_split(arr, num_chunks, axis=arr.ndim - 1)
    return [paddle.to_tensor(c) for c in chunks]


def process_dir_grouped_md5(dir_path: Path, nums: int, heads: int, ks: int, vs: int) -> Path:
    if heads + ks + vs != nums:
        raise ValueError("heads+ks+vs 必须等于 nums")

    pattern = os.path.join(str(dir_path), "*.distcp")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("目录下没有 .distcp 文件")

    flat_states: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        flat_states.append((os.path.basename(fp), load_flat_state(Path(fp))))

    common_keys = None
    for _, st in flat_states:
        ks_set = set(st.keys())
        common_keys = ks_set if common_keys is None else (common_keys & ks_set)
    # 仅处理包含 'qkv' 的 keys
    if common_keys:
        common_keys = sorted([k for k in common_keys if "qkv" in k])
    else:
        common_keys = []

    out_path = dir_path / "grouped_md5.log"
    with open(out_path, "w", encoding="utf-8") as f:
        for key in common_keys:
            f.write(f"[{key}]\n")
            # heads 段
            for _, st in flat_states:
                t = to_tensor(st[key])
                chunks = split_last_dim_to_chunks(t, nums)
                for c in chunks[0:heads]:
                    f.write(format_chunk_line(c) + "\n")
            # ks 段
            for _, st in flat_states:
                t = to_tensor(st[key])
                chunks = split_last_dim_to_chunks(t, nums)
                for c in chunks[heads:heads+ks]:
                    f.write(format_chunk_line(c) + "\n")
            # vs 段
            for _, st in flat_states:
                t = to_tensor(st[key])
                chunks = split_last_dim_to_chunks(t, nums)
                for c in chunks[heads+ks:nums]:
                    f.write(format_chunk_line(c) + "\n")
            f.write("\n")
    return out_path


def main():
    if len(sys.argv) != 6:
        print("用法: python print_distcp.py <dir_contains_distcp> <nums> <heads> <ks> <vs>")
        sys.exit(1)

    dir_path = Path(sys.argv[1])
    if not dir_path.exists() or not dir_path.is_dir():
        print("错误: 第一个参数是目录路径")
        sys.exit(1)
    try:
        nums = int(sys.argv[2])
        heads = int(sys.argv[3])
        ks = int(sys.argv[4])
        vs = int(sys.argv[5])
        assert nums >= 1 and heads >= 0 and ks >= 0 and vs >= 0
    except Exception:
        print("错误: nums/heads/ks/vs 需为整数，且非负（nums>=1）")
        sys.exit(1)

    try:
        out_log = process_dir_grouped_md5(dir_path, nums, heads, ks, vs)
    except Exception as e:
        print(f"处理失败: {e}")
        sys.exit(1)
    print(f"生成日志: {out_log}")


if __name__ == "__main__":
    main()
