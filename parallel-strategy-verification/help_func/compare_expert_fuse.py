#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import paddle


def build_regex_from_wildcard(pattern: str) -> re.Pattern:
    """
    将类似 "ernie.layers.1.mlp.experts.*.down_proj.weight" 的通配符模式
    转为正则表达式，只有一个 '*'，且表示一个或多个数字 (\d+)。
    返回编译好的正则对象，并捕获该数字分组。
    """
    if pattern.count("*") != 1:
        raise ValueError("通配符模式必须且只能包含一个 '*'")
    # 对 '.' 做转义，其余字符保持原样
    escaped = []
    for ch in pattern:
        if ch == '.':
            escaped.append(r"\.")
        elif ch == '*':
            escaped.append(r"(\d+)")
        else:
            escaped.append(ch)
    regex_str = "".join(escaped)
    return re.compile(f"^{regex_str}$")


def list_distcp_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"找不到文件夹: {folder}")
    files = sorted(folder.glob("*.distcp"))
    if not files:
        raise FileNotFoundError(f"目录 {folder} 未找到任何 .distcp 文件")
    return files


def load_state_from_files(files: List[Path]) -> List[Tuple[Path, dict]]:
    results = []
    for f in files:
        try:
            state = paddle.load(str(f))
        except Exception as e:
            raise RuntimeError(f"加载失败 {f}: {e}")
        results.append((f, state))
    return results


def ensure_numpy_array(value) -> np.ndarray:
    """
    将权重值转换为 numpy.ndarray。
    可能是 numpy.ndarray、paddle.Tensor 或其他可转 numpy 的类型。
    """
    if isinstance(value, np.ndarray):
        return value
    try:
        # paddle.Tensor 支持 .numpy()
        if hasattr(value, 'numpy'):
            return value.numpy()
    except Exception:
        pass
    # 回退到 np.array 尝试
    try:
        return np.array(value)
    except Exception as e:
        raise TypeError(f"无法将类型 {type(value)} 转为 numpy.ndarray: {e}")


def md5_of_ndarray(arr: np.ndarray) -> str:
    # 使用 C 连续内存的字节
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    h = hashlib.md5()
    h.update(arr.tobytes(order='C'))
    return h.hexdigest()


def collect_expert_tensors(
    folder: Path, pattern: str
) -> Dict[int, np.ndarray]:
    """
    在 folder 中所有 .distcp 文件里查找匹配 pattern(包含一个 * 通配符) 的 key，
    收集为 {index: ndarray}，index 为 * 捕获的数字，确保每个 index 唯一。
    """
    regex = build_regex_from_wildcard(pattern)
    print(f"regex: {regex}")
    files = list_distcp_files(folder)
    tensors_by_index: Dict[int, np.ndarray] = {}

    for p, state in load_state_from_files(files):
        for k, v in state.items():
            if "ernie.layers.1.mlp.experts.0.down_proj.weight" in k:
                print("--------------------------------")
                print(f"k: {k}")
                print("--------------------------------")
            if not isinstance(k, str):
                continue
            k=k.strip()
            m = regex.fullmatch(k)
            if m is None:
                continue
            idx = int(m.group(1))
            if idx in tensors_by_index:
                raise ValueError(
                    f"在 {p} 中发现重复 index={idx} 的 key: {k}，已在其他文件中存在。"
                )
            tensors_by_index[idx] = ensure_numpy_array(v)

    if not tensors_by_index:
        raise KeyError(f"未在 {folder} 中找到匹配通配符模式的 key: {pattern}")

    return tensors_by_index


def find_fused_tensor(folder: Path, fused_key: str) -> np.ndarray:
    files = list_distcp_files(folder)
    found: List[Tuple[Path, np.ndarray]] = []
    for p, state in load_state_from_files(files):
        if fused_key in state:
            found.append((p, ensure_numpy_array(state[fused_key])))

    if not found:
        raise KeyError(f"未在 {folder} 中找到 fused key: {fused_key}")
    if len(found) > 1:
        paths = ", ".join(str(x[0]) for x in found)
        raise ValueError(f"fused key 在多个文件中出现: {paths}")
    return found[0][1]


def validate_and_compare(
    experts: Dict[int, np.ndarray], fused: np.ndarray, axis: int
) -> None:
    if axis < 0:
        axis = fused.ndim + axis
    if axis < 0 or axis >= fused.ndim:
        raise ValueError(f"axis 超出范围: {axis}, fused.ndim={fused.ndim}")

    # 依 index 从小到大排序
    ordered: List[Tuple[int, np.ndarray]] = sorted(experts.items(), key=lambda x: x[0])

    # 形状校验：非 axis 维度必须一致，axis 维度之和匹配 fused 的 axis 维
    fused_shape = fused.shape
    sum_axis = 0
    ref_other_dims = None
    sizes: List[Tuple[int, int]] = []  # (index, size_along_axis)
    for idx, arr in ordered:
        if arr.ndim != fused.ndim:
            raise ValueError(f"index={idx} 维度不匹配: expert.ndim={arr.ndim}, fused.ndim={fused.ndim}")
        if ref_other_dims is None:
            ref_other_dims = arr.shape
        # 检查非 axis 维
        for d in range(arr.ndim):
            if d == axis:
                continue
            if arr.shape[d] != fused_shape[d]:
                raise ValueError(
                    f"index={idx} 非 axis 维 {d} 大小不匹配: expert={arr.shape[d]}, fused={fused_shape[d]}"
                )
        size = arr.shape[axis]
        sum_axis += size
        sizes.append((idx, size))

    if sum_axis != fused_shape[axis]:
        raise ValueError(
            f"axis={axis} 维度求和不等: sum(experts)={sum_axis}, fused={fused_shape[axis]}"
        )

    # Debug: 打印分片尺寸与专家/融合的 MD5 列表，便于快速对照
    print(f"axis={axis} 分片尺寸(按专家 index 升序): {[s for _, s in sizes]}")
    expert_md5_list: List[Tuple[int, str]] = []
    for idx, arr in ordered:
        expert_md5_list.append((idx, md5_of_ndarray(arr)))
    print("专家张量 MD5 列表:")
    print([f"idx={i}:{m}" for i, m in expert_md5_list])

    fused_md5_list: List[Tuple[int, str]] = []  # 与专家顺序一致
    offset = 0
    for idx, size in sizes:
        slicers = [slice(None)] * fused.ndim
        slicers[axis] = slice(offset, offset + size)
        fused_slice = fused[tuple(slicers)]
        fused_md5_list.append((idx, md5_of_ndarray(fused_slice)))
        offset += size
    print("fused 张量按 n 份切片后的 MD5 列表:")
    print([f"idx={i}:{m}" for i, m in fused_md5_list])

    # 比较：按顺序在 fused 上沿 axis 切片，与每个 expert 的 md5 比较
    offset = 0
    for idx, arr in ordered:
        size = arr.shape[axis]
        slicers = [slice(None)] * fused.ndim
        slicers[axis] = slice(offset, offset + size)
        fused_slice = fused[tuple(slicers)]

        md5_expert = md5_of_ndarray(arr)
        md5_fused = md5_of_ndarray(fused_slice)
        ok = md5_expert == md5_fused

        print(
            f"index={idx}: shape={arr.shape} vs fused_slice={fused_slice.shape}, md5_equal={ok}, "
            f"expert_md5={md5_expert}, fused_md5={md5_fused}"
        )

        if not ok:
            raise AssertionError(
                f"index={idx} 的张量与 fused 对应切片 md5 不一致"
            )

        offset += size

    print("所有专家张量与 fused 切片 md5 一致，校验通过。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="专家权重与 fused 权重对齐校验工具")
    parser.add_argument("folder_experts", type=str, help="包含专家 .distcp 的目录")
    parser.add_argument("folder_fused", type=str, help="包含 fused .distcp 的目录")
    parser.add_argument(
        "--pattern",
        required=True,
        type=str,
        help="专家通配符 key，例如: ernie.layers.1.mlp.experts.*.down_proj.weight",
    )
    parser.add_argument(
        "--fused_key",
        required=True,
        type=str,
        help="fused key，例如: ernie.layers.1.mlp.experts_fused.down_proj.weight",
    )
    parser.add_argument(
        "--axis", type=int, required=True, help="沿哪个维度进行拼接校验，例如 1"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder_experts = Path(args.folder_experts)
    folder_fused = Path(args.folder_fused)

    print(f"读取专家目录: {folder_experts}")
    experts = collect_expert_tensors(folder_experts, args.pattern)
    print(f"共匹配专家张量: {len(experts)} 个; indices={sorted(experts.keys())}")

    print(f"读取 fused 目录: {folder_fused}")
    fused = find_fused_tensor(folder_fused, args.fused_key)
    print(f"fused 形状: {fused.shape}")

    print(f"开始校验 axis={args.axis} ...")
    validate_and_compare(experts, fused, args.axis)


if __name__ == "__main__":
    main()


