#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==================== 环境变量设置 ====================
echo "设置环境变量..."

export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
unset GLOG_vmodule GLOG_v
export PADDLE_DISABLE_CUDNN_FA=1
export FLAGS_use_auto_growth_pinned_allocator=True
export FLAGS_pipeline_nccl_comm_init_option=1
export FLAGS_sharding_v2_check_zero_padding=1
export FLAGS_use_paddle_recall_error=0
export FLAGS_tcp_max_syn_backlog=16384
export FLAGS_call_stack_level=2
export FLAGS_cudnn_deterministic=True
export FLAGS_embedding_deterministic=1 


# 检查 GPU 计算能力
SM=`nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//g'`
echo "GPU 计算能力: $SM"

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:./ernie

export R0_MOE_GROUP="dummy"
export R0_DATA_PARALLEL_DEGREE=2
export R0_TENSOR_PARALLEL_DEGREE=1
export R0_PIPELINE_PARALLEL_DEGREE=1
export R0_EXPERT_PARALLEL_DEGREE=1
export R0_SHARDING_PARALLEL_DEGREE=1
export R0_VIRTUAL_PP_DEGREE=1

export R1_MOE_GROUP="dummy"
export R1_DATA_PARALLEL_DEGREE=2
export R1_TENSOR_PARALLEL_DEGREE=1
export R1_PIPELINE_PARALLEL_DEGREE=1
export R1_EXPERT_PARALLEL_DEGREE=1
export R1_SHARDING_PARALLEL_DEGREE=1
export R1_VIRTUAL_PP_DEGREE=1


# 统一根目录与任务名（对齐 run_pretrain_llm.sh 的结构）
ROOT_DIR="/home/ERNIE/examples/pre-training"
task_name="DP2"

case_temp0_out_dir="${ROOT_DIR}/temp0/${task_name}"
case_temp0_log_dir="${ROOT_DIR}/temp0/${task_name}_log"

case_temp1_out_dir="${ROOT_DIR}/temp1/${task_name}"
case_temp1_log_dir="${ROOT_DIR}/temp1/${task_name}_log"


# # 清理旧目录
# rm -rf "$case_temp0_out_dir" "$case_temp0_log_dir" \
#        "$case_temp1_out_dir" "$case_temp1_log_dir" \

run_with_yaml() {
    local OUT_DIR="$1"
    local LOG_DIR="$2"
    local RESUME_FROM="$3"   # 为空则不写入
    local MAX_STEPS="$4"     # 每轮步数
    local SAVE_STEPS="$5"    # 保存步数
    local MOE_GROUP_IN="$6"  # moe_group（未传入则默认 ep）
    local SHARD_DEG_IN="$7"  # sharding_parallel_degree（未传入则默认 1）
    local DP_DEG_IN="$8"     # data_parallel_degree（未传入则默认 1）
    local TP_DEG_IN="$9"     # tensor_parallel_degree（未传入则默认 1）
    local EP_DEG_IN="${10}"  # expert_parallel_degree（未传入则默认 1）
    local PP_DEG_IN="${11}"  # pipeline_parallel_degree（未传入则默认 1）
    local VPP_DEG_IN="${12}" # virtual_pp_degree（未传入则默认 1）

    # 默认值
    local MOE_GROUP_VAL=${MOE_GROUP_IN:-ep}
    local SHARD_DEG_VAL=${SHARD_DEG_IN:-1}
    local DP_DEG_VAL=${DP_DEG_IN:-1}
    local TP_DEG_VAL=${TP_DEG_IN:-1}
    local EP_DEG_VAL=${EP_DEG_IN:-1}
    local PP_DEG_VAL=${PP_DEG_IN:-1}
    local VPP_DEG_VAL=${VPP_DEG_IN:-1}

    local TEMP_CONFIG_FILE="/tmp/pretrain_config_$$.yaml"

    cat > $TEMP_CONFIG_FILE << EOF
# -----------环境变量----------------------#
env:
    HOME: null

# ---------------------------model args-------------------------------------------------#
model_args:
    model_name_or_path: model_configs/ERNIE-4p5-21B-A3B/
    tokenizer_name: ./ernie/src/tokenizers/tokenizer_model
    output_dir: ${OUT_DIR}
    data_load_process_num: 40
    max_seq_length: 1024
    base_seq_length: 1024
    num_consecutive: 32

    enable_global_training_logs: False
    enable_mtp_magic_send: False
    moe_use_aux_free_update_coef: 0.001
    global_logging_interval: 1

    model_config:
        hidden_size: 320
        intermediate_size: 5504
        use_quant_before_a2a: true
        use_async_a2a: false
        use_rms_qkv_recompute: false
        moe_logging: true
        use_recompute: false
        use_bias: false
        multi_token_pred_depth: 1
        use_fp8_mlp: false
        fuse_attention_qkv: true
        fuse_attention_ffn: true
        num_hidden_layers: 2
        remove_tail_layer: 0
        use_fp8_fuse_node: false
        use_flash_attention: 0
        use_ep_comm_overlap: false
        fp8_mem_configs:
            recompute_fwd_gate_up: false
            dequant_input: true
            shared_expert: false
        fp8_fused_ops_configs:
            stack_quant: true
            swiglu_probs_bwd: true
            split_group_gemm: true
            spaq: true
            transpose_split_quant: true
        use_combine_before_a2a: true

# ---------------------------trainer args-------------------------------------------------#
trainer_args:
    input_dir: "0.4 ./demo_data/data-1-part0 0.6 ./demo_data/data-1-part0"
    split: "998,1,1"
    gc_interval: 100000
    use_ortho_loss_callback: true
    do_train: True
    dataloader_num_workers: 8
    prefetch_factor: 32
    overwrite_output_dir: 0
    disable_tqdm: 1
    report_to: none
    logging_steps: 1
    eval_steps: 1000000
    eval_iters: -1
    save_steps: ${SAVE_STEPS}
    max_steps: ${MAX_STEPS}
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_epsilon: 1e-8
    learning_rate: 3.14e-4
    min_lr: 3.14e-6
    gradient_accumulation_steps: 16
    per_device_train_batch_size: 1
    $( [ -n "$RESUME_FROM" ] && echo "resume_from_checkpoint: ${RESUME_FROM}" )

    lr_scheduler: wsd:603000
    decay_function: 1-sqrt
    max_grad_norm: 1.0
    weight_decay: 0.1
    warmup_steps: 2000
    save_total_limit: 5
    fp16: True
    fp16_opt_level: "O2"
    scale_loss: 4096
    seed: 42
    use_train_part_sharding: 1
    pre_alloc_memory: 60
    sharding_comm_buffer_size_MB: 2048
    offload_optim: false

    # 并行度（可配，未传入则默认 1）
    data_parallel_degree: ${DP_DEG_VAL}
    tensor_parallel_degree: ${TP_DEG_VAL}
    pipeline_parallel_degree: ${PP_DEG_VAL}
    expert_parallel_degree: ${EP_DEG_VAL}
    virtual_pp_degree: ${VPP_DEG_VAL}
    sharding: "stage1"
    sharding_parallel_degree: ${SHARD_DEG_VAL}
    sharding_parallel_config: split_param
    amp_master_grad: 1

    ignore_data_skip: 0
    same_data: True
    enable_timer: 1
    skip_profile_timer: False
    skip_load_data_seq_cache: 1

    load_sharded_model: false
    save_sharded_model: false
    ignore_load_lr_and_optim: False
    moe_with_send_router_loss: False

    use_moe: true
    moe_group: ${MOE_GROUP_VAL}
    from_scratch: 1
    enable_optimizer_timer: False
    load_checkpoint_format: "flex_checkpoint" 
    save_checkpoint_format: "flex_checkpoint"
    aoa_config: {
    "aoa_statements": [
    "ernie.layers.1.mlp.experts.*.down_proj.weight -> ernie.layers.1.mlp.experts_fused.down_proj.weight,axis=1"
    ]
}
EOF

    echo "配置文件已创建: $TEMP_CONFIG_FILE"
    echo "开始训练... 日志: $LOG_DIR"

    # 运行（对齐 run_pretrain_llm.sh 的风格，产生日志目录）
    python -m paddle.distributed.launch \
        --gpus "$CUDA_VISIBLE_DEVICES" \
        --log_dir "$LOG_DIR" \
        /home/ERNIE/examples/pre-training/ernie/pretrain.py \
        --config $TEMP_CONFIG_FILE

    echo "清理临时文件..."
    rm -f $TEMP_CONFIG_FILE
}

# ########################################
# # Round 0: 预训练（产出 checkpoint-5）
# ########################################
# # 控制卡数，避免DP
# export CUDA_VISIBLE_DEVICES=0,1

# run_with_yaml "$case_temp0_out_dir" "$case_temp0_log_dir" "" 1 1 \
#     "${R0_MOE_GROUP}" "${R0_SHARDING_PARALLEL_DEGREE}" "${R0_DATA_PARALLEL_DEGREE}" \
#     "${R0_TENSOR_PARALLEL_DEGREE}" "${R0_EXPERT_PARALLEL_DEGREE}" "${R0_PIPELINE_PARALLEL_DEGREE}" \
#     "${R0_VIRTUAL_PP_DEGREE}"



########################################
# Round 1: 加载 Round 0 的 ckpt 继续训练（模拟一次转换后的加载）
########################################
export CUDA_VISIBLE_DEVICES=0,1
run_with_yaml "$case_temp1_out_dir" "$case_temp1_log_dir" "${case_temp0_out_dir}/checkpoint-1" 1 1 \
    "${R1_MOE_GROUP}" "${R1_SHARDING_PARALLEL_DEGREE}" "${R1_DATA_PARALLEL_DEGREE}" \
    "${R1_TENSOR_PARALLEL_DEGREE}" "${R1_EXPERT_PARALLEL_DEGREE}" "${R1_PIPELINE_PARALLEL_DEGREE}" \
    "${R1_VIRTUAL_PP_DEGREE}"


###测试要点：
# 1. round0测试时不配置aoa
# 2. round1测试时配置aoa
# 3. 需要在trainer文件中，dist.load_state_dict(sharded_state_dict, resume_from_checkpoint, aoa_config=self.args.aoa_config)
#   之前，给dst的model_state_dict添加一个对应的key和value，value大小要等于合并后的大小(这个key即fused对应的key)。
# 例如：model_sharded_state_dict["ernie.layers.1.mlp.experts_fused.down_proj.weight"]=ShardedWeight(
                    #     key="ernie.layers.1.mlp.experts_fused.down_proj.weight",
                    #     local_tensor=paddle.zeros(shape=[1536, 20480],dtype=model_sharded_state_dict["ernie.layers.1.mlp.experts.0.down_proj.weight"].local_tensor.dtype),
                    #     local_shape=[1536, 20480],
                    #     global_shape=[1536, 20480],
                    #     global_offset=[0, 0],
                    # )





