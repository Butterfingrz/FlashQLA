#!/bin/bash
export NCCL_DEBUG=WARN
# torchrun --nproc_per_node=4 --master_port=29500 profile/profile_inter_cp.py \
#     --set cp \
#     --fla

torchrun --nproc_per_node=4 --master_port=29500 profile/profile_inter_intra_cp.py \
    --seqlen 32768 \
    --nvh 32 \
    --nkh 32 \
    --cu-seqlens 0-32768