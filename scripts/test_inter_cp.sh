#!/bin/bash
export NCCL_DEBUG=WARN
torchrun --nproc_per_node=4 --master_port=29500 tests/test_inter_cp.py