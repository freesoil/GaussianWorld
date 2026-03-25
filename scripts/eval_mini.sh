#!/bin/bash
PY_CONFIG="config/nusc_surroundocc_mini_eval.py"
CKPT_PATH=$1
WORK_DIR=$2

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 63545";

echo "command = [torchrun $DISTRIBUTED_ARGS eval.py]"
torchrun $DISTRIBUTED_ARGS eval.py \
    --py-config $PY_CONFIG \
    --load-from $CKPT_PATH \
    --work-dir $WORK_DIR ${@:3}