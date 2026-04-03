PY_CONFIG=$1
WORK_DIR=$2

if [ -n "${NPROC_PER_NODE}" ]; then
  NPROC="${NPROC_PER_NODE}"
else
  NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ -z "${NPROC}" ] || [ "${NPROC}" -lt 1 ]; then
    NPROC=1
  fi
fi

DISTRIBUTED_ARGS="--nproc_per_node ${NPROC} --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 63545";

echo "command = [torchrun $DISTRIBUTED_ARGS train.py]"
torchrun $DISTRIBUTED_ARGS train.py \
    --py-config $PY_CONFIG \
    --work-dir $WORK_DIR
