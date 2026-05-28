#!/bin/bash
# Eval-only jobs: load the last checkpoint within the 2-hr wall-clock budget
# for each model and report validation loss at that point.
# Submit all:    sbatch --array=0-2 scripts/gipfel-eval.sbatch
# Single model:  sbatch --array=0   scripts/gipfel-eval.sbatch   # 3b
#
# Array index → model / checkpoint (all confirmed within 2hr from job start):
#   0  3b   iter_4500   (1h51min)
#   1  1.5b iter_3500   (1h51min)
#   2  8b   iter_10000  (1h56min)
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=00:25:00
#SBATCH --job-name=gipfel-eval
#SBATCH --output=logs/%x-%a-%j.log
#SBATCH --error=logs/%x-%a-%j.log
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/iopsstor/scratch/cscs/course_00179/lsaie-ss26-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache

SEQ_LEN=4096

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) MODEL=3b;   NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8; MBS=4; GBS=128; TP=2; PP=2; EVAL_ITER=4500;  LR=1.5e-4; MIN_LR=1.5e-5; SRC_EXP=train-3b-8n ;;
  1) MODEL=1.5b; NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4; MBS=4; GBS=256; TP=2; PP=1; EVAL_ITER=3500;  LR=2e-4;   MIN_LR=2e-5;   SRC_EXP=train-1.5b-8n ;;
  2) MODEL=8b;   NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8; MBS=2; GBS=32;  TP=4; PP=2; EVAL_ITER=10000; LR=1e-4;   MIN_LR=1e-5;   SRC_EXP=train-8b-8n ;;
esac

SRC_CKPT=/iopsstor/scratch/cscs/$USER/gipfelsturm/gipfelsturm/${SRC_EXP}/checkpoints
EVAL_CKPT=/iopsstor/scratch/cscs/$USER/gipfelsturm/eval-tmp/eval-${MODEL}-${EVAL_ITER}-${SLURM_JOB_ID}

PROJECT_NAME=gipfelsturm
EXP_NAME=eval-${MODEL}-${EVAL_ITER}s-${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
#########################################

# Point a temporary checkpoint dir at the target iteration
mkdir -p $EVAL_CKPT $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR
ln -sfn $SRC_CKPT/iter_$(printf '%07d' $EVAL_ITER) $EVAL_CKPT/iter_$(printf '%07d' $EVAL_ITER)
echo $EVAL_ITER > $EVAL_CKPT/latest_checkpointed_iteration.txt

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export NVTE_FLASH_ATTN=1; export NVTE_FUSED_ATTN=1; export NVTE_FUSED_MLP=1
export NCCL_NET_GDR_LEVEL=PHB; export NCCL_CROSS_NIC=1; export NCCL_BUFFSIZE=2097152
MASTER_ADDR=$(hostname); MASTER_PORT=25679

TRAINING_CMD="torchrun \
    --nproc-per-node $SLURM_GPUS_PER_NODE --nnodes $SLURM_NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d \
    --max_restarts 0 --tee 3 \
    $MEGATRON_LM_DIR/pretrain_gpt.py \
    --transformer-impl transformer_engine --use-precision-aware-optimizer \
    --main-grads-dtype bf16 --fp8-format hybrid --fp8-margin 0 \
    --fp8-amax-history-len 1024 --fp8-amax-compute-algo max \
    --num-layers $NUM_LAYERS --hidden-size $HIDDEN --ffn-hidden-size $FFN \
    --num-attention-heads $HEADS --group-query-attention --num-query-groups $KV_HEADS \
    --max-position-embeddings $SEQ_LEN --position-embedding-type rope \
    --normalization RMSNorm --swiglu --untie-embeddings-and-output-weights \
    --seq-length $SEQ_LEN --use-mcore-models \
    --micro-batch-size $MBS --global-batch-size $GBS \
    --train-iters $EVAL_ITER --eval-interval $EVAL_ITER --eval-iters 32 \
    --cross-entropy-loss-fusion --disable-bias-linear --optimizer adam \
    --dataloader-type single --no-check-for-nan-in-loss-and-grad \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
    --lr $LR --min-lr $MIN_LR --lr-decay-style cosine \
    --lr-decay-iters $EVAL_ITER --lr-warmup-iters 100 \
    --seed 42 --init-method-std 0.02 --bf16 \
    --tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --log-throughput --log-progress \
    --tensorboard-dir $TENSORBOARD_DIR \
    --load $EVAL_CKPT \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $WORKDIR/data/gpt2-vocab.json \
    --merge-file $WORKDIR/data/gpt2-merges.txt \
    --data-path $DATA_PREFIX --data-cache-path $DATASET_CACHE_DIR \
    --split 99,1,0 --num-workers 16"

export WANDB_MODE=disabled
echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "numactl --membind=0-3 $TRAINING_CMD"

rm -rf $EVAL_CKPT
echo "END TIME: $(date)"
