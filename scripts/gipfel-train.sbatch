#!/bin/bash
# Training runs for all key experiments across 30-min, 1-hr, and 2-hr budgets.
# Submit all:       sbatch --array=0-7 gipfel-train.sbatch
# Single run:       sbatch --array=2   gipfel-train.sbatch   # 760m 2hr
#
# Array index → experiment:
#   0  760m  lr=4e-4  1819s  30-min best
#   1  760m           3638s  1-hr
#   2  760m           7277s  2-hr best overall
#   3  1.5b           2293s  1-hr
#   4  1.5b           4587s  2-hr
#   5  3b             2819s  1-hr
#   6  3b             5639s  2-hr
#   7  8b            12298s  2-hr
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --job-name=gipfel-train
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

case ${SLURM_ARRAY_TASK_ID:-2} in
  0) MODEL=760m; NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4; MBS=4; GBS=512;  TP=1; PP=1; STEPS=1819;  LR=4e-4; MIN_LR=4e-5; EXP_SUFFIX="-lr4e-4" ;;
  1) MODEL=760m; NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4; MBS=4; GBS=512;  TP=1; PP=1; STEPS=3638;  LR=2e-4; MIN_LR=2e-5; EXP_SUFFIX="" ;;
  2) MODEL=760m; NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4; MBS=4; GBS=512;  TP=1; PP=1; STEPS=7277;  LR=2e-4; MIN_LR=2e-5; EXP_SUFFIX="" ;;
  3) MODEL=1.5b; NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4; MBS=4; GBS=256;  TP=2; PP=1; STEPS=2293;  LR=2e-4; MIN_LR=2e-5; EXP_SUFFIX="" ;;
  4) MODEL=1.5b; NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4; MBS=4; GBS=256;  TP=2; PP=1; STEPS=4587;  LR=2e-4; MIN_LR=2e-5; EXP_SUFFIX="" ;;
  5) MODEL=3b;   NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8; MBS=4; GBS=128;  TP=2; PP=2; STEPS=2819;  LR=1.5e-4; MIN_LR=1.5e-5; EXP_SUFFIX="" ;;
  6) MODEL=3b;   NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8; MBS=4; GBS=128;  TP=2; PP=2; STEPS=5639;  LR=1.5e-4; MIN_LR=1.5e-5; EXP_SUFFIX="" ;;
  7) MODEL=8b;   NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8; MBS=2; GBS=32;   TP=4; PP=2; STEPS=12298; LR=1e-4;   MIN_LR=1e-5;   EXP_SUFFIX="" ;;
esac

WARMUP=$((STEPS / 20)); [ "$WARMUP" -lt 100 ] && WARMUP=100

PROJECT_NAME=gipfelsturm
EXP_NAME=train-${MODEL}-${SLURM_NNODES}n${EXP_SUFFIX}
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR $LOG_DIR/checkpoints

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
MASTER_ADDR=$(hostname); MASTER_PORT=25678

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
    --train-iters $STEPS --log-interval 1 \
    --eval-interval $STEPS --eval-iters 10 \
    --cross-entropy-loss-fusion --disable-bias-linear --optimizer adam \
    --dataloader-type single --no-check-for-nan-in-loss-and-grad \
    --manual-gc --manual-gc-interval 50 --exit-signal-handler \
    --recompute-granularity selective \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
    --lr $LR --min-lr $MIN_LR --lr-decay-style cosine \
    --lr-decay-iters $STEPS --lr-warmup-iters $WARMUP \
    --seed 42 --init-method-std 0.02 --bf16 \
    --tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --log-throughput --log-progress \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-timers-to-tensorboard --log-memory-to-tensorboard \
    --save $LOG_DIR/checkpoints --save-interval 500 \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $WORKDIR/data/gpt2-vocab.json \
    --merge-file $WORKDIR/data/gpt2-merges.txt \
    --data-path $DATA_PREFIX --data-cache-path $DATASET_CACHE_DIR \
    --split 99,1,0 --num-workers 16"

if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
