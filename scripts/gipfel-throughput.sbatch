#!/bin/bash
# Throughput benchmark for all model sizes.
# Submit all:      sbatch --array=0-5 gipfel-throughput.sbatch
# Single model:    sbatch --array=2   gipfel-throughput.sbatch   # 760m
#
# Array index → model:  0=125m  1=350m  2=760m  3=1.5b  4=3b  5=8b
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --job-name=gipfel-throughput
#SBATCH --output=logs/%x-%a-%j.log
#SBATCH --error=logs/%x-%a-%j.log
#SBATCH --nodes=4
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
TRAINING_STEPS=50

case ${SLURM_ARRAY_TASK_ID:-0} in
  0) MODEL=125m; NUM_LAYERS=12; HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4;  MBS=16; GBS=512;  TP=1; PP=1 ;;
  1) MODEL=350m; NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4;  MBS=8;  GBS=256;  TP=1; PP=1 ;;
  2) MODEL=760m; NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4;  MBS=4;  GBS=256;  TP=1; PP=1 ;;
  3) MODEL=1.5b; NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4;  MBS=4;  GBS=256;  TP=2; PP=1 ;;
  4) MODEL=3b;   NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8;  MBS=4;  GBS=128;  TP=2; PP=2 ;;
  5) MODEL=8b;   NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8;  MBS=2;  GBS=32;   TP=4; PP=2 ;;
esac

PROJECT_NAME=gipfelsturm
EXP_NAME=throughput-${MODEL}-${SLURM_NNODES}n
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
    --train-iters $TRAINING_STEPS --log-interval 1 \
    --eval-interval $TRAINING_STEPS --eval-iters 0 \
    --cross-entropy-loss-fusion --disable-bias-linear --optimizer adam \
    --dataloader-type single --no-check-for-nan-in-loss-and-grad \
    --manual-gc --manual-gc-interval 50 --recompute-granularity selective \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
    --lr 2e-4 --min-lr 2e-5 --lr-decay-style cosine \
    --lr-decay-iters $TRAINING_STEPS --lr-warmup-iters 10 \
    --seed 42 --init-method-std 0.02 --bf16 \
    --tensor-model-parallel-size $TP --pipeline-model-parallel-size $PP \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --log-throughput --log-progress \
    --save $LOG_DIR/checkpoints --save-interval $TRAINING_STEPS \
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

echo "END TIME: $(date)"
