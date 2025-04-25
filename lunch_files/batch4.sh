#!/usr/bin/env bash
# Batch 4: K=90 (blockwise_sequential) across GPUs 0 and 1

child_pids=()
cleanup() {
  echo "\nTerminating Batch 4 processes..."
  for pid in "${child_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
    fi
  done
  exit 1
}
trap cleanup SIGINT SIGTERM

echo "Starting Batch 4 experiments: K=90"

# Common parameters
ROUNDS=5000
BATCH_SIZE=128
REPORT_RATE=20
STEP_SIZES=(0.001 0.0001 0.00001)

# Distribute K=90 experiments across GPUs for balance
for STEP in "${STEP_SIZES[@]}"; do
  # Assign the middle step to GPU1, others to GPU0
  if [[ "$STEP" == "0.0001" ]]; then
    CUDA=1
  else
    CUDA=0
  fi
  echo "Launching K=90, step_size=$STEP on GPU $CUDA"
  python3 -m src.dol1 \
    --model cnn \
    --dataset_name mnist \
    --training_mode blockwise_sequential \
    --step_size $STEP \
    --batch_size $BATCH_SIZE \
    --rounds $ROUNDS \
    --K 90 \
    --cuda_core $CUDA \
    --report_sampling_rate $REPORT_RATE &
  child_pids+=("$!")
done

echo "Launched ${#child_pids[@]} processes for Batch 4."
wait
echo "Batch 4 complete."