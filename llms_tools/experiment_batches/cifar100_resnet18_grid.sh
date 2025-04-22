#!/usr/bin/env bash
# Grid experiments with dynamic batching based on NUM_GPUS and TASKSPERCORE
# Batches are of size NUM_GPUS * TASKSPERCORE
# Each batch launches experiments in parallel and waits for their completion

set -e

# Hyperparameters
KS=(1 2 5 10 50)       # Using the original KS for consistency unless specified otherwise
STEPS=(0.001 0.0001 0.00001 0.000001 0.0000001) # Reduced to 3 step sizes as per the example
NUM_GPUS=4             # Total number of GPUs available
TASKSPERCORE=2         # Number of tasks to assign to each GPU per batch

BATCH_CHUNK_SIZE=$(( NUM_GPUS * TASKSPERCORE ))
ROUNDS=10000
BATCH_SIZE=128
REPORT_RATE=20

# --- Generate the list of all experiment configurations ---
EXPERIMENT_LIST=()
for K in "${KS[@]}"; do
  for STEP in "${STEPS[@]}"; do
    EXPERIMENT_LIST+=("${K}_${STEP}")
  done
done

TOTAL_EXPERIMENTS=${#EXPERIMENT_LIST[@]}
echo "Total experiments to run: $TOTAL_EXPERIMENTS"
echo "Batch chunk size: $BATCH_CHUNK_SIZE"
echo "Number of batches: $(( (TOTAL_EXPERIMENTS + BATCH_CHUNK_SIZE - 1) / BATCH_CHUNK_SIZE ))" # Ceiling division

echo "Cleaning up reports directory..."
mkdir -p reports
rm -rf reports/*

echo "Starting CIFAR-100/ResNet18 grid with dynamic batching..."

COUNTER=0 # Overall experiment counter

# Loop through the experiment list in chunks
for (( i=0; i<TOTAL_EXPERIMENTS; i+=BATCH_CHUNK_SIZE )); do
  CURRENT_BATCH_EXPERIMENTS=("${EXPERIMENT_LIST[@]:i:BATCH_CHUNK_SIZE}")
  BATCH_SIZE_ACTUAL=${#CURRENT_BATCH_EXPERIMENTS[@]}
  BATCH_START_INDEX=$i

  echo "-- Starting batch $(( i / BATCH_CHUNK_SIZE + 1 )) (experiments $BATCH_START_INDEX to $(( BATCH_START_INDEX + BATCH_SIZE_ACTUAL - 1 )) out of $TOTAL_EXPERIMENTS) --"

  PIDS=()
  batch_task_counter=0 # Counter for tasks within the current batch

  # Loop through experiments in the current batch chunk
  for experiment_str in "${CURRENT_BATCH_EXPERIMENTS[@]}"; do
    # Split the string into K and STEP
    IFS=_ read -r current_k current_step <<< "$experiment_str"

    MODE="blockwise_sequential"
    if [ "$current_k" -eq 1 ]; then MODE="entire"; fi

    # Calculate GPU based on batch_task_counter and TASKSPERCORE
    GPU=$(( (batch_task_counter / TASKSPERCORE) % NUM_GPUS ))

    COUNTER=$(( COUNTER + 1 )) # Increment overall counter
    echo "[$COUNTER/$TOTAL_EXPERIMENTS] Launching K=$current_k, step_size=$current_step on GPU $GPU (Task index in batch: $batch_task_counter)"

    python3 -m src.dol1 \
      --model ResNet18 \
      --dataset_name cifar100 \
      --training_mode $MODE \
      --step_size "$current_step" \
      --batch_size "$BATCH_SIZE" \
      --rounds "$ROUNDS" \
      --K "$current_k" \
      --cuda_core "$GPU" \
      --report_sampling_rate "$REPORT_RATE" &

    PIDS+=("$!")
    batch_task_counter=$(( batch_task_counter + 1 )) # Increment batch task counter

  done

  # Wait for all processes in the current batch chunk to finish
  echo "Waiting for batch to complete ($BATCH_SIZE_ACTUAL tasks)..."
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  echo "Batch complete."

done

echo "All $TOTAL_EXPERIMENTS CIFAR-100/ResNet18 experiments completed."
echo "Reports are in the reports/ directory."