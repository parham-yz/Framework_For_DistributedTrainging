#!/bin/bash

# Define the number of instances
NUM_INSTANCES=2

# Define arrays for hyperparameters
declare -a stepsizes=("0.00005" "0.00005" "0.00005" "0.00005" "0.00005")
declare -a k_values=("60" "90" "5" "10" "20")
declare -a cuda_cores=(1 0 1 0 1)
declare -a training_modes=("blockwise_sequential" "blockwise_sequential" "blockwise_sequential" "blockwise_sequential" "blockwise_sequential")
declare -a n_workers=("1" "1" "1" "1" "1") # Added n_workers array

# Define global variables
ROUNDS="5000"
REPORT_SAMPLING_RATE="5"
DATASET="cifar100"
BATCH_SIZE="128"
MODEL="ResNet18" # Global variable for model type
COMMUNICATION_DELAY="0" # Added communication delay variable

# Array to keep track of child process PIDs
child_pids=()

# Cleanup function to terminate child processes
cleanup() {
    echo "Terminating child processes..."
    for pid in "${child_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating child process $pid"
            kill -TERM "$pid"
        fi
    done
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM signals and call the cleanup function
trap cleanup SIGINT SIGTERM

# Launch the instances in parallel
for ((i=0; i<$NUM_INSTANCES; i++))
do
    echo "Starting instance $i with stepsize ${stepsizes[$i]}, batch size $BATCH_SIZE, K ${k_values[$i]}, rounds $ROUNDS, dataset $DATASET, on CUDA core ${cuda_cores[$i]}, training mode ${training_modes[$i]}, model $MODEL, report sampling rate $REPORT_SAMPLING_RATE, communication delay $COMMUNICATION_DELAY, n_workers ${n_workers[$i]}"
    python3 -m src.dol1 --step_size "${stepsizes[$i]}" --batch_size "$BATCH_SIZE" --K "${k_values[$i]}" --rounds "$ROUNDS" --dataset_name "$DATASET" --cuda_core "${cuda_cores[$i]}" --training_mode "${training_modes[$i]}" --model "$MODEL" --report_sampling_rate "$REPORT_SAMPLING_RATE" --communication_delay "$COMMUNICATION_DELAY" --n_workers "${n_workers[$i]}" &
    # Save the PID of the background process
    child_pids+=($!)
done

# Wait for all background processes to complete
wait

echo "All instances completed."