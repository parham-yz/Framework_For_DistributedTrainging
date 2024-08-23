import os
import json
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import threading
import logging
from queue import Queue
import utils

# Setup logging
logging.basicConfig(filename='process_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_progress_tracker():
    logging.info("Starting initialization of progress tracker.")
    # Clear the progress_log.txt
    if os.path.exists('process_log.txt'):
        with open('process_log.txt', 'w') as log_file:
            log_file.write("")  # Clear the contents of the file
        logging.info("Cleared process_log.txt")
    
    # Remove the old progress tracker
    if os.path.exists('progress_tracker.json'):
        os.remove('progress_tracker.json')
        logging.info("Removed old progress_tracker.json")
        
    # Initialize the progress tracker
    with open('progress_tracker.json', 'w') as file:
        json.dump({}, file)
        logging.info("Initialized new progress_tracker.json")
    logging.info("Finished initialization of progress tracker.")

def launch_slave_process(step_size, K, epochs, gpu_queue):
    logging.info("Starting to launch a slave process.")
    logging.debug(f"Parameters received - step_size: {step_size}, K: {K}, epochs: {epochs}")
    
    log_filename = f"saved_logs/lr={step_size}_K={K}.json"
    logging.debug(f"Generated log filename: {log_filename}")
    
    if os.path.exists(log_filename):
        logging.info(f"Log file {log_filename} already exists. Skipping process for step_size={step_size} and K={K}.")
        logging.debug(f"Returning None, 0, {epochs} as no process is launched.")
        return None, 0, epochs  # No process launched, all epochs are skipped

    gpu_id = gpu_queue.get()  # Get an available GPU ID
    logging.debug(f"Acquired GPU ID: {gpu_id} from the queue.")
    
    process = subprocess.Popen(['python3', 'dol1.py', str(step_size), str(K), str(epochs), str(gpu_id)])
    logging.info(f"Launched process with step_size={step_size}, K={K}, epochs={epochs}, gpu_id={gpu_id}")
    logging.debug(f"Process details - PID: {process.pid}, GPU ID: {gpu_id}")
    
    logging.debug(f"Returning process, {epochs}, 0 as the process is successfully launched.")
    return (process, gpu_id), epochs, 0  # Return the process, total epochs, and skipped epochs

def update_progress_bar(total_epochs):
    with open('progress_tracker.json', 'r') as file:
        data = json.load(file)
    
    total_done = sum(data.values())
    progress = total_done / total_epochs
    progress_bar_length = 40
    num_hashes = int(progress * progress_bar_length)
    progress_bar = '#' * num_hashes + '-' * (progress_bar_length - num_hashes)
    
    # Print progress bar
    print(f"[{progress_bar}] {total_done}/{total_epochs} epochs completed ({progress*100:.2f}%)", end='\r')
    
    return total_done, progress

def check_and_reassign_gpus(processes, gpu_queue, hyperparameters):
    for process, gpu_id in processes:
        if process.poll() is not None:  # Process has completed
            gpu_queue.put(gpu_id)  # Reassign GPU to the queue
            processes.remove((process, gpu_id))
            logging.info(f"Process {process.pid} completed. Reassigned GPU {gpu_id} to the queue.")
    
    # Launch new processes if GPUs are available
    while not gpu_queue.empty():
        gpu_id = gpu_queue.get()
        for step_size, K, epochs in hyperparameters:
            log_filename = f"saved_logs/lr={step_size}_K={K}.json"
            if not os.path.exists(log_filename):
                process = subprocess.Popen(['python3', 'dol1.py', str(step_size), str(K), str(epochs), str(gpu_id)])
                processes.append((process, gpu_id))
                logging.info(f"Launched new process with step_size={step_size}, K={K}, epochs={epochs}, gpu_id={gpu_id}")
                break

def admin(total_epochs, processes, gpu_queue, hyperparameters):
    logging.info("Starting to monitor progress.")
    if total_epochs == 0:
        logging.info("No epochs to monitor. Exiting admin.")
        return
    last_logged_progress = 0  # Initialize last logged progress to 0

    while True:
        logging.info("Started monitoring loop.")
        
        total_done, progress = update_progress_bar(total_epochs)
        
        # Print GPU status
        gpu_status = []
        for gpu_id in range(8):
            running_process = next((p for p, g in processes if g == gpu_id and p.poll() is None), None)
            if running_process:
                gpu_status.append(f"{gpu_id}:{running_process.pid}")
            else:
                gpu_status.append(f"{gpu_id}:-")
        print(f"GPU Status: {' '.join(gpu_status)}", end='\r')

        # Log progress every 5%

        running_processes = sum(1 for p, _ in processes if p.poll() is None)
        logging.info(f"{total_done}/{total_epochs} epochs completed ({progress*100:.2f}%) - Running processes: {running_processes}")
        last_logged_progress += 5

        # Check for completed processes and reassign GPUs
        check_and_reassign_gpus(processes, gpu_queue, hyperparameters)

        if total_done >= total_epochs:
            logging.info("All epochs completed.")
            break
        
        time.sleep(3)
    logging.info("Finished monitoring progress.")

def evaluate_results(K_values, step_sizes):
    logging.info("Starting evaluation of results.")
    best_results = {}
    
    for K in K_values:
        best_step_size = None
        best_min_loss = float('inf')
        
        for step_size in step_sizes:
            log_filename = f"saved_logs/lr={step_size}_K={K}.json"
            if os.path.exists(log_filename):
                with open(log_filename, 'r') as f:
                    log_data = json.load(f)
                    min_loss = min(log_data['loss_log'])
                    
                    if min_loss < best_min_loss:
                        best_min_loss = min_loss
                        best_step_size = step_size

        best_results[K] = best_step_size
        logging.info(f"For K={K}, the best step size is {best_step_size} with min loss {best_min_loss}")
        print(f"For K={K}, the best step size is {best_step_size} with min loss {best_min_loss}")

    logging.info("Finished evaluation of results.")
    return best_results

def plot_results(K_values, step_sizes):
    logging.info("Starting to plot results.")
    best_step_sizes = {}
    
    for K in K_values:
        best_step_size = None
        best_accuracy = float('inf')
        
        for step_size in step_sizes:
            log_filename = f"saved_logs/lr={step_size}_K={K}.json"
            if os.path.exists(log_filename):
                with open(log_filename, 'r') as f:
                    log_data = json.load(f)
                    loss_log = log_data.get('loss_log', [])
                    if loss_log:
                        min_accuracy = min(loss_log)
                        if min_accuracy < best_accuracy:
                            best_accuracy = min_accuracy
                            best_step_size = step_size
        
        best_step_sizes[K] = best_step_size

    plt.figure(figsize=(12, 8))
    
    for K, step_size in best_step_sizes.items():
        log_filename = f"saved_logs/lr={step_size}_K={K}.json"
        if os.path.exists(log_filename):
            with open(log_filename, 'r') as f:
                log_data = json.load(f)
                plt.plot(np.log(log_data['loss_log']), label=f'K={K}, step_size={step_size}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Comparison of Loss vs Epochs for Different K Values')
    plt.legend()
    plt.savefig('comparison_loss_vs_epochs.png')
    plt.close()
    logging.info("Saved plot comparison_loss_vs_epochs.png")
    logging.info("Finished plotting results.")

def kill_child_processes(parent_pid):
    logging.info("Starting to kill child processes.")
    try:
        child_processes = subprocess.check_output(['pgrep', '-P', str(parent_pid)]).decode().split()
        for pid in child_processes:
            os.kill(int(pid), signal.SIGTERM)
            logging.info(f"Killed child process {pid}")
    except Exception as e:
        print(f"Error killing child processes: {e}")
        logging.error(f"Error killing child processes: {e}")
    logging.info("Finished killing child processes.")

def signal_handler(sig, frame):
        print("\n\n\n>>>>>Interrupt received, killing child processes...<<<<<<<<\n\n\n")
        logging.warning("Interrupt received, killing child processes.")
        kill_child_processes(os.getpid())
        sys.exit(0)

if __name__ == "__main__":
    logging.info("Starting master process.")
    signal.signal(signal.SIGINT, signal_handler)
    step_sizes = np.logspace(-8, -4, 4)
    K_values = [1, 2, 5, 10]
    epochs_per_process = 20000

    initialize_progress_tracker()

    
    utils.clear_gpu_memory()
    gpu_queue = Queue()
    for gpu_id in range(8):
        gpu_queue.put(gpu_id)

    processes = []
    total_epochs = 0
    hyperparameters = [(step_size, K, epochs_per_process) for step_size in step_sizes for K in K_values]

    for step_size, K, epochs in hyperparameters:
        process_info, total, skipped = launch_slave_process(step_size, K, epochs, gpu_queue)
        if process_info:
            processes.append(process_info)
        total_epochs += total
    utils.clear_gpu_memory()
    admin(total_epochs, processes, gpu_queue, hyperparameters)
    
    for process, _ in processes:
        process.wait()
    
    best_results = evaluate_results(K_values, step_sizes)
    
    plot_results(K_values, step_sizes)

    print("Master process completed.")
    logging.info("Master process completed.")