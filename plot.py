import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from termcolor import colored
import subprocess
import sys
import time
import threading

def plot_results(K_values, step_sizes, dataset,beta=0):
    # Ensure the figures directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Use getBestStepsizes to get the best step sizes
    best_step_sizes = getBestStepsizes(K_values, step_sizes, dataset)

    plt.figure(figsize=(12, 8))


    for K, step_size in best_step_sizes.items():
        log_filename = f"saved_logs/lr={step_size}_K={K}_dataset={dataset}.json"
        if os.path.exists(log_filename):
            with open(log_filename, 'r') as f:
                log_data = json.load(f)
                loss_log = log_data['loss_log']
                
                # Compute running average
                running_avg = []
                avg = loss_log[0]
                for i, loss in enumerate(loss_log):
                    avg = beta * avg + (1 - beta) * loss
                    running_avg.append(avg)
                
                iterations = np.arange(len(running_avg)) * 200
                plt.plot(iterations, np.log10(running_avg), label=f'K={K}, step_size={step_size}')
    
    plt.xlabel('Rounds')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Comparison of Loss vs Iterations for Different K Values (Dataset: {dataset})')
    plt.legend()
    plt.savefig(f'figures/comparison_loss_vs_iterations_{dataset}.png')
    plt.close()

def plot_results_for_specific_K(K, step_sizes, dataset):
    # Ensure the figures directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    plt.figure(figsize=(12, 8))
    
    for step_size in step_sizes:
        log_filename = f"saved_logs/lr={step_size}_K={K}_dataset={dataset}.json"
        if os.path.exists(log_filename):
            with open(log_filename, 'r') as f:
                log_data = json.load(f)
                plt.plot(np.log(log_data['loss_log']), label=f'step_size={step_size}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Comparison of Loss vs Epochs for K={K} (Dataset: {dataset})')
    plt.legend()
    plt.savefig(f'figures/comparison_loss_vs_epochs_K={K}_{dataset}.png')
    plt.close()

def getBestStepsizes(K_values, step_sizes, dataset):
    best_results = {}
    
    for K in K_values:
        best_step_size = None
        best_min_loss = float('inf')
        
        for step_size in step_sizes:
            log_filename = f"saved_logs/lr={step_size}_K={K}_dataset={dataset}.json"
            if os.path.exists(log_filename):
                with open(log_filename, 'r') as f:
                    log_data = json.load(f)
                    min_loss = min(log_data['loss_log'])
                    
                    if min_loss < best_min_loss:
                        best_min_loss = min_loss
                        best_step_size = step_size

        if best_step_size is not None:
            best_results[K] = best_step_size
            print(f"For K={K}, the best step size is {best_step_size} with min loss {best_min_loss}")

    return best_results


def extract_K_and_stepsizes(directory):
    K_values = set()
    step_sizes = set()
    datasets = set()
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            parts = filename.split('_')
            if len(parts) >= 3:
                step_size = parts[0].split('=')[1]
                K = int(parts[1].split('=')[1])
                dataset = parts[2].split('=')[1].split('.')[0]
                step_sizes.add(step_size)
                K_values.add(K)
                datasets.add(dataset)
    
    return sorted(K_values), sorted(step_sizes, key=float), sorted(datasets)

def get_free_gpu_cores(reports_folder='reports'):
    total_cores = set()
    used_cores = set()
    
    for filename in os.listdir(reports_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(reports_folder, filename), 'r') as file:
                lines = file.readlines()
                status_line = None
                hyperparameters_line = None
                
                for line in lines:
                    if line.startswith('Status:'):
                        status_line = line.strip()
                    elif line.startswith('Hyperparameters:'):
                        hyperparameters_line = line.strip()
                
                if status_line and hyperparameters_line:
                    status = status_line.split(': ')[1]
                    hyperparameters = json.loads(hyperparameters_line[len("Hyperparameters: "):])
                    cuda_core = hyperparameters.get('cuda_core')
                    
                    if cuda_core is not None:
                        total_cores.add(cuda_core)
                        if status != 'dead':
                            used_cores.add(cuda_core)
    
    free_cores = total_cores - used_cores
    print(f"Total GPU Cores: {sorted(total_cores)}")
    print(f"Free GPU Cores: {sorted(free_cores)}")
    return sorted(free_cores)

def monitor_reports(reports_folder='reports'):
    def read_report(filename):
        with open(os.path.join(reports_folder, filename), 'r') as file:
            lines = file.readlines()
            status_line = None
            progress_lines = []
            
            for line in lines:
                if line.startswith('Status:'):
                    status_line = line.strip()
                elif line.startswith('['):
                    progress_lines.append(line.strip())
            
            return status_line, progress_lines

    def parse_progress_line(line):
        parts = line.split(' ')
        iteration_index = parts.index("Iteration")
        iteration = int(parts[iteration_index + 1].strip(':'))
        fid_loss = float(parts[parts.index('FID') + 3])
        return iteration, fid_loss

    def monitor():
        while not exit_monitoring:
            alive_reports = []
            for filename in os.listdir(reports_folder):
                if filename.endswith('.txt'):
                    status_line, progress_lines = read_report(filename)
                    if status_line and 'alive' in status_line:
                        alive_reports.append((filename, progress_lines))
            
            os.system('clear')  # Clear the console
            print("Monitoring alive reports:")
            for filename, progress_lines in alive_reports:
                print(f"\nReport: {filename}")
                
                if progress_lines:
                    last_line = progress_lines[-1]
                    iteration, fid_loss = parse_progress_line(last_line)
                    first_line = progress_lines[0]
                    parts = first_line.split(', ')
                    K = int(parts[0].split(': ')[1])
                    rounds = int(parts[4].split(': ')[1])
                    total_iterations = rounds * K  # Assuming total iterations are 20000 times K value
                    progress = iteration / total_iterations
                    print(f"Iteration: {iteration}, FID Loss: {fid_loss}")
                    print(f"Progress: [{'#' * int(progress * 50)}{'.' * (50 - int(progress * 50))}] {progress * 100:.2f}%")
                else:
                    print("No progress lines found.")
            
            time.sleep(5)  # Update every 5 seconds

    exit_monitoring = False
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    print("Press 'x' to exit monitoring mode.")
    while True:
        if input().strip().lower() == 'x':
            exit_monitoring = True
            monitor_thread.join()
            break



def save_and_export_logs():
    import shutil
    from datetime import datetime
    import zipfile

    # Create the exports directory if it doesn't exist
    if not os.path.exists('exports'):
        os.makedirs('exports')

    # Create a timestamp for the export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join('exports', f"export_{timestamp}.zip")

    # Create a ZIP file
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add saved_logs directory to the ZIP file
        for root, dirs, files in os.walk('saved_logs'):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.join('saved_logs', os.path.relpath(file_path, 'saved_logs')))

        # Add figures directory to the ZIP file
        for root, dirs, files in os.walk('figures'):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.join('figures', os.path.relpath(file_path, 'figures')))

        # Generate README file with the output of option_2
        readme_path = 'README.txt'
        with open(readme_path, 'w') as readme_file:
            original_stdout = sys.stdout
            sys.stdout = readme_file
            option_2()
            sys.stdout = original_stdout

        # Add README file to the ZIP file
        zipf.write(readme_path, 'README.txt')

    # Remove the README file after adding it to the ZIP
    os.remove(readme_path)

    print(f"Logs and data have been exported to {zip_filename}")


if __name__ == "__main__":
    import sys

    def display_menu():
        print("Please choose an option:")
        print("1. Plot results")
        print("2. List all trained hyperparameters")
        print("3. Monitor reports")
        print("4. Free Gpu Cores")
        print("5. Save and export logs")

    def option_1():
        # Extract the K values, step sizes, and datasets from the saved logs directory
        K_values, step_sizes, datasets = extract_K_and_stepsizes('saved_logs')
        
        # Ask for beta value
        beta = float(input("Please enter the beta value for running average computation (e.g., 0.9): ").strip())
        
        # Plot the results for each dataset
        for dataset in datasets:
            plot_results(K_values, step_sizes, dataset, beta=beta)
            for K in K_values:
                plot_results_for_specific_K(K, step_sizes, dataset)

    def option_2():
        hyperparameters = defaultdict(list)
        
        for filename in os.listdir('saved_logs'):
            if filename.endswith('.json'):
                parts = filename.split('_')
                if len(parts) >= 3:
                    step_size = parts[0].split('=')[1]
                    K = int(parts[1].split('=')[1])
                    dataset = parts[2].split('=')[1].split('.')[0]
                    hyperparameters[dataset].append((K, step_size))
        
        print("Listing all trained hyperparameters:")
        for dataset, params in hyperparameters.items():
            print(f"\nDataset: {dataset}")
            K_values = sorted(set(K for K, _ in params))
            step_sizes = sorted(set(step_size for _, step_size in params), key=float)
            
            print(f"K = {K_values}")
            print(f"Step Sizes = {step_sizes}")
            
            missing_combinations = []
            for K in K_values:
                for step_size in step_sizes:
                    if (K, step_size) not in params:
                        missing_combinations.append((K, step_size))
            
            if missing_combinations:
                print("\nMissing combinations:")
                for K, step_size in missing_combinations:
                    print(colored(f"  K={K}, Step Size={step_size}", 'red'))
        
        run_training = input("Do you want to run training for the missing combinations? (yes/no): ").strip().lower()
        if run_training in ['yes', 'y']:
            dataset_to_use = input("Please specify the dataset to use for the missing combinations: ").strip()
            free_cores = get_free_gpu_cores()
            
            if not free_cores:
                print("No free GPU cores available. Cannot run training.")
            else:
                for i, (K, step_size) in enumerate(missing_combinations):
                    if i < len(free_cores):
                        gpu_core = free_cores[i]
                        command = f"python3 dol1.py {step_size} {K} 20000 {dataset_to_use} {gpu_core}"
                        print(f"Running command: {command}")
                        subprocess.Popen(command, shell=True)
                    else:
                        print("Not enough free GPU cores to run all missing combinations.")
                        break

    def option_3():
        monitor_reports()

    def option_4():
        free_cores = get_free_gpu_cores()
        print(f"Free GPU cores: {free_cores}")

    def option_5():
        save_and_export_logs()

    options = {
        "1": option_1,
        "2": option_2,
        "3": option_3,
        "4": option_4,
        "5": option_5
    }

    display_menu()
    choice = input("Enter your choice (1-5): ")

    if choice in options:
        options[choice]()
    else:
        print("Invalid choice. Please select a valid option.")
