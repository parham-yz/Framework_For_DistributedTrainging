#!/usr/bin/env python3
"""
Tool: parse_reports.py
Purpose: Scan the `reports/` directory for CNN-on-MNIST runs and compute rounds to convergence.
Generates a Markdown summary table at llms_tools/cnn_MNIST_summary.md.

Usage:
  python3 llms_tools/parse_reports.py [--acc_threshold 0.98] [--loss_threshold 0.05]
"""
import os
import re
import json
import argparse

def get_paths():
    """Return absolute paths for reports dir and output MD file."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    reports_dir = os.path.join(repo_root, 'reports')
    output_md = os.path.join(os.path.dirname(__file__), 'cnn_MNIST_summary.md')
    return reports_dir, output_md

def parse_report(path, acc_threshold, loss_threshold):
    """
    Parse one report file. Return (K, rounds_to_conv) if CNN-on-MNIST run, else None.
    Convergence: first epoch where acc >= acc_threshold or loss <= loss_threshold.
    NOTE: Adjust `metric_line_pattern` to match the log format of your reports.
    """
    with open(path) as f:
        lines = f.readlines()
    # Extract hyperparameters JSON
    H = None
    for line in lines:
        if line.startswith('Hyperparameters:'):
            _, json_str = line.split(':', 1)
            H = json.loads(json_str)
            break
    if not H:
        return None
    model = H.get('model')
    dataset = H.get('dataset_name') or H.get('dataset')
    if model not in ('cnn', 'cnn_ensemble') or dataset.lower() != 'mnist':
        return None
    K = H.get('K')
    # Scan for lines reporting Accuracy; treat each as one measurement epoch
    acc_pattern = re.compile(r'Accuracy\s*=\s*([0-9\.]+)')
    epoch_counter = 0
    best_acc = -1.0
    best_epoch = None
    for line in lines:
        m = acc_pattern.search(line)
        if not m:
            continue
        epoch_counter += 1
        acc = float(m.group(1))
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch_counter
    # Return K, step_size, best accuracy, and epoch at which best accuracy occurred
    step_size = H.get('step_size')
    return (K, step_size, best_acc, best_epoch)

def main():
    parser = argparse.ArgumentParser(description='Parse CNN-MNIST reports')
    parser.add_argument('--acc_threshold', type=float, default=0.98,
                        help='Accuracy threshold for convergence')
    parser.add_argument('--loss_threshold', type=float, default=None,
                        help='Loss threshold for convergence')
    args = parser.parse_args()

    reports_dir, output_md = get_paths()
    # results[K] = dict mapping step_size -> best_epoch
    results = {}
    step_sizes = set()
    if not os.path.isdir(reports_dir):
        print(f'Reports directory not found: {reports_dir}')
        return
    for fn in os.listdir(reports_dir):
        if not fn.startswith('R') or not fn.endswith('.txt'):
            continue
        path = os.path.join(reports_dir, fn)
        parsed = parse_report(path, args.acc_threshold, args.loss_threshold)
        if parsed:
            K, step, best_acc, best_epoch = parsed
            if best_epoch is not None:
                results.setdefault(K, {})[step] = best_epoch
                step_sizes.add(step)
    # Compute best (min) epoch per K across all step sizes
    summary = {K: min(epochs.values()) for K, epochs in results.items()}
    # Prepare sorted list of all step sizes
    step_list = sorted(step_sizes)
    # Write Markdown table with one column per step size and a Best column
    header_cols = ['K', 'Best'] + [str(s) for s in step_list]
    sep_cols = ['---'] * len(header_cols)
    with open(output_md, 'w') as out:
        out.write('# CNN on MNIST: Rounds to Convergence\n\n')
        out.write('| ' + ' | '.join(header_cols) + ' |\n')
        out.write('| ' + ' | '.join(sep_cols) + ' |\n')
        for K in sorted(summary):
            row = [str(K), str(summary[K])]
            epochs = results.get(K, {})
            for s in step_list:
                row.append(str(epochs.get(s, '')))
            out.write('| ' + ' | '.join(row) + ' |\n')
    print(f'Summary written: {output_md}')

if __name__ == '__main__':
    main()