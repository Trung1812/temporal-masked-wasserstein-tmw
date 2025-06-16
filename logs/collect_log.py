import re
import csv
from pathlib import Path

def parse_log_file(log_path):
    best_params = None
    test_accuracy = None
    inference_time = None
    dataset_name = None

    with open(log_path, 'r') as f:
        for line in f:
            # Dataset name
            if "Dataset:" in line:
                dataset_match = re.search(r"Dataset:\s*(\S+)", line)
                if dataset_match:
                    dataset_name = dataset_match.group(1)

            # Best params
            if "Best params" in line:
                match = re.search(r"w=(\d+), lambda=([\d.]+), train_acc=([\d.]+)", line)
                if match:
                    w = int(match.group(1))
                    lam = float(match.group(2))
                    train_acc = float(match.group(3))
                    best_params = {'w': w, 'lambda': lam, 'train_acc': train_acc}

            # Test accuracy
            if "Test accuracy" in line:
                acc_match = re.search(r"Test accuracy:\s*([\d.]+)", line)
                if acc_match:
                    test_accuracy = float(acc_match.group(1))

            # Inference time
            if "Inference time" in line:
                time_match = re.search(r"Inference time \(s\):\s*([\d.]+)", line)
                if time_match:
                    inference_time = float(time_match.group(1))

    return {
        "dataset": dataset_name or log_path.parent.name,
        "log_file": log_path.name,
        "w": best_params['w'] if best_params else None,
        "lambda": best_params['lambda'] if best_params else None,
        "train_acc": best_params['train_acc'] if best_params else None,
        "test_acc": test_accuracy,
        "inference_time": inference_time
    }

def extract_logs_to_csv(root_folder, output_csv):
    root = Path(root_folder)
    logs = list(root.rglob("*.log"))

    results = []
    for log in logs:
        result = parse_log_file(log)
        result["log_path"] = str(log.relative_to(root))
        results.append(result)

    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "log_path", "log_file", "w", "lambda", "train_acc", "test_acc", "inference_time"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Extracted {len(results)} log files to {output_csv}")

if __name__ == "__main__":
    extract_logs_to_csv(".", "./logging.csv")
