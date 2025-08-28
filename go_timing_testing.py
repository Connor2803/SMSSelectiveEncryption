import subprocess
import time
import os
import pandas as pd

def get_files_by_pattern_os(pattern):
    matching_files = []
    for filename in os.listdir('.'):
        if os.path.isfile(filename) and filename.endswith(pattern.lstrip('*')):
            matching_files.append(filename)
    return matching_files


def run_go_program(go_file_path, output_csv_prefix, run_number, script_dir):
    output_csv_path = f"{output_csv_prefix}_{run_number}.csv"
    expected_output_path = os.path.join(script_dir, "WATER_household_level_encryption_ratio_selector", "ML_metrics_WATER.csv")

    command = ["go", "run", go_file_path, "1", "12"]
    print(f"\nStarting run:{run_number}...")
    start_time = time.time()
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=900, cwd=script_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Run {run_number} finished in {execution_time:.2f} seconds.")

        if os.path.exists(expected_output_path):
            os.rename(expected_output_path, output_csv_path)
            print(f"Saved output to {output_csv_path}")
            return execution_time, output_csv_path
        else:
            print(f"Error: Could not find the expected output file at {expected_output_path}")
            return None, None

    except FileNotFoundError:
        print("Error: 'go' command not found. Please ensure Go is installed and in your PATH.")
        return None, None
    except subprocess.CalledProcessError as e:
        print(f"Error running Go program on iteration {run_number}:")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}")
        print(f"Error Output:\n{e.stderr}")
        return None, None
    except subprocess.TimeoutExpired:
        print(f"Error: Go program timed out on iteration {run_number}.")
        return None, None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    go_file = "./WATER_household_level_encryption_ratio_selector/generate_household_level_metrics.go"
    output_prefix = "WATER_household_metrics_run"
    num_runs = 10
    execution_times = []
    generated_files = []

    for i in range(1, num_runs + 1):
        exec_time, output_file = run_go_program(go_file, output_prefix, i, script_dir)
        if exec_time is not None and output_file is not None:
            execution_times.append(exec_time)
            generated_files.append(output_file)
        else:
            print(f"Stopping script due to error in run {i}.")
            return

    if execution_times:
        average_time = sum(execution_times) / len(execution_times)
        print("\n***Execution Time Summary***")
        print(f"Average execution time over {num_runs} runs: {average_time:.2f} seconds")
        print(f"Fastest run: {min(execution_times):.2f} seconds")
        print(f"Slowest run: {max(execution_times):.2f} seconds")


if __name__ == "__main__":
    main()
