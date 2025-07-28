import os
import re
import matplotlib.pyplot as plt

# Directories
input_folder = './metrics_tests_text/'
output_folder = './metrics_tests_sumvar_line_graphs/'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Line colours for Summation and Deviation
line_colours = {
    "Summation": "yellow",
    "Deviation": "red",
}


# Function to extract Summation/Deviation times
def extract_data(file_path):
    dataset, strategy, target = None, None, None
    thresholds = []
    summations = []
    deviations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract metadata
    for line in lines:
        if "Strategy:" in line:
            strategy = line.split("Strategy:")[1].strip().split()[0]  # Extract first word (Global, Household, Random)
        elif "Dataset:" in line:
            dataset = line.split("Dataset:")[1].strip()
        elif "Target:" in line:
            target = line.split("Target:")[1].strip()

    # Store last known threshold
    last_threshold = None

    for line in lines:
        # Extract threshold values
        threshold_match = re.match(r'threshold = ([0-9.]+)', line.strip())
        print(threshold_match)

        # If we find a threshold, store it (may not have summation/deviation yet)
        if threshold_match:
            last_threshold = float(threshold_match.group(1))
            thresholds.append(last_threshold)
            print(thresholds)

            # If threshold is 0.0 or 0.1, set summation/deviation to 0.0
            if last_threshold in [0.0, 0.1]:
                summations.append(0.0)
                deviations.append(0.0)

        # Extract Summation/Deviation times
        time_match = re.search(r'Summation/Deviation time .*:\s*([-+]?[0-9]*\.?[0-9]+)ms,\s*([-+]?[0-9]*\.?[0-9]+)ms',
                               line)

        if time_match:
            # Only add times if a threshold has already been found
            if last_threshold is not None:
                summations.append(float(time_match.group(1)))  # Summation Time
                deviations.append(float(time_match.group(2)))  # Deviation Time
                print(summations, deviations)

    return dataset, strategy, target, thresholds, summations, deviations


# Store data: { (dataset, strategy, target): (thresholds, summations, deviations) }
data_store = {}

# Process all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)

        # Extract data
        dataset, strategy, target, thresholds, summations, deviations = extract_data(file_path)

        # Ensure all lists have valid values and the same length
        if not (dataset and strategy and target):
            print(f"Skipping {filename} due to missing or mismatched data.")
            continue

        key = (dataset, strategy, target)
        data_store[key] = (thresholds, summations, deviations)


# Function to generate Summation & Deviation graphs
def plot_sumvar_graphs(data_dict):
    """
    Generates graphs where:
    - X-axis = Encryption Ratio (Threshold)
    - Y-axis = Time (ms)
    - Two lines: Summation (yellow) & Deviation (red)
    """
    for (dataset, strategy, target), (thresholds, summations, deviations) in data_dict.items():
        plt.figure(figsize=(10, 6))

        # Ensure values are sorted for correct plotting
        sorted_data = sorted(zip(thresholds, summations, deviations))
        sorted_thresholds, sorted_summations, sorted_deviations = zip(*sorted_data)

        # Plot Summation and Deviation
        plt.plot(sorted_thresholds, sorted_summations, marker='o', linestyle='-', color=line_colours["Summation"],
                 label="Summation Time")
        plt.plot(sorted_thresholds, sorted_deviations, marker='s', linestyle='--', color=line_colours["Deviation"],
                 label="Deviation Time")

        # Labels and title
        plt.xlabel("Encryption Ratio (Threshold)")
        plt.ylabel("Time (ms)")
        title = f"{dataset}_{strategy}_{target}_Summation-Deviation"
        plt.title(title.replace(" ", "-"))

        # Legend and grid
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save graph
        filename_out = f"{dataset}_{strategy}_{target}_Summation-Deviation.png"
        output_path = os.path.join(output_folder, filename_out)
        plt.savefig(output_path)
        plt.close()


# Generate all Summation/Deviation graphs
plot_sumvar_graphs(data_store)

print("Summation & Deviation graphs successfully generated.")
