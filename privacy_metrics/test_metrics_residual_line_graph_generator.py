import os
import re
import matplotlib.pyplot as plt

# Directories
input_folder = './metrics_tests_text/'
output_folder = './metrics_tests_residual_line_graphs/'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Mapping for strategy names and colours
strategy_colours = {
    "Global": "red",
    "Household": "blue",
    "Random": "green"
}

# Function to extract data from a file
def extract_data(file_path):
    strategy, dataset, target = None, None, None
    thresholds = []
    residual_entropy = []
    residual_transition = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract metadata (Strategy, Dataset, Target)
    for line in lines:
        if "Strategy:" in line:
            strategy = line.split("Strategy:")[1].strip().split()[0]  # First word (Global, Household, Random)
        elif "Dataset:" in line:
            dataset = line.split("Dataset:")[1].strip()
        elif "Target:" in line:
            target = line.split("Target:")[1].strip()

    # Extract threshold and entropy/transition remain values
    for line in lines:
        match = re.match(r'threshold = ([0-9.]+), entropy/transition remain = ([0-9.-]+),([0-9.-]+)', line.strip())
        if match:
            thresholds.append(float(match.group(1)))
            residual_entropy.append(float(match.group(2)))
            residual_transition.append(float(match.group(3)))

    return strategy, dataset, target, thresholds, residual_entropy, residual_transition

# Organise data by dataset and household count
data_store = {}  # { (dataset, maxHouseholdsNumber, strategy): (thresholds, residual_entropy, residual_transition) }

# Process all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)

        # Extract data
        strategy, dataset, target, thresholds, residual_entropy, residual_transition = extract_data(file_path)

        if not (strategy and dataset and thresholds):
            print(f"Skipping {filename} due to missing data.")
            continue

        # Extract maxHouseholdsNumber from filename (last part of the split)
        maxHouseholdsNumber = filename.split('_')[-1].split('.')[0]

        key = (dataset, maxHouseholdsNumber, strategy)
        data_store[key] = (thresholds, residual_entropy, residual_transition)

# Function to generate and save line graphs
def plot_graphs(data_dict, target_type, value_index):
    """
    target_type: 'Entropy based' or 'Transition based'
    value_index: 0 for residual_entropy, 1 for residual_transition
    """
    grouped_data = {}  # To group data by (dataset)

    # Step 1: Group data by dataset, collecting strategies inside
    for (dataset, maxHouseholdsNumber, strategy), (thresholds, entropy_values, transition_values) in data_dict.items():
        if dataset not in grouped_data:
            grouped_data[dataset] = {}

        grouped_data[dataset][strategy] = (thresholds, entropy_values, transition_values)

    # Step 2: Iterate over grouped data
    for dataset, strategy_data in grouped_data.items():
        plt.figure(figsize=(10, 6))

        for strategy, values in strategy_data.items():
            thresholds, entropy_values, transition_values = values

            # Choose the correct Y-axis values
            if target_type == "Entropy based":
                y_values = entropy_values if value_index == 0 else transition_values
            else:
                y_values = transition_values if value_index == 1 else entropy_values

            ylabel = "Residual Entropy (bits)" if value_index == 0 else "Residual Transition (10³)"
            residual_type = "Residual-Entropy" if value_index == 0 else "Residual-Transition"

            plt.plot(thresholds, y_values, marker='o', linestyle='-', color=strategy_colours[strategy], label=strategy)

        # Labels and title
        plt.xlabel("Encryption Ratio (Threshold)")
        plt.ylabel(ylabel)

        # Constructing the title and filename explicitly
        title = f"{dataset}_{target_type.replace(' ', '-')}_{residual_type}"
        plt.title(title)

        # Legend and grid
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save graph with the desired filename format
        filename_out = f"{dataset}_{target_type.replace(' ', '-')}_{residual_type}.png"
        output_path = os.path.join(output_folder, filename_out)
        plt.savefig(output_path)
        plt.close()

# Generate all 8 graphs with explicit target and residual types
plot_graphs(data_store, "Entropy based", 0)   # Residual Entropy from Entropy-based data
plot_graphs(data_store, "Transition based", 1)  # Residual Transition from Transition-based data
plot_graphs(data_store, "Entropy based", 1)  # Residual Transition from Entropy-based data
plot_graphs(data_store, "Transition based", 0)  # Residual Entropy from Transition-based data

print("Graphs successfully generated and saved in 'test_metrics_graphs/'.")
