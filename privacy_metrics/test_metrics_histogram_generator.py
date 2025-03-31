import os
import re
import matplotlib.pyplot as plt

# Directories
input_folder = './metrics_tests_text/'
output_folder = './metrics_tests_histograms/'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to extract data from a file
def extract_entropy_histogram(file_path):
    x_values = []
    y_values = []
    strategy, dataset, target = None, None, None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract metadata (Strategy, Dataset, Target)
    for line in lines:
        if "Strategy:" in line:
            strategy = line.split("Strategy:")[1].strip()
        elif "Dataset:" in line:
            dataset = line.split("Dataset:")[1].strip()
        elif "Target:" in line:
            target = line.split("Target:")[1].strip()
        elif "Entropy Histograme:" in line:
            break  # Stop scanning for metadata once we reach the histogram

    # Extract entropy histogram values
    histogram_started = False
    for line in lines:
        if "Entropy Histograme:" in line:
            histogram_started = True
            continue  # Skip the title line

        if histogram_started:
            if re.match(r'^\d+\.\d+,\d+$', line.strip()):  # Match "0.025,129"
                parts = line.strip().split(',')
                x_values.append(float(parts[0]))  # Convert first value to float
                y_values.append(int(parts[1]))    # Convert second value to int
            else:
                break  # Stop reading when the histogram ends

    return x_values, y_values, strategy, dataset, target

# Function to generate and save histograms
def generate_histograms():
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)

            # Extract data
            x_values, y_values, strategy, dataset, target = extract_entropy_histogram(file_path)

            if not (strategy and dataset and target and x_values and y_values):
                print(f"Skipping {filename} due to missing data.")
                continue

            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.bar(x_values, y_values, width=0.04, color='#A7C7E7', edgecolor='black', alpha=0.8)

            # Labels and title
            if target == 'Entropy based':
                plt.xlabel('Entropy Bins')
            else:
                plt.xlabel('Transition Bins')
            plt.ylabel('Frequency')
            title = f"{strategy}_{dataset}_{target}"
            plt.title(title)

            # Save histogram
            filename_out = f"{title}.png".replace(" ", "_")  # Replace spaces with underscores
            output_path = os.path.join(output_folder, filename_out)
            plt.savefig(output_path)
            plt.close()

# Run the function to process files
generate_histograms()
