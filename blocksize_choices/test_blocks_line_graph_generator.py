import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Directory where the .txt files are located
folder_path = './block_tests_text/'

# Subfolder to save the histograms
output_folder = './block_tests_histograms/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the section sizes and their corresponding powers of 2
section_sizes = [2**i for i in range(10, 16)]

# Function to extract data from a single file
def extract_data(file_path):
    summation_times = []
    deviation_times = []
    section_sizes_found = []

    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract dataset and strategy from the file
    dataset_match = re.search(r'Dataset:\s*(\w+)', content)
    strategy_match = re.search(r'Strategy:\s*(.*?)\s*Dataset:', content)

    dataset = dataset_match.group(1) if dataset_match else None
    strategy = strategy_match.group(1) if strategy_match else None

    # Extract time values for each section size
    section_matches = re.findall(r'SectionSize = (\d+).*?Summation time for.*?(\d+\.\d+)(ms|s).*?Deviation time for.*?(\d+\.\d+)(ms|s)', content, re.DOTALL)

    for section, summation_time, summation_unit, deviation_time, deviation_unit in section_matches:
        section_size = int(section)
        summation_time = float(summation_time)
        deviation_time = float(deviation_time)

        # Convert time from ms to s if needed
        if summation_unit == 'ms':
            summation_time /= 1000
        if deviation_unit == 'ms':
            deviation_time /= 1000

        # Append to lists
        summation_times.append(summation_time)
        deviation_times.append(deviation_time)
        section_sizes_found.append(section_size)

    return section_sizes_found, summation_times, deviation_times, dataset, strategy

# Function to plot the line charts
def plot_line_charts():
    # Loop through all files in the block_tests folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # Extract data
            section_sizes, summation_times, deviation_times, dataset, strategy = extract_data(file_path)

            # Create the line chart plot
            plt.figure(figsize=(10, 6))

            # Plot Summation time
            plt.plot(section_sizes, summation_times, label='Summation', marker='o', linestyle='-', color='blue')

            # Plot Deviation time (Variance)
            plt.plot(section_sizes, deviation_times, label='Variance (Deviation)', marker='s', linestyle='-', color='red')

            # Labels and title
            plt.xlabel('Section Size (Block size)')
            plt.ylabel('Time (seconds)')
            plt.title(f'Time Comparison for {strategy} - {dataset}')

            # Set the X-axis to show the powers of 2 explicitly
            plt.xticks(section_sizes, [f'2^{int(np.log2(s))}' for s in section_sizes])

            # Add grid and legend
            plt.grid(True)
            plt.legend()

            # Set the X-axis limits to ensure it spans 2^10 to 2^15 (from 1024 to 32768)
            plt.xlim(min(section_sizes), max(section_sizes))

            # Save the plot as an image in the block_tests_histograms folder
            plot_filename = f"{dataset}_{strategy}_{filename.replace('.txt', '.png')}"
            plot_filepath = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()

# Call the function to generate line charts
plot_line_charts()
