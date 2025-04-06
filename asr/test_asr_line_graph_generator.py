import os
import re
import matplotlib.pyplot as plt

# Directories
input_folder = './asr_tests_electricity_entropy_texts/asr_tests_electricity_entropy_0_text'
output_path = './asr_electricity_line_graphs/electricity_entropy-based/asr_electricity_entropy_0_line_graph.png'

# Strategy colour map
strategy_colours = {
    "Global": "red",
    "Household": "blue",
    "Random": "green"
}

# Data store: { strategy: { atdSize: mean_ASR } }
data = {
    "Global": {},
    "Household": {},
    "Random": {}
}

# Function to extract data from filename and file content
def extract_asr_data(file_path):
    # Get strategy from filename
    filename = os.path.basename(file_path)
    parts = filename.split('_')

    strategy_code = parts[2]
    strategy = {"1": "Global", "2": "Household", "3": "Random"}.get(strategy_code, None)

    if not strategy:
        return None, None, None

    # Extract atdSize from filename
    atd_size_match = re.search(r'_([3-9]|[1-4][0-9])_80\.txt$', filename)
    if atd_size_match:
        atd_size = int(atd_size_match.group(1))
    else:
        atd_size = int(parts[-2])  # Fallback if regex fails

    mean_asr = None

    # Extract mean ASR from file content
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'mean ASR:\s*([0-9.]+)', line)
            if match:
                mean_asr = float(match.group(1))
                break

    return strategy, atd_size, mean_asr

# Loop through files and gather data
for filename in os.listdir(input_folder):
    if filename.endswith('.txt') and filename.startswith('ASR_time_'):
        file_path = os.path.join(input_folder, filename)
        strategy, atd_size, mean_asr = extract_asr_data(file_path)

        if strategy and atd_size is not None and mean_asr is not None:
            data[strategy][atd_size] = mean_asr

# Plotting
plt.figure(figsize=(10, 6))

for strategy, values in data.items():
    # Sort by atdSize
    sorted_items = sorted(values.items())
    x_vals = [item[0] for item in sorted_items]
    y_vals = [item[1] for item in sorted_items]

    plt.plot(x_vals, y_vals, marker='o', linestyle='-', label=strategy, color=strategy_colours[strategy])

plt.xlabel('Leaked Block Sizes')
plt.ylabel('ASR')
plt.title('Encryption Ratio 0')
plt.xticks([3, 18, 33, 48])
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Line graph saved to: {output_path}")
