import os
import re

import matplotlib.pyplot as plt

# Folder containing the files
folder_path = r"C:/Users/Conno/Documents/GitHub/SMSSelectiveEncryption/ASR Electric Threshold"

# Extract mean ASR from a file
def extract_mean_asr(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(r"mean ASR: ([0-9.]+)", content)
        if match:
            return float(match.group(1))
    return None

# Get all files in the folder
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Extract mean ASR values for each file
mean_asrs = []
valid_file_names = []

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    mean_asr = extract_mean_asr(file_path)
    if mean_asr is not None:
        mean_asrs.append(mean_asr)
        valid_file_names.append(file_name)

# Remove .txt postfix and sort by file name
valid_file_names_no_ext = [os.path.splitext(name)[0] for name in valid_file_names]
sorted_data = sorted(zip(valid_file_names_no_ext, mean_asrs), key=lambda x: x[0])

# Unzip sorted data
sorted_file_names, sorted_mean_asrs = zip(*sorted_data)

# Plot the data
plt.figure(figsize=(10, 6))
plt.bar(sorted_file_names, sorted_mean_asrs, color='skyblue')
plt.xlabel('Encryption Threshold', fontsize=12)
plt.ylabel('Mean ASR', fontsize=12)
plt.title('Electricity Mean ASR vs Encryption Threshold', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('mean_asr_plot.png')
plt.show()