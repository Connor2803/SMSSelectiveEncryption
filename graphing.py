import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Directory containing the results files
results_folder = './results/'
output_folder = './results/graphs/'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def debug_content(filename, content):
    """Debug function to show what data is available in the file"""
    print(f"  DEBUG - Content preview for {filename}:")
    lines = content.split('\n')[:10]  # Show first 10 lines
    for i, line in enumerate(lines):
        if line.strip():
            print(f"    Line {i+1}: {line[:100]}...")
    
    # Fix the f-string issue by calculating the length outside the f-string
    total_lines = len(content.split('\n'))
    print(f"  Total lines: {total_lines}")

def extract_asr_data(content):
    """Extract ASR values from content with multiple patterns"""
    patterns = [
        r'ASR:\s*([0-9.-]+)',
        r'ASR\s*=\s*([0-9.-]+)',
        r'Success Rate:\s*([0-9.-]+)',
        r'Attack Success:\s*([0-9.-]+)',
    ]
    asrs = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        asrs.extend([float(match) for match in matches])
    return asrs

def extract_time_data(content):
    """Extract processing time values with multiple patterns"""
    patterns = [
        r'Time:\s*([0-9.]+)s?',
        r'Processing Time:\s*([0-9.]+)',
        r'Duration:\s*([0-9.]+)',
        r'Elapsed:\s*([0-9.]+)',
        r'([0-9.]+)\s*seconds',
    ]
    times = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        times.extend([float(match) for match in matches])
    return times

def extract_ratio_data(content):
    """Extract encryption ratio values with multiple patterns"""
    patterns = [
        r'Ratio:\s*(\d+)%',
        r'ratio:\s*(\d+)%?',
        r'(\d+)%\s*ratio',
        r'Encryption.*?(\d+)%',
        r'Testing.*?(\d+)%',
        r'(\d+)%',  # Generic percentage
    ]
    ratios = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        ratios.extend([int(match) for match in matches if 0 <= int(match) <= 100])
    return sorted(list(set(ratios)))  # Remove duplicates and sort

def extract_threshold_data(content):
    """Extract threshold values with multiple patterns"""
    patterns = [
        r'threshold:\s*(\d+)%?',
        r'Threshold:\s*(\d+)%?',
        r'matching.*?(\d+)%',
        r'(\d+)%.*?threshold',
    ]
    thresholds = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        thresholds.extend([int(match) for match in matches])
    return thresholds

def extract_section_size_data(content):
    """Extract section size values with multiple patterns"""
    patterns = [
        r'Section Size:\s*(\d+)',
        r'size:\s*(\d+)',
        r'Size:\s*(\d+)',
        r'Testing.*?size.*?(\d+)',
        r'(\d+).*?section',
        r'section.*?(\d+)',
    ]
    sizes = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        # Filter reasonable section sizes (typically powers of 2, 64-8192)
        sizes.extend([int(match) for match in matches if 64 <= int(match) <= 16384])
    return sorted(list(set(sizes)))

def extract_window_size_data(content):
    """Extract window size values with multiple patterns"""
    patterns = [
        r'Window Size:\s*(\d+)',
        r'window:\s*(\d+)',
        r'(\d+).*?window',
        r'temporal.*?(\d+)',
    ]
    windows = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        windows.extend([int(match) for match in matches if int(match) > 10])
    return windows

def extract_atd_size_data(content):
    """Extract ATD size values with multiple patterns"""
    patterns = [
        r'ATD Size:\s*(\d+)',
        r'ATD:\s*(\d+)',
        r'(\d+).*?hours?',
        r'(\d+).*?ATD',
    ]
    atd_sizes = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        atd_sizes.extend([int(match) for match in matches if 1 <= int(match) <= 168])  # 1 hour to 1 week
    return atd_sizes

def extract_memory_data(content):
    """Extract memory usage values with multiple patterns"""
    patterns = [
        r'Memory:\s*([0-9.]+)\s*(MB|KB|GB|bytes)',
        r'([0-9.]+)\s*(MB|KB|GB)\s*memory',
        r'memory.*?([0-9.]+)\s*(MB|KB|GB)',
    ]
    memory_values = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            value, unit = match
            value = float(value)
            # Convert to MB for consistency
            if unit.upper() == 'KB':
                value /= 1024
            elif unit.upper() == 'GB':
                value *= 1024
            elif unit.upper() == 'BYTES':
                value /= (1024 * 1024)
            memory_values.append(value)
    return memory_values

def extract_numerical_sequences(content):
    """Extract sequences of numbers that might represent test results"""
    lines = content.split('\n')
    sequences = []
    
    for line in lines:
        # Look for lines with multiple numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', line)
        if len(numbers) >= 2:
            try:
                float_numbers = [float(n) for n in numbers]
                sequences.append(float_numbers)
            except ValueError:
                continue
    
    return sequences

def plot_adaptive_results(filename, content):
    """Plot adaptive threshold results with enhanced pattern matching"""
    asrs = extract_asr_data(content)
    ratios = extract_ratio_data(content)
    
    print(f"  DEBUG - Adaptive: Found {len(asrs)} ASRs, {len(ratios)} ratios")
    
    # Try to extract from specific line patterns
    if not (asrs and ratios):
        lines = content.split('\n')
        adaptive_asrs = []
        adaptive_ratios = []
        
        for line in lines:
            if 'adaptive' in line.lower() and any(char.isdigit() for char in line):
                # Extract numbers from adaptive lines
                numbers = re.findall(r'\d+\.?\d*', line)
                if len(numbers) >= 2:
                    try:
                        # Assume format: ratio%, ASR
                        ratio = int(float(numbers[0]))
                        asr = float(numbers[1])
                        if 0 <= ratio <= 100 and 0 <= asr <= 1:
                            adaptive_ratios.append(ratio)
                            adaptive_asrs.append(asr)
                    except (ValueError, IndexError):
                        continue
        
        if adaptive_asrs and adaptive_ratios:
            asrs = adaptive_asrs
            ratios = adaptive_ratios
    
    if asrs and ratios and len(asrs) >= 2:
        plt.figure(figsize=(10, 6))
        if len(asrs) == len(ratios):
            plt.plot(ratios, asrs, 'bo-', linewidth=2, markersize=8)
        else:
            # If lengths don't match, use indices
            plt.plot(range(len(asrs)), asrs, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Adaptive Ratio (%)' if len(asrs) == len(ratios) else 'Test Index')
        plt.ylabel('ASR')
        plt.title('Adaptive Threshold Analysis')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return True
    return False

def plot_hybrid_results(filename, content):
    """Plot hybrid strategy results with enhanced pattern matching"""
    # Look for entropy and uniqueness weights - avoid extracting "..."
    entropy_pattern = r'(?:entropy|ent).*?([0-9]+\.?[0-9]*)'
    uniqueness_pattern = r'(?:uniqueness|uniq).*?([0-9]+\.?[0-9]*)'
    
    entropy_weights = []
    uniqueness_weights = []
    
    # Extract only valid numbers, not "..."
    for match in re.findall(entropy_pattern, content, re.IGNORECASE):
        try:
            val = float(match)
            if 0 <= val <= 1:  # Valid weight range
                entropy_weights.append(val)
        except ValueError:
            continue
    
    for match in re.findall(uniqueness_pattern, content, re.IGNORECASE):
        try:
            val = float(match)
            if 0 <= val <= 1:  # Valid weight range
                uniqueness_weights.append(val)
        except ValueError:
            continue
    
    asrs = extract_asr_data(content)
    
    print(f"  DEBUG - Hybrid: Found {len(entropy_weights)} entropy, {len(uniqueness_weights)} uniqueness, {len(asrs)} ASRs")
    
    # Try alternative patterns if nothing found
    if not entropy_weights and not uniqueness_weights:
        lines = content.split('\n')
        for line in lines:
            if 'hybrid' in line.lower() and ':' in line and '...' not in line:
                # Look for patterns like "Entropy: 0.8, Uniqueness: 0.2"
                entropy_match = re.search(r'entropy.*?([0-9.]+)', line, re.IGNORECASE)
                uniqueness_match = re.search(r'uniqueness.*?([0-9.]+)', line, re.IGNORECASE)
                if entropy_match:
                    try:
                        entropy_weights.append(float(entropy_match.group(1)))
                    except ValueError:
                        continue
                if uniqueness_match:
                    try:
                        uniqueness_weights.append(float(uniqueness_match.group(1)))
                    except ValueError:
                        continue
    
    if (entropy_weights or uniqueness_weights) and len(asrs) >= 2:
        plt.figure(figsize=(12, 8))
        
        if entropy_weights and len(entropy_weights) == len(asrs):
            plt.subplot(1, 2, 1)
            plt.plot(entropy_weights, asrs, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Entropy Weight')
            plt.ylabel('ASR')
            plt.title('Entropy Weight vs ASR')
            plt.grid(True, alpha=0.3)
        
        if uniqueness_weights and len(uniqueness_weights) == len(asrs):
            plt.subplot(1, 2, 2 if entropy_weights else 1)
            plt.plot(uniqueness_weights, asrs, 'mo-', linewidth=2, markersize=8)
            plt.xlabel('Uniqueness Weight')
            plt.ylabel('ASR')
            plt.title('Uniqueness Weight vs ASR')
            plt.grid(True, alpha=0.3)
        
        # If we only have ASRs, plot them by index
        if not entropy_weights and not uniqueness_weights and len(asrs) >= 2:
            plt.plot(range(len(asrs)), asrs, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Test Configuration')
            plt.ylabel('ASR')
            plt.title('Hybrid Strategy Results')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

# Add these missing functions to your graphing.py file after line 573:

def plot_atd_results(filename, content):
    """Plot ATD size analysis results"""
    asrs = extract_asr_data(content)
    atd_sizes = extract_atd_size_data(content)
    
    if asrs and atd_sizes:
        plt.figure(figsize=(12, 8))
        
        # Group by dataset if possible
        electricity_asrs = []
        electricity_sizes = []
        water_asrs = []
        water_sizes = []
        
        lines = content.split('\n')
        current_dataset = None
        
        for i, line in enumerate(lines):
            if 'electricity' in line.lower() and 'dataset' in line.lower():
                current_dataset = 'electricity'
            elif 'water' in line.lower() and 'dataset' in line.lower():
                current_dataset = 'water'
            elif 'ATD Size:' in line and 'ASR:' in line:
                atd_match = re.search(r'ATD Size:\s*(\d+)', line)
                asr_match = re.search(r'ASR:\s*([0-9.-]+)', line)
                if atd_match and asr_match:
                    atd_size = int(atd_match.group(1))
                    asr = float(asr_match.group(1))
                    if current_dataset == 'electricity':
                        electricity_sizes.append(atd_size)
                        electricity_asrs.append(asr)
                    elif current_dataset == 'water':
                        water_sizes.append(atd_size)
                        water_asrs.append(asr)
        
        if electricity_sizes and electricity_asrs:
            plt.plot(electricity_sizes, electricity_asrs, 'ro-', label='Electricity', linewidth=2, markersize=8)
        if water_sizes and water_asrs:
            plt.plot(water_sizes, water_asrs, 'bo-', label='Water', linewidth=2, markersize=8)
        
        plt.xlabel('ATD Size (hours)')
        plt.ylabel('ASR')
        plt.title('ATD Size Impact Analysis')
        if electricity_sizes or water_sizes:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return True
    return False

def plot_memory_results(filename, content):
    """Plot memory efficiency results"""
    household_pattern = r'with\s*(\d+)\s*households'
    households = [int(match) for match in re.findall(household_pattern, content)]
    memory_values = extract_memory_data(content)
    asrs = extract_asr_data(content)
    times = extract_time_data(content)
    
    print(f"  DEBUG - Memory: Found {len(households)} households, {len(memory_values)} memory, {len(asrs)} ASRs, {len(times)} times")
    
    if households and len(households) >= 2:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        if memory_values and len(memory_values) == len(households):
            plt.plot(households, memory_values, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Number of Households')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory vs Households')
            plt.grid(True, alpha=0.3)
        elif asrs and len(asrs) >= 2:
            # If no memory data, show ASR vs households
            plt.plot(households[:len(asrs)], asrs, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Number of Households')
            plt.ylabel('ASR')
            plt.title('ASR vs Households')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        if asrs and len(asrs) >= len(households):
            plt.plot(households, asrs[:len(households)], 'go-', linewidth=2, markersize=8)
            plt.xlabel('Number of Households')
            plt.ylabel('ASR')
            plt.title('ASR vs Households')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        if times and len(times) >= len(households):
            plt.plot(households, times[:len(households)], 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Households')
            plt.ylabel('Processing Time (s)')
            plt.title('Time vs Households')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

def plot_performance_results(filename, content):
    """Plot performance benchmark results"""
    ratios = extract_ratio_data(content)
    asrs = extract_asr_data(content)
    times = extract_time_data(content)
    
    print(f"  DEBUG - Performance: Found {len(ratios)} ratios, {len(asrs)} ASRs, {len(times)} times")
    
    # Extract dataset information from lines
    lines = content.split('\n')
    electricity_data = {'ratios': [], 'asrs': [], 'times': []}
    water_data = {'ratios': [], 'asrs': [], 'times': []}
    
    for line in lines:
        if 'electricity' in line.lower():
            ratio_match = re.search(r'(\d+)%', line)
            asr_match = re.search(r'ASR:\s*([0-9.-]+)', line)
            time_match = re.search(r'Time:\s*([0-9.]+)', line)
            
            if ratio_match and asr_match and time_match:
                electricity_data['ratios'].append(int(ratio_match.group(1)))
                electricity_data['asrs'].append(float(asr_match.group(1)))
                electricity_data['times'].append(float(time_match.group(1)))
        elif 'water' in line.lower():
            ratio_match = re.search(r'(\d+)%', line)
            asr_match = re.search(r'ASR:\s*([0-9.-]+)', line)
            time_match = re.search(r'Time:\s*([0-9.]+)', line)
            
            if ratio_match and asr_match and time_match:
                water_data['ratios'].append(int(ratio_match.group(1)))
                water_data['asrs'].append(float(asr_match.group(1)))
                water_data['times'].append(float(time_match.group(1)))
    
    # If we have data, create plots
    if (electricity_data['ratios'] or water_data['ratios']) or (asrs and ratios):
        plt.figure(figsize=(15, 5))
        
        # ASR vs Ratio
        plt.subplot(1, 3, 1)
        if electricity_data['ratios']:
            plt.plot(electricity_data['ratios'], electricity_data['asrs'], 'ro-', label='Electricity', linewidth=2, markersize=8)
        if water_data['ratios']:
            plt.plot(water_data['ratios'], water_data['asrs'], 'bo-', label='Water', linewidth=2, markersize=8)
        if not electricity_data['ratios'] and not water_data['ratios'] and asrs and ratios:
            plt.plot(ratios[:len(asrs)], asrs, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Encryption Ratio (%)')
        plt.ylabel('ASR')
        plt.title('ASR vs Encryption Ratio')
        if electricity_data['ratios'] or water_data['ratios']:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Processing Time vs Ratio
        plt.subplot(1, 3, 2)
        if electricity_data['times']:
            plt.plot(electricity_data['ratios'], electricity_data['times'], 'ro-', label='Electricity', linewidth=2, markersize=8)
        if water_data['times']:
            plt.plot(water_data['ratios'], water_data['times'], 'bo-', label='Water', linewidth=2, markersize=8)
        if not electricity_data['times'] and not water_data['times'] and times and ratios:
            plt.plot(ratios[:len(times)], times, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Encryption Ratio (%)')
        plt.ylabel('Processing Time (s)')
        plt.title('Processing Time vs Encryption Ratio')
        if electricity_data['times'] or water_data['times']:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ASR vs Time (Performance Trade-off)
        plt.subplot(1, 3, 3)
        if electricity_data['asrs']:
            plt.scatter(electricity_data['times'], electricity_data['asrs'], c='red', s=100, alpha=0.7, label='Electricity')
        if water_data['asrs']:
            plt.scatter(water_data['times'], water_data['asrs'], c='blue', s=100, alpha=0.7, label='Water')
        if not electricity_data['asrs'] and not water_data['asrs'] and asrs and times:
            plt.scatter(times[:len(asrs)], asrs, c='green', s=100, alpha=0.7)
        plt.xlabel('Processing Time (s)')
        plt.ylabel('ASR')
        plt.title('Security vs Performance Trade-off')
        if electricity_data['asrs'] or water_data['asrs']:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

def plot_temporal_results(filename, content):
    """Plot temporal variations results"""
    windows = extract_window_size_data(content)
    asrs = extract_asr_data(content)
    times = extract_time_data(content)
    
    print(f"  DEBUG - Temporal: Found {len(windows)} windows, {len(asrs)} ASRs, {len(times)} times")
    
    # Try to extract window sizes from lines with "data points"
    if not windows:
        window_pattern = r'(\d+)\s*data\s*points'
        windows = [int(match) for match in re.findall(window_pattern, content, re.IGNORECASE)]
    
    if (windows or asrs) and len(asrs) >= 2:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        if windows and len(windows) == len(asrs):
            plt.plot(windows, asrs, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Window Size (data points)')
            plt.xscale('log', base=2)
        else:
            plt.plot(range(len(asrs)), asrs, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
        plt.ylabel('ASR')
        plt.title('ASR vs Temporal Window Size')
        plt.grid(True, alpha=0.3)
        
        if times and len(times) >= 2:
            plt.subplot(1, 3, 2)
            if windows and len(windows) == len(times):
                plt.plot(windows, times, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Window Size (data points)')
                plt.xscale('log', base=2)
            else:
                plt.plot(range(len(times)), times, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Test Index')
            plt.ylabel('Processing Time (s)')
            plt.title('Processing Time vs Window Size')
            plt.grid(True, alpha=0.3)
        
        # Efficiency analysis
        plt.subplot(1, 3, 3)
        if windows and times and len(windows) == len(times):
            time_per_point = [time * 1000 / window for time, window in zip(times, windows)]
            plt.plot(windows, time_per_point, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Window Size (data points)')
            plt.ylabel('Time per Data Point (ms)')
            plt.title('Efficiency vs Window Size')
            plt.xscale('log', base=2)
        elif len(asrs) >= 2 and len(times) >= 2:
            min_len = min(len(asrs), len(times))
            efficiency = [asr / time if time > 0 else 0 for asr, time in zip(asrs[:min_len], times[:min_len])]
            plt.plot(range(len(efficiency)), efficiency, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
            plt.ylabel('Efficiency (ASR/Time)')
            plt.title('Efficiency Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

def plot_matching_threshold_results(filename, content):
    """Plot matching threshold results with enhanced pattern matching"""
    thresholds = extract_threshold_data(content)
    asrs = extract_asr_data(content)
    
    print(f"  DEBUG - Matching: Found {len(thresholds)} thresholds, {len(asrs)} ASRs")
    
    # Try to extract from percentage patterns
    if not thresholds:
        percentages = re.findall(r'(\d+)%', content)
        thresholds = [int(p) for p in percentages if 10 <= int(p) <= 100]
    
    if asrs and len(asrs) >= 2:
        plt.figure(figsize=(10, 6))
        if thresholds and len(thresholds) == len(asrs):
            plt.plot(thresholds, asrs, 'co-', linewidth=2, markersize=8)
            plt.xlabel('Matching Threshold (%)')
        else:
            plt.plot(range(len(asrs)), asrs, 'co-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
        plt.ylabel('ASR')
        plt.title('Matching Threshold Analysis')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return True
    return False

def plot_section_results(filename, content):
    """Plot section size results with enhanced pattern matching"""
    sizes = extract_section_size_data(content)
    asrs = extract_asr_data(content)
    times = extract_time_data(content)
    
    print(f"  DEBUG - Section: Found {len(sizes)} sizes, {len(asrs)} ASRs, {len(times)} times")
    
    # Try to extract from lines containing numbers
    if not sizes:
        # Look for common section sizes
        common_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        for size in common_sizes:
            if str(size) in content:
                sizes.append(size)
    
    if asrs and len(asrs) >= 2:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        if sizes and len(sizes) == len(asrs):
            plt.plot(sizes, asrs, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Section Size')
            plt.xscale('log', base=2)
        else:
            plt.plot(range(len(asrs)), asrs, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
        plt.ylabel('ASR')
        plt.title('ASR vs Section Size')
        plt.grid(True, alpha=0.3)
        
        if times and len(times) >= 2:
            plt.subplot(2, 2, 2)
            if sizes and len(sizes) == len(times):
                plt.plot(sizes, times, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Section Size')
                plt.xscale('log', base=2)
            else:
                plt.plot(range(len(times)), times, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Test Index')
            plt.ylabel('Processing Time (s)')
            plt.title('Processing Time vs Section Size')
            plt.grid(True, alpha=0.3)
        
        if len(asrs) >= 3:
            plt.subplot(2, 2, 3)
            if sizes:
                best_idx = asrs.index(min([asr for asr in asrs if asr >= 0]))
                if best_idx < len(sizes):
                    plt.bar(['Recommended'], [sizes[best_idx]], color='green', alpha=0.7)
                    plt.ylabel('Section Size')
                    plt.title(f'Best Size: {sizes[best_idx]}')
                else:
                    plt.bar(['Best Index'], [best_idx], color='green', alpha=0.7)
                    plt.ylabel('Test Index')
                    plt.title(f'Best Test: {best_idx}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

def plot_scalability_results(filename, content):
    """Plot scalability analysis results with enhanced pattern matching"""
    ratios = extract_ratio_data(content)
    asrs = extract_asr_data(content)
    times = extract_time_data(content)
    
    print(f"  DEBUG - Scalability: Found {len(ratios)} ratios, {len(asrs)} ASRs, {len(times)} times")
    
    # If we have any data, create meaningful plots
    if asrs and len(asrs) >= 2:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        if ratios and len(ratios) == len(asrs):
            plt.plot(ratios, asrs, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Encryption Ratio (%)')
        else:
            plt.plot(range(len(asrs)), asrs, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
        plt.ylabel('ASR')
        plt.title('Scalability: ASR Analysis')
        plt.grid(True, alpha=0.3)
        
        if times and len(times) >= 2:
            plt.subplot(1, 3, 2)
            if ratios and len(ratios) == len(times):
                plt.plot(ratios, times, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Encryption Ratio (%)')
            else:
                plt.plot(range(len(times)), times, 'ro-', linewidth=2, markersize=8)
                plt.xlabel('Test Index')
            plt.ylabel('Processing Time (s)')
            plt.title('Scalability: Time Analysis')
            plt.grid(True, alpha=0.3)
        
        # Growth analysis if we have enough data
        if len(asrs) >= 3:
            plt.subplot(1, 3, 3)
            if len(times) == len(asrs):
                efficiency = [asr / time if time > 0 else 0 for asr, time in zip(asrs, times)]
                plt.plot(range(len(efficiency)), efficiency, 'bo-', linewidth=2, markersize=8)
                plt.ylabel('Efficiency (ASR/Time)')
                plt.title('Efficiency Analysis')
            else:
                growth_rates = [(asrs[i] - asrs[i-1]) for i in range(1, len(asrs))]
                plt.plot(range(len(growth_rates)), growth_rates, 'bo-', linewidth=2, markersize=8)
                plt.ylabel('ASR Change')
                plt.title('ASR Growth Analysis')
            plt.xlabel('Test Step')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

def plot_security_results(filename, content):
    """Plot security analysis results with enhanced pattern matching"""
    lines = content.split('\n')
    scenarios = []
    scenario_asrs = []
    
    # Look for different security scenario patterns
    security_patterns = [
        r'(.*attack.*|.*attacker.*|.*scenario.*):.*?([0-9.-]+)',
        r'(.*security.*|.*threat.*):.*?([0-9.-]+)',
        r'([^:]+):.*?ASR.*?([0-9.-]+)',
    ]
    
    for line in lines:
        for pattern in security_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                scenario = match.group(1).strip()
                try:
                    asr = float(match.group(2))
                    if 0 <= asr <= 1:
                        scenarios.append(scenario)
                        scenario_asrs.append(asr)
                        break
                except ValueError:
                    continue
    
    print(f"  DEBUG - Security: Found {len(scenarios)} scenarios with ASRs")
    
    # If no specific scenarios found, use generic ASR data
    if not scenarios:
        asrs = extract_asr_data(content)
        if len(asrs) >= 2:
            scenarios = [f'Test {i+1}' for i in range(len(asrs))]
            scenario_asrs = asrs
    
    if scenarios and scenario_asrs and len(scenarios) >= 2:
        plt.figure(figsize=(12, 8))
        
        # Bar chart of ASR by scenario
        plt.subplot(2, 1, 1)
        colors = ['red' if asr > 0.5 else 'orange' if asr > 0.3 else 'green' for asr in scenario_asrs]
        bars = plt.bar(range(len(scenarios)), scenario_asrs, color=colors, alpha=0.7)
        plt.xticks(range(len(scenarios)), scenarios, rotation=45, ha='right')
        plt.ylabel('ASR')
        plt.title('Security Analysis: ASR by Scenario')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, asr in zip(bars, scenario_asrs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{asr:.3f}', ha='center', va='bottom')
        
        # Security level summary
        plt.subplot(2, 1, 2)
        worst_asr = max(scenario_asrs) if scenario_asrs else 0
        avg_asr = sum(scenario_asrs) / len(scenario_asrs) if scenario_asrs else 0
        
        plt.bar(['Worst Case', 'Average'], [worst_asr, avg_asr], 
                color=['red' if worst_asr > 0.4 else 'orange' if worst_asr > 0.2 else 'green', 'blue'], 
                alpha=0.7)
        plt.ylabel('ASR')
        plt.title('Security Summary')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False

def plot_generic_results(filename, content):
    """Enhanced generic plotting for unrecognized file types"""
    asrs = extract_asr_data(content)
    times = extract_time_data(content)
    ratios = extract_ratio_data(content)
    
    print(f"  DEBUG - Generic: Found {len(asrs)} ASRs, {len(times)} times, {len(ratios)} ratios")
    
    # Try to extract numerical sequences if standard patterns fail
    if not asrs and not times:
        sequences = extract_numerical_sequences(content)
        if sequences:
            # Use the longest sequence as potential data
            longest_seq = max(sequences, key=len)
            if len(longest_seq) >= 2:
                asrs = longest_seq
    
    if asrs or times or ratios:
        plt.figure(figsize=(12, 6))
        
        subplot_count = sum([bool(asrs), bool(times), bool(ratios)])
        subplot_idx = 1
        
        if asrs and len(asrs) >= 2:
            plt.subplot(1, max(2, subplot_count), subplot_idx)
            plt.plot(range(len(asrs)), asrs, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
            plt.ylabel('ASR')
            plt.title('ASR Values')
            plt.grid(True, alpha=0.3)
            subplot_idx += 1
        
        if times and len(times) >= 2:
            plt.subplot(1, max(2, subplot_count), subplot_idx)
            plt.plot(range(len(times)), times, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
            plt.ylabel('Processing Time (s)')
            plt.title('Processing Times')
            plt.grid(True, alpha=0.3)
            subplot_idx += 1
        
        if ratios and len(ratios) >= 2:
            plt.subplot(1, max(2, subplot_count), subplot_idx)
            plt.plot(range(len(ratios)), ratios, 'go-', linewidth=2, markersize=8)
            plt.xlabel('Test Index')
            plt.ylabel('Ratio (%)')
            plt.title('Encryption Ratios')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return True
    return False



def generate_graphs_for_results():
    """Main function to generate graphs for all result files"""
    # Get all .txt files in results folder
    txt_files = [f for f in os.listdir(results_folder) if f.endswith('.txt')]
    
    print(f"Found {len(txt_files)} result files to process...")
    
    for filename in txt_files:
        print(f"Processing {filename}...")
        
        file_path = os.path.join(results_folder, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Debug: Show content preview for failed files
            debug_content(filename, content)
            
            # Determine the appropriate plotting function based on filename
            plot_generated = False
            
            if 'adaptive' in filename.lower():
                plot_generated = plot_adaptive_results(filename, content)
            elif 'atd' in filename.lower():
                plot_generated = plot_atd_results(filename, content)
            elif 'hybrid' in filename.lower():
                plot_generated = plot_hybrid_results(filename, content)
            elif 'matching' in filename.lower():
                plot_generated = plot_matching_threshold_results(filename, content)
            elif 'memory' in filename.lower():
                plot_generated = plot_memory_results(filename, content)
            elif 'performance' in filename.lower():
                plot_generated = plot_performance_results(filename, content)
            elif 'scalability' in filename.lower():
                plot_generated = plot_scalability_results(filename, content)
            elif 'section' in filename.lower():
                plot_generated = plot_section_results(filename, content)
            elif 'security' in filename.lower():
                plot_generated = plot_security_results(filename, content)
            elif 'temporal' in filename.lower():
                plot_generated = plot_temporal_results(filename, content)
            else:
                plot_generated = plot_generic_results(filename, content)
            
            if plot_generated:
                # Save the plot
                output_filename = f"{os.path.splitext(filename)[0]}_analysis.png"
                output_path = os.path.join(output_folder, output_filename)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Generated graph: {output_filename}")
            else:
                print(f"  ✗ Could not generate graph for {filename} (insufficient data)")
                
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    print("=== SMS Selective Encryption Results Visualization ===")
    print(f"Input folder: {results_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 50)
    
    generate_graphs_for_results()
    
    print("=" * 50)
    print("Graph generation completed!")
    print(f"Check the '{output_folder}' folder for generated graphs.")