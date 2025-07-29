import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the data
df = pd.read_csv('comprehensive_optimization_results.csv')

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 24))

# 1. Section Size vs ASR
plt.subplot(4, 3, 1)
section_data = df[df['TestName'] == 'SectionSize']
plt.plot(section_data['SectionSize'], section_data['ASR'], 'o-', linewidth=2, markersize=8)
plt.xlabel('Section Size')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Section Size Optimization')
plt.grid(True, alpha=0.3)

# 2. ATD Size vs ASR (by Dataset)
plt.subplot(4, 3, 2)
atd_data = df[df['TestName'] == 'ATDSize']
for dataset in atd_data['Dataset'].unique():
    data = atd_data[atd_data['Dataset'] == dataset]
    plt.plot(data['ATDSize'], data['ASR'], 'o-', label=dataset, linewidth=2, markersize=8)
plt.xlabel('ATD Size (hours)')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('ATD Size Impact Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Matching Threshold vs ASR
plt.subplot(4, 3, 3)
threshold_data = df[df['TestName'] == 'MatchingThreshold']
plt.plot(threshold_data['MatchingThreshold'], threshold_data['ASR'], 'o-', 
         color='red', linewidth=2, markersize=8)
plt.xlabel('Matching Threshold (%)')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Matching Threshold Analysis')
plt.grid(True, alpha=0.3)

# 4. Hybrid Strategy Weight Analysis
plt.subplot(4, 3, 4)
hybrid_data = df[df['TestName'] == 'Hybrid']
plt.scatter(hybrid_data['EntropyWeight'], hybrid_data['ASR'], 
           s=100, alpha=0.7, c=hybrid_data['UniquenessWeight'], cmap='viridis')
plt.colorbar(label='Uniqueness Weight')
plt.xlabel('Entropy Weight')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Hybrid Strategy Optimization')
plt.grid(True, alpha=0.3)

# 5. Performance Benchmarking by Strategy
plt.subplot(4, 3, 5)
perf_data = df[df['TestName'] == 'Performance']
strategies = perf_data['Strategy'].unique()
x_pos = np.arange(len(strategies))
asr_means = [perf_data[perf_data['Strategy'] == s]['ASR'].mean() for s in strategies]
asr_stds = [perf_data[perf_data['Strategy'] == s]['ASR'].std() for s in strategies]

plt.bar(x_pos, asr_means, yerr=asr_stds, capsize=5, alpha=0.7)
plt.xlabel('Strategy')
plt.ylabel('Mean ASR')
plt.title('Performance by Strategy')
plt.xticks(x_pos, strategies, rotation=45)
plt.grid(True, alpha=0.3)

# 6. Performance by Encryption Ratio
plt.subplot(4, 3, 6)
for strategy in strategies:
    data = perf_data[perf_data['Strategy'] == strategy]
    ratios = sorted(data['EncryptionRatio'].unique())
    asr_by_ratio = [data[data['EncryptionRatio'] == r]['ASR'].mean() for r in ratios]
    plt.plot(ratios, asr_by_ratio, 'o-', label=strategy, linewidth=2, markersize=6)

plt.xlabel('Encryption Ratio (%)')
plt.ylabel('Mean ASR')
plt.title('ASR vs Encryption Ratio by Strategy')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Security Analysis
plt.subplot(4, 3, 7)
security_data = df[df['TestName'].str.contains('Security_', na=False)]
attacker_types = [name.replace('Security_', '') for name in security_data['TestName']]
plt.barh(range(len(attacker_types)), security_data['ASR'], alpha=0.7)
plt.yticks(range(len(attacker_types)), attacker_types)
plt.xlabel('Attack Success Rate (ASR)')
plt.title('Security vs Attacker Types')
plt.grid(True, alpha=0.3)

# 8. Temporal Variations
plt.subplot(4, 3, 8)
temporal_data = df[df['TestName'].str.contains('Temporal_', na=False)]
window_sizes = [int(name.replace('Temporal_', '')) for name in temporal_data['TestName']]
plt.plot(window_sizes, temporal_data['ASR'], 'o-', color='purple', linewidth=2, markersize=8)
plt.xlabel('Temporal Window Size')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Temporal Variations Analysis')
plt.grid(True, alpha=0.3)

# 9. Memory Efficiency
plt.subplot(4, 3, 9)
memory_data = df[df['TestName'].str.contains('Memory_', na=False)]
household_counts = [int(name.replace('Memory_', '')) for name in memory_data['TestName']]
plt.scatter(household_counts, memory_data['ASR'], s=memory_data['MemoryUsage']/10, 
           alpha=0.7, c='orange')
plt.xlabel('Number of Households')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Memory Efficiency (bubble size = memory usage)')
plt.grid(True, alpha=0.3)

# 10. Scalability Analysis
plt.subplot(4, 3, 10)
scalability_data = df[df['TestName'].str.contains('Scalability_', na=False)]
ratios = [int(name.replace('Scalability_', '')) for name in scalability_data['TestName']]
plt.plot(ratios, scalability_data['ASR'], 'o-', color='green', linewidth=2, markersize=8)
plt.xlabel('Encryption Ratio (%)')
plt.ylabel('Attack Success Rate (ASR)')
plt.title('Scalability Analysis')
plt.grid(True, alpha=0.3)

# 11. Processing Time Analysis
plt.subplot(4, 3, 11)
# Filter out zero processing times for better visualization
time_data = df[df['ProcessingTime'] > 0]
test_types = time_data['TestName'].unique()
mean_times = [time_data[time_data['TestName'] == t]['ProcessingTime'].mean() for t in test_types]

plt.bar(range(len(test_types)), mean_times, alpha=0.7)
plt.xticks(range(len(test_types)), test_types, rotation=45, ha='right')
plt.ylabel('Mean Processing Time (s)')
plt.title('Processing Time by Test Type')
plt.grid(True, alpha=0.3)

# 12. Overall ASR Distribution
plt.subplot(4, 3, 12)
plt.hist(df['ASR'], bins=20, alpha=0.7, edgecolor='black')
plt.axvline(df['ASR'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["ASR"].mean():.3f}')
plt.axvline(df['ASR'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["ASR"].median():.3f}')
plt.xlabel('Attack Success Rate (ASR)')
plt.ylabel('Frequency')
plt.title('Overall ASR Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_optimization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary statistics table
print("=== OPTIMIZATION RESULTS SUMMARY ===")
print(f"Total test configurations: {len(df)}")
print(f"Mean ASR: {df['ASR'].mean():.4f}")
print(f"Median ASR: {df['ASR'].median():.4f}")
print(f"Min ASR: {df['ASR'].min():.4f}")
print(f"Max ASR: {df['ASR'].max():.4f}")
print(f"Standard deviation: {df['ASR'].std():.4f}")

print("\n=== BEST CONFIGURATIONS (Lowest ASR) ===")
best_configs = df.nsmallest(5, 'ASR')[['TestName', 'SectionSize', 'ATDSize', 'MatchingThreshold', 'ASR', 'Dataset', 'Strategy']]
print(best_configs)

print("\n=== WORST CONFIGURATIONS (Highest ASR) ===")
worst_configs = df.nlargest(5, 'ASR')[['TestName', 'SectionSize', 'ATDSize', 'MatchingThreshold', 'ASR', 'Dataset', 'Strategy']]
print(worst_configs)

# Create a heatmap for strategy vs dataset performance
plt.figure(figsize=(10, 6))
pivot_data = df[df['TestName'] == 'Performance'].pivot_table(
    values='ASR', 
    index='Strategy', 
    columns='Dataset', 
    aggfunc='mean'
)
sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', fmt='.3f')
plt.title('Strategy vs Dataset Performance Heatmap')
plt.savefig('strategy_dataset_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()