import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
water_data = {
    'Encryption_Ratio': [0, 20, 40, 60, 80, 100],
    'ASR': [0.890, 0.677, 0.596, 0.511, 0.370, 0.311],
    'Standard_Error': [0.050, 0.033, 0.029, 0.032, 0.032, 0.029]
}

electricity_data = {
    'Encryption_Ratio': [0, 20, 40, 60, 80, 100],
    'ASR': [1.000, 0.860, 0.560, 0.400, 0.150, 0.000],
    'Standard_Error': [0.000, 0.040, 0.037, 0.033, 0.050, 0.000],
    'Encryption_Time': [0.00, 50.49, 91.75, 125.42, 166.22, 217.83]
}

# Create DataFrames
df_water = pd.DataFrame(water_data)
df_electricity = pd.DataFrame(electricity_data)

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Selective Encryption Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: ASR Comparison
ax1.errorbar(df_water['Encryption_Ratio'], df_water['ASR'], 
             yerr=df_water['Standard_Error'], 
             marker='o', linewidth=2, markersize=8, capsize=5,
             label='Water (Entropy-based)', color='blue')
ax1.errorbar(df_electricity['Encryption_Ratio'], df_electricity['ASR'], 
             yerr=df_electricity['Standard_Error'], 
             marker='s', linewidth=2, markersize=8, capsize=5,
             label='Electricity (Uniqueness-based)', color='red')
ax1.set_xlabel('Encryption Ratio (%)')
ax1.set_ylabel('Attack Success Rate (ASR)')
ax1.set_title('ASR vs Encryption Ratio')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0, 1.1)

# Plot 2: ASR Reduction Rate
water_reduction = [(df_water['ASR'].iloc[0] - asr) / df_water['ASR'].iloc[0] * 100 
                   for asr in df_water['ASR']]
electricity_reduction = [(df_electricity['ASR'].iloc[0] - asr) / df_electricity['ASR'].iloc[0] * 100 
                        for asr in df_electricity['ASR']]

ax2.plot(df_water['Encryption_Ratio'], water_reduction, 
         marker='o', linewidth=2, markersize=8, label='Water (Entropy-based)', color='blue')
ax2.plot(df_electricity['Encryption_Ratio'], electricity_reduction, 
         marker='s', linewidth=2, markersize=8, label='Electricity (Uniqueness-based)', color='red')
ax2.set_xlabel('Encryption Ratio (%)')
ax2.set_ylabel('ASR Reduction (%)')
ax2.set_title('Privacy Improvement (ASR Reduction from 0% encryption)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Encryption Time vs Ratio (Electricity only)
ax3.plot(df_electricity['Encryption_Ratio'], df_electricity['Encryption_Time'], 
         marker='s', linewidth=2, markersize=8, color='red')
ax3.set_xlabel('Encryption Ratio (%)')
ax3.set_ylabel('Encryption Time (seconds)')
ax3.set_title('Encryption Time vs Ratio (Uniqueness-based on Electricity)')
ax3.grid(True, alpha=0.3)

# Plot 4: Privacy-Performance Trade-off
ax4.scatter(df_electricity['Encryption_Time'], df_electricity['ASR'], 
           s=100, alpha=0.7, color='red', edgecolors='black')
for i, ratio in enumerate(df_electricity['Encryption_Ratio']):
    ax4.annotate(f'{ratio}%', 
                (df_electricity['Encryption_Time'].iloc[i], df_electricity['ASR'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
ax4.set_xlabel('Encryption Time (seconds)')
ax4.set_ylabel('Attack Success Rate (ASR)')
ax4.set_title('Privacy-Performance Trade-off (Electricity/Uniqueness)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot as PNG with high DPI
plt.savefig('selective_encryption_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Plot saved as 'selective_encryption_analysis.png'")

# Also save individual plots
fig1, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(df_water['Encryption_Ratio'], df_water['ASR'], 
            yerr=df_water['Standard_Error'], 
            marker='o', linewidth=2, markersize=8, capsize=5,
            label='Water (Entropy-based)', color='blue')
ax.errorbar(df_electricity['Encryption_Ratio'], df_electricity['ASR'], 
            yerr=df_electricity['Standard_Error'], 
            marker='s', linewidth=2, markersize=8, capsize=5,
            label='Electricity (Uniqueness-based)', color='red')
ax.set_xlabel('Encryption Ratio (%)')
ax.set_ylabel('Attack Success Rate (ASR)')
ax.set_title('ASR vs Encryption Ratio Comparison')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('asr_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("ASR comparison plot saved as 'asr_comparison.png'")

plt.show()

# Print summary statistics
print("=== SUMMARY STATISTICS ===")
print("\nWater Dataset (Entropy-based):")
print(f"ASR reduction from 0% to 100%: {(df_water['ASR'].iloc[0] - df_water['ASR'].iloc[-1]) / df_water['ASR'].iloc[0] * 100:.1f}%")
print(f"Best ASR achieved: {df_water['ASR'].min():.3f} at {df_water.loc[df_water['ASR'].idxmin(), 'Encryption_Ratio']}% encryption")

print("\nElectricity Dataset (Uniqueness-based):")
print(f"ASR reduction from 0% to 100%: {(df_electricity['ASR'].iloc[0] - df_electricity['ASR'].iloc[-1]) / df_electricity['ASR'].iloc[0] * 100:.1f}%")
print(f"Best ASR achieved: {df_electricity['ASR'].min():.3f} at {df_electricity.loc[df_electricity['ASR'].idxmin(), 'Encryption_Ratio']}% encryption")
print(f"Maximum encryption time: {df_electricity['Encryption_Time'].max():.1f} seconds")

# Privacy efficiency analysis
print("\n=== PRIVACY EFFICIENCY ANALYSIS ===")
print("ASR per 20% encryption increase:")
for i in range(1, len(df_water)):
    water_efficiency = (df_water['ASR'].iloc[i-1] - df_water['ASR'].iloc[i]) / 20
    electricity_efficiency = (df_electricity['ASR'].iloc[i-1] - df_electricity['ASR'].iloc[i]) / 20
    print(f"{df_water['Encryption_Ratio'].iloc[i-1]}%-{df_water['Encryption_Ratio'].iloc[i]}%: "
          f"Water={water_efficiency:.4f}, Electricity={electricity_efficiency:.4f}")