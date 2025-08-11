import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_process_data(csv_file):
    """Load CSV data and extract the results section"""
    import pandas as pd
    
    # Read all lines first
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    
    # Find the header line
    header_line = -1
    for i, line in enumerate(lines):
        if line.startswith('Approach,'):
            header_line = i
            break
    
    if header_line == -1:
        raise ValueError("Could not find header line starting with 'Approach,'")
    
    # Find where data ends (empty line or summary starts)
    data_end = len(lines)
    for i in range(header_line + 1, len(lines)):
        if lines[i].strip() == '' or 'Summary Statistics' in lines[i]:
            data_end = i
            break
    
    # Read only the data section
    data_lines = lines[header_line:data_end]
    
    # Write to temporary string and read with pandas
    from io import StringIO
    data_string = ''.join(data_lines)
    df = pd.read_csv(StringIO(data_string))
    
    print(f"Data loaded: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Sample data:")
    print(df.head())
    
    # Clean and convert data types
    df['Tolerance_%'] = pd.to_numeric(df['Tolerance_%'])
    df['Original_ASR'] = pd.to_numeric(df['Original_ASR'])
    df['Broken_ASR'] = pd.to_numeric(df['Broken_ASR'])
    df['ASR_Improvement'] = pd.to_numeric(df['ASR_Improvement'])
    df['ASR_Reduction_%'] = pd.to_numeric(df['ASR_Reduction_%'])
    df['Total_Preservation_%'] = pd.to_numeric(df['Total_Preservation_%'])
    df['Efficiency_Gain_%'] = pd.to_numeric(df['Efficiency_Gain_%'])
    
    # Add Pattern_Disruption_% column if it doesn't exist (set to 0.0)
    if 'Pattern_Disruption_%' not in df.columns:
        print("Note: Pattern_Disruption_% column not found. Adding with 0.0 values.")
        df['Pattern_Disruption_%'] = 0.0
    else:
        df['Pattern_Disruption_%'] = pd.to_numeric(df['Pattern_Disruption_%'])
    
    return df

def create_comprehensive_plots(df, output_prefix='series_breaking_analysis'):
    """Create comprehensive visualization plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with larger size to accommodate subplots
    fig = plt.figure(figsize=(28, 24))
    
    # Split data by approach
    sliding_data = df[df['Approach'] == 'Sliding Window Breaking']
    adaptive_data = df[df['Approach'] == 'Adaptive Pattern Breaking']
    
    # 1. ASR Reduction by Tolerance (Line Plot)
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(sliding_data['Tolerance_%'], sliding_data['ASR_Reduction_%'], 
             'o-', label='Sliding Window', linewidth=2, markersize=6, color='steelblue')
    ax1.plot(adaptive_data['Tolerance_%'], adaptive_data['ASR_Reduction_%'], 
             's-', label='Adaptive Pattern', linewidth=2, markersize=6, color='darkorange')
    ax1.set_xlabel('Tolerance (%)')
    ax1.set_ylabel('ASR Reduction (%)')
    ax1.set_title('Security Improvement vs Tolerance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 2. Data Preservation by Tolerance
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(sliding_data['Tolerance_%'], sliding_data['Total_Preservation_%'], 
             'o-', label='Sliding Window', linewidth=2, markersize=6, color='green')
    ax2.plot(adaptive_data['Tolerance_%'], adaptive_data['Total_Preservation_%'], 
             's-', label='Adaptive Pattern', linewidth=2, markersize=6, color='darkgreen')
    ax2.set_xlabel('Tolerance (%)')
    ax2.set_ylabel('Data Preservation (%)')
    ax2.set_title('Data Integrity vs Tolerance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    
    # 3. Security vs Preservation Trade-off
    ax3 = plt.subplot(4, 3, 3)
    scatter1 = ax3.scatter(sliding_data['Total_Preservation_%'], sliding_data['ASR_Reduction_%'], 
                          c=sliding_data['Tolerance_%'], cmap='Reds', s=100, alpha=0.7, 
                          marker='o', label='Sliding Window')
    scatter2 = ax3.scatter(adaptive_data['Total_Preservation_%'], adaptive_data['ASR_Reduction_%'], 
                          c=adaptive_data['Tolerance_%'], cmap='Blues', s=100, alpha=0.7, 
                          marker='s', label='Adaptive Pattern')
    ax3.set_xlabel('Data Preservation (%)')
    ax3.set_ylabel('ASR Reduction (%)')
    ax3.set_title('Security vs Preservation Trade-off')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter1, ax=ax3)
    cbar.set_label('Tolerance (%)')
    
    # 4. Best Configurations Highlighting
    ax4 = plt.subplot(4, 3, 4)
    # Filter for successful configurations
    successful_configs = df[df['Success_Status'] == 'SUCCESS']
    
    colors = []
    for _, row in successful_configs.iterrows():
        if row['Approach'] == 'Sliding Window Breaking':
            colors.append('steelblue')
        else:
            colors.append('darkorange')
    
    bars = ax4.bar(range(len(successful_configs)), successful_configs['ASR_Reduction_%'], 
                   color=colors, alpha=0.8)
    ax4.set_xlabel('Successful Configurations')
    ax4.set_ylabel('ASR Reduction (%)')
    ax4.set_title('Performance of Successful Configurations')
    ax4.set_xticks(range(len(successful_configs)))
    ax4.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in successful_configs.iterrows()], 
                        rotation=45, ha='right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Algorithm Performance Summary
    ax5 = plt.subplot(4, 3, 5)
    summary_data = df.groupby('Approach').agg({
        'ASR_Reduction_%': 'mean',
        'Total_Preservation_%': 'mean',
        'Efficiency_Gain_%': 'mean'
    }).reset_index()
    
    x_labels = ['Security\nImprovement', 'Data\nPreservation', 'Efficiency\nGain']
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    sliding_values = [
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['ASR_Reduction_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Total_Preservation_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Efficiency_Gain_%'].iloc[0]
    ]
    
    adaptive_values = [
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['ASR_Reduction_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Total_Preservation_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Efficiency_Gain_%'].iloc[0]
    ]
    
    bars1 = ax5.bar(x_pos - width/2, sliding_values, width, label='Sliding Window', 
                    color='steelblue', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, adaptive_values, width, label='Adaptive Pattern', 
                    color='darkorange', alpha=0.8)
    
    ax5.set_xlabel('Performance Metrics')
    ax5.set_ylabel('Average Performance')
    ax5.set_title('Overall Algorithm Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(x_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Success Rate Distribution
    ax6 = plt.subplot(4, 3, 6)
    success_counts = df['Success_Status'].value_counts()
    colors_pie = ['lightgreen', 'gold', 'lightcoral'][:len(success_counts)]
    wedges, texts, autotexts = ax6.pie(success_counts.values, labels=success_counts.index, 
                                       autopct='%1.1f%%', colors=colors_pie)
    ax6.set_title('Success Rate Distribution')
    
    # 7. Tolerance Impact on Security
    ax7 = plt.subplot(4, 3, 7)
    # Group by tolerance ranges for cleaner visualization
    tolerance_ranges = ['5-15%', '20-30%', '35-45%', '50-60%', '65-75%', '80-90%']
    
    sliding_range_means = []
    adaptive_range_means = []
    
    for i, range_label in enumerate(tolerance_ranges):
        start = 5 + i * 15
        end = start + 10
        
        sliding_range = sliding_data[(sliding_data['Tolerance_%'] >= start) & 
                                   (sliding_data['Tolerance_%'] <= end)]
        adaptive_range = adaptive_data[(adaptive_data['Tolerance_%'] >= start) & 
                                     (adaptive_data['Tolerance_%'] <= end)]
        
        sliding_range_means.append(sliding_range['ASR_Reduction_%'].mean() if len(sliding_range) > 0 else 0)
        adaptive_range_means.append(adaptive_range['ASR_Reduction_%'].mean() if len(adaptive_range) > 0 else 0)
    
    x_pos = np.arange(len(tolerance_ranges))
    bars1 = ax7.bar(x_pos - width/2, sliding_range_means, width, label='Sliding Window', 
                    color='steelblue', alpha=0.8)
    bars2 = ax7.bar(x_pos + width/2, adaptive_range_means, width, label='Adaptive Pattern', 
                    color='darkorange', alpha=0.8)
    
    ax7.set_xlabel('Tolerance Ranges')
    ax7.set_ylabel('Average ASR Reduction (%)')
    ax7.set_title('Security Improvement by Tolerance Range')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(tolerance_ranges)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Correlation Matrix
    ax8 = plt.subplot(4, 3, 8)
    corr_data = df[['Tolerance_%', 'ASR_Reduction_%', 'Total_Preservation_%', 'Efficiency_Gain_%']]
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax8, fmt='.2f')
    ax8.set_title('Performance Metrics Correlation')
    
    # 9. Top Performers Highlight
    ax9 = plt.subplot(4, 3, 9)
    # Get top 5 performers by ASR reduction
    top_performers = df.nlargest(5, 'ASR_Reduction_%')
    
    colors = ['steelblue' if 'Sliding' in approach else 'darkorange' 
              for approach in top_performers['Approach']]
    
    bars = ax9.bar(range(len(top_performers)), top_performers['ASR_Reduction_%'], 
                   color=colors, alpha=0.8)
    ax9.set_xlabel('Top Configurations')
    ax9.set_ylabel('ASR Reduction (%)')
    ax9.set_title('Top 5 Security Performers')
    ax9.set_xticks(range(len(top_performers)))
    ax9.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in top_performers.iterrows()], 
                        rotation=45, ha='right', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 10. Efficiency Analysis
    ax10 = plt.subplot(4, 3, 10)
    ax10.plot(sliding_data['Tolerance_%'], sliding_data['Efficiency_Gain_%'], 
              'o-', label='Sliding Window', linewidth=2, markersize=6, color='blue')
    ax10.plot(adaptive_data['Tolerance_%'], adaptive_data['Efficiency_Gain_%'], 
              's-', label='Adaptive Pattern', linewidth=2, markersize=6, color='navy')
    ax10.set_xlabel('Tolerance (%)')
    ax10.set_ylabel('Efficiency Gain (%)')
    ax10.set_title('Processing Efficiency vs Tolerance')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 11. Warning Zone Analysis
    ax11 = plt.subplot(4, 3, 11)
    warning_configs = df[df['Success_Status'] == 'WARNING']
    
    if len(warning_configs) > 0:
        colors = ['red' if 'Sliding' in approach else 'darkred' 
                  for approach in warning_configs['Approach']]
        
        bars = ax11.bar(range(len(warning_configs)), warning_configs['Total_Preservation_%'], 
                       color=colors, alpha=0.8)
        ax11.set_xlabel('Warning Configurations')
        ax11.set_ylabel('Data Preservation (%)')
        ax11.set_title('Configurations Below 95% Preservation')
        ax11.set_xticks(range(len(warning_configs)))
        ax11.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                             for _, row in warning_configs.iterrows()], 
                            rotation=45, ha='right', fontsize=8)
        ax11.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    else:
        ax11.text(0.5, 0.5, 'No Warning\nConfigurations', ha='center', va='center', 
                 transform=ax11.transAxes, fontsize=14, fontweight='bold', color='green')
        ax11.set_title('Warning Zone Analysis')
    
    # 12. Optimal Range Identification
    ax12 = plt.subplot(4, 3, 12)
    
    # Define optimal configurations (high security, good preservation)
    optimal_configs = df[(df['ASR_Reduction_%'] > 10) & (df['Total_Preservation_%'] > 99)]
    
    if len(optimal_configs) > 0:
        colors = ['green' if 'Sliding' in approach else 'darkgreen' 
                  for approach in optimal_configs['Approach']]
        
        scatter = ax12.scatter(optimal_configs['Tolerance_%'], optimal_configs['ASR_Reduction_%'], 
                              c=optimal_configs['Total_Preservation_%'], cmap='Greens', 
                              s=150, alpha=0.8, edgecolors='black', linewidth=2)
        ax12.set_xlabel('Tolerance (%)')
        ax12.set_ylabel('ASR Reduction (%)')
        ax12.set_title('Optimal Configuration Zone\n(>10% Security, >99% Preservation)')
        ax12.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax12)
        cbar.set_label('Data Preservation (%)')
        
        # Annotate points
        for _, row in optimal_configs.iterrows():
            ax12.annotate(f"{row['Tolerance_%']}%", 
                         (row['Tolerance_%'], row['ASR_Reduction_%']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    else:
        ax12.text(0.5, 0.5, 'No Optimal\nConfigurations\nFound', ha='center', va='center', 
                 transform=ax12.transAxes, fontsize=14, fontweight='bold', color='red')
        ax12.set_title('Optimal Configuration Zone')
    
    # Adjust layout with more space
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, 
                       wspace=0.3, hspace=0.4)
    plt.savefig(f'{output_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_thesis_summary_plot(df, output_prefix='thesis_summary'):
    """Create a clean summary plot for thesis presentation"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Split data by approach
    sliding_data = df[df['Approach'] == 'Sliding Window Breaking']
    adaptive_data = df[df['Approach'] == 'Adaptive Pattern Breaking']
    
    # 1. Security Improvement Trends
    ax1.plot(sliding_data['Tolerance_%'], sliding_data['ASR_Reduction_%'], 
             'o-', label='Sliding Window', linewidth=3, markersize=8, color='steelblue')
    ax1.plot(adaptive_data['Tolerance_%'], adaptive_data['ASR_Reduction_%'], 
             's-', label='Adaptive Pattern', linewidth=3, markersize=8, color='darkorange')
    
    ax1.set_xlabel('Tolerance Level (%)', fontsize=12)
    ax1.set_ylabel('ASR Reduction (%)', fontsize=12)
    ax1.set_title('Security Improvement vs Tolerance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Highlight best performers
    best_sliding = sliding_data.loc[sliding_data['ASR_Reduction_%'].idxmax()]
    best_adaptive = adaptive_data.loc[adaptive_data['ASR_Reduction_%'].idxmax()]
    
    ax1.annotate(f'Best: {best_sliding["ASR_Reduction_%"]:.1f}%', 
                xy=(best_sliding['Tolerance_%'], best_sliding['ASR_Reduction_%']),
                xytext=(10, 10), textcoords='offset points', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    ax1.annotate(f'Best: {best_adaptive["ASR_Reduction_%"]:.1f}%', 
                xy=(best_adaptive['Tolerance_%'], best_adaptive['ASR_Reduction_%']),
                xytext=(10, -20), textcoords='offset points', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsalmon', alpha=0.7))
    
    # 2. Data Preservation Trends
    ax2.plot(sliding_data['Tolerance_%'], sliding_data['Total_Preservation_%'], 
             'o-', label='Sliding Window', linewidth=3, markersize=8, color='green')
    ax2.plot(adaptive_data['Tolerance_%'], adaptive_data['Total_Preservation_%'], 
             's-', label='Adaptive Pattern', linewidth=3, markersize=8, color='darkgreen')
    
    ax2.set_xlabel('Tolerance Level (%)', fontsize=12)
    ax2.set_ylabel('Data Preservation (%)', fontsize=12)
    ax2.set_title('Data Integrity vs Tolerance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, linewidth=2, label='95% Threshold')
    
    # 3. Algorithm Performance Summary
    summary_data = df.groupby('Approach').agg({
        'ASR_Reduction_%': 'mean',
        'Total_Preservation_%': 'mean',
        'Efficiency_Gain_%': 'mean'
    }).reset_index()
    
    metrics = ['Security\nImprovement', 'Data\nPreservation', 'Efficiency\nGain']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    sliding_values = [
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['ASR_Reduction_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Total_Preservation_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Efficiency_Gain_%'].iloc[0]
    ]
    
    adaptive_values = [
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['ASR_Reduction_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Total_Preservation_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Efficiency_Gain_%'].iloc[0]
    ]
    
    bars1 = ax3.bar(x_pos - width/2, sliding_values, width, label='Sliding Window', 
                    color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, adaptive_values, width, label='Adaptive Pattern', 
                    color='darkorange', alpha=0.8)
    
    ax3.set_xlabel('Performance Metrics', fontsize=12)
    ax3.set_ylabel('Average Performance', fontsize=12)
    ax3.set_title('Overall Algorithm Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        for bar, values in [(bar1, sliding_values), (bar2, adaptive_values)]:
            height = bar.get_height()
            if i == 1:  # Data preservation
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{values[i]:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            else:
                ax3.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                        f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                        fontweight='bold', fontsize=10)
    
    # 4. Key Findings Summary
    ax4.clear()
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Create text summary
    findings_text = f"""
KEY FINDINGS

🏆 BEST SECURITY PERFORMANCE:
   {best_sliding['Approach']} at {best_sliding['Tolerance_%']}%
   → {best_sliding['ASR_Reduction_%']:.1f}% ASR Reduction
   → {best_sliding['Total_Preservation_%']:.2f}% Data Preservation

📊 ALGORITHM COMPARISON:
   Sliding Window: {summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['ASR_Reduction_%'].iloc[0]:.1f}% avg security
   Adaptive Pattern: {summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['ASR_Reduction_%'].iloc[0]:.1f}% avg security

⚠️  TRADE-OFFS OBSERVED:
   Higher tolerance = Better security
   Higher tolerance = Lower data preservation
   
✅ SUCCESS RATE:
   {len(df[df['Success_Status'] == 'SUCCESS'])} successful configurations
   {len(df[df['Success_Status'] == 'WARNING'])} warning configurations
"""
    
    ax4.text(0.05, 0.95, findings_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    ax4.set_title('Research Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_thesis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load the data
    csv_file = "compiled.csv"
    df = load_and_process_data(csv_file)
    
    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Create comprehensive plots
    create_comprehensive_plots(df)
    
    # Create thesis summary plots
    create_thesis_summary_plot(df)
    
    print("\n=== KEY FINDINGS ===")
    best_security = df.loc[df['ASR_Reduction_%'].idxmax()]
    print(f"Best Security: {best_security['Approach']} at {best_security['Tolerance_%']}% tolerance")
    print(f"  - ASR Reduction: {best_security['ASR_Reduction_%']:.1f}%")
    print(f"  - Data Preservation: {best_security['Total_Preservation_%']:.2f}%")
    
    # Analyze optimal range
    optimal_configs = df[(df['ASR_Reduction_%'] > 20) & (df['Total_Preservation_%'] > 99)]
    if len(optimal_configs) > 0:
        print(f"\nOptimal configurations (>20% security, >99% preservation): {len(optimal_configs)}")
        for _, row in optimal_configs.iterrows():
            print(f"  - {row['Approach']} ({row['Tolerance_%']}%): {row['ASR_Reduction_%']:.1f}% security, {row['Total_Preservation_%']:.2f}% preservation")
    else:
        print("\nNo configurations meet the optimal criteria (>20% security, >99% preservation)")
        
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Success rate: {len(df[df['Success_Status'] == 'SUCCESS'])/len(df)*100:.1f}%")