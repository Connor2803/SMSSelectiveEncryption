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
    df['Pattern_Disruption_%'] = pd.to_numeric(df['Pattern_Disruption_%'])
    df['Efficiency_Gain_%'] = pd.to_numeric(df['Efficiency_Gain_%'])
    
    return df

def create_comprehensive_plots(df, output_prefix='series_breaking_analysis'):
    """Create comprehensive visualization plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots (4x3 grid to add pattern disruption plots)
    fig = plt.figure(figsize=(24, 20))
    
    # 1. ASR Comparison (Original vs Broken)
    ax1 = plt.subplot(4, 3, 1)
    x_pos = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, df['Original_ASR'], width, 
                    label='Original ASR', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x_pos + width/2, df['Broken_ASR'], width,
                    label='Broken ASR', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Test Configuration')
    ax1.set_ylabel('Attack Success Rate')
    ax1.set_title('ASR Comparison: Original vs Broken Data')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in df.iterrows()], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. ASR Reduction Percentage
    ax2 = plt.subplot(4, 3, 2)
    colors = ['green' if x > 0 else 'red' for x in df['ASR_Reduction_%']]
    bars = ax2.bar(range(len(df)), df['ASR_Reduction_%'], color=colors, alpha=0.7)
    ax2.set_xlabel('Test Configuration')
    ax2.set_ylabel('ASR Reduction (%)')
    ax2.set_title('Security Improvement by Configuration')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in df.iterrows()], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 3. Pattern Disruption (NEW!)
    ax3 = plt.subplot(4, 3, 3)
    bars = ax3.bar(range(len(df)), df['Pattern_Disruption_%'], color='purple', alpha=0.7)
    ax3.set_xlabel('Test Configuration')
    ax3.set_ylabel('Pattern Disruption (%)')
    ax3.set_title('Pattern Breaking Effectiveness')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in df.iterrows()], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Total Preservation
    ax4 = plt.subplot(4, 3, 4)
    bars = ax4.bar(range(len(df)), df['Total_Preservation_%'], color='forestgreen', alpha=0.8)
    ax4.set_xlabel('Test Configuration')
    ax4.set_ylabel('Total Preservation (%)')
    ax4.set_title('Data Integrity Preservation')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in df.iterrows()], rotation=45, ha='right')
    ax4.set_ylim(99.8, 100.0)  # Focus on the relevant range
    ax4.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                f'{height:.2f}%', ha='center', va='top', fontsize=9)
    
    # 5. Efficiency Gain
    ax5 = plt.subplot(4, 3, 5)
    colors = ['blue' if x > 0 else 'orange' for x in df['Efficiency_Gain_%']]
    bars = ax5.bar(range(len(df)), df['Efficiency_Gain_%'], color=colors, alpha=0.7)
    ax5.set_xlabel('Test Configuration')
    ax5.set_ylabel('Efficiency Gain (%)')
    ax5.set_title('Processing Speed Improvement')
    ax5.set_xticks(range(len(df)))
    ax5.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                         for _, row in df.iterrows()], rotation=45, ha='right')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.1),
                f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 6. Multi-Metric Performance by Tolerance (NEW!)
    ax6 = plt.subplot(4, 3, 6)
    sliding_data = df[df['Approach'] == 'Sliding Window Breaking']
    adaptive_data = df[df['Approach'] == 'Adaptive Pattern Breaking']
    
    # Plot multiple metrics
    ax6.plot(sliding_data['Tolerance_%'], sliding_data['ASR_Reduction_%'], 
             'o-', label='Security (Sliding)', linewidth=2, markersize=8, color='red')
    ax6.plot(adaptive_data['Tolerance_%'], adaptive_data['ASR_Reduction_%'], 
             's-', label='Security (Adaptive)', linewidth=2, markersize=8, color='darkred')
    
    # Add pattern disruption on secondary y-axis
    ax6_twin = ax6.twinx()
    ax6_twin.plot(sliding_data['Tolerance_%'], sliding_data['Pattern_Disruption_%'], 
                  '^--', label='Pattern (Sliding)', linewidth=2, markersize=8, color='purple')
    ax6_twin.plot(adaptive_data['Tolerance_%'], adaptive_data['Pattern_Disruption_%'], 
                  'v--', label='Pattern (Adaptive)', linewidth=2, markersize=8, color='darkviolet')
    
    ax6.set_xlabel('Tolerance (%)')
    ax6.set_ylabel('ASR Reduction (%)', color='red')
    ax6_twin.set_ylabel('Pattern Disruption (%)', color='purple')
    ax6.set_title('Security vs Pattern Disruption by Tolerance')
    ax6.legend(loc='upper left')
    ax6_twin.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 7. Success Status Distribution
    ax7 = plt.subplot(4, 3, 7)
    success_counts = df['Success_Status'].value_counts()
    colors_pie = ['lightgreen', 'lightcoral', 'gold']
    wedges, texts, autotexts = ax7.pie(success_counts.values, labels=success_counts.index, 
                                       autopct='%1.1f%%', colors=colors_pie[:len(success_counts)])
    ax7.set_title('Overall Success Rate Distribution')
    
    # 8. Enhanced Multi-metric radar chart (including pattern disruption)
    ax8 = plt.subplot(4, 3, 8, projection='polar')
    
    # Prepare data for radar chart (4 metrics now)
    metrics = ['ASR_Reduction_%', 'Total_Preservation_%', 'Pattern_Disruption_%', 'Efficiency_Gain_%']
    sliding_means = sliding_data[metrics].mean()
    adaptive_means = adaptive_data[metrics].mean()
    
    # Normalize metrics for better visualization
    sliding_normalized = [
        max(0, sliding_means['ASR_Reduction_%'] / 30 * 100),  # Scale to 0-100
        sliding_means['Total_Preservation_%'],
        sliding_means['Pattern_Disruption_%'] * 10,  # Scale up for visibility
        max(0, (sliding_means['Efficiency_Gain_%'] + 5) / 10 * 100)  # Scale to 0-100
    ]
    adaptive_normalized = [
        max(0, adaptive_means['ASR_Reduction_%'] / 30 * 100),
        adaptive_means['Total_Preservation_%'],
        adaptive_means['Pattern_Disruption_%'] * 10,
        max(0, (adaptive_means['Efficiency_Gain_%'] + 5) / 10 * 100)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    sliding_normalized.append(sliding_normalized[0])
    adaptive_normalized.append(adaptive_normalized[0])
    
    ax8.plot(angles, sliding_normalized, 'o-', linewidth=2, label='Sliding Window')
    ax8.fill(angles, sliding_normalized, alpha=0.25)
    ax8.plot(angles, adaptive_normalized, 's-', linewidth=2, label='Adaptive Pattern')
    ax8.fill(angles, adaptive_normalized, alpha=0.25)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(['Security\nImprovement', 'Data\nPreservation', 'Pattern\nDisruption', 'Efficiency\nGain'])
    ax8.set_title('Multi-Metric Performance Comparison\n(Including Pattern Disruption)')
    ax8.legend()
    ax8.set_ylim(0, 100)
    
    # 9. Correlation matrix (expanded)
    ax9 = plt.subplot(4, 3, 9)
    corr_data = df[['Tolerance_%', 'ASR_Reduction_%', 'Total_Preservation_%', 'Pattern_Disruption_%', 'Efficiency_Gain_%']]
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax9, fmt='.2f')
    ax9.set_title('Expanded Metric Correlations')
    
    # 10. Security vs Pattern Disruption Scatter Plot (NEW!)
    ax10 = plt.subplot(4, 3, 10)
    sliding_scatter = ax10.scatter(sliding_data['Pattern_Disruption_%'], sliding_data['ASR_Reduction_%'], 
                                   c=sliding_data['Tolerance_%'], cmap='Reds', s=100, alpha=0.7, 
                                   marker='o', label='Sliding Window')
    adaptive_scatter = ax10.scatter(adaptive_data['Pattern_Disruption_%'], adaptive_data['ASR_Reduction_%'], 
                                    c=adaptive_data['Tolerance_%'], cmap='Blues', s=100, alpha=0.7, 
                                    marker='s', label='Adaptive Pattern')
    
    ax10.set_xlabel('Pattern Disruption (%)')
    ax10.set_ylabel('ASR Reduction (%)')
    ax10.set_title('Security vs Pattern Disruption\n(Color = Tolerance)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax10.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(sliding_scatter, ax=ax10)
    cbar.set_label('Tolerance (%)')
    
    # 11. Performance Score with Pattern Disruption Weight (NEW!)
    ax11 = plt.subplot(4, 3, 11)
    # Calculate enhanced composite score
    df['Enhanced_Performance_Score'] = (
        df['ASR_Reduction_%'] * 0.3 +  # 30% weight on security
        (df['Total_Preservation_%'] - 99) * 20 * 0.25 +  # 25% weight on preservation
        df['Pattern_Disruption_%'] * 0.25 +  # 25% weight on pattern disruption
        df['Efficiency_Gain_%'] * 0.2  # 20% weight on efficiency
    )
    
    bars = ax11.bar(range(len(df)), df['Enhanced_Performance_Score'], 
                    color=['green' if x > 0 else 'red' for x in df['Enhanced_Performance_Score']], alpha=0.8)
    ax11.set_xlabel('Test Configuration')
    ax11.set_ylabel('Enhanced Performance Score')
    ax11.set_title('Overall Performance Ranking\n(Including Pattern Disruption)')
    ax11.set_xticks(range(len(df)))
    ax11.set_xticklabels([f"{row['Approach'][:8]}\n{row['Tolerance_%']}%" 
                          for _, row in df.iterrows()], rotation=45, ha='right')
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax11.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.5),
                 f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 12. Algorithm Comparison Summary (NEW!)
    ax12 = plt.subplot(4, 3, 12)
    
    # Create summary metrics by approach
    summary_data = df.groupby('Approach').agg({
        'ASR_Reduction_%': 'mean',
        'Total_Preservation_%': 'mean',
        'Pattern_Disruption_%': 'mean',
        'Efficiency_Gain_%': 'mean'
    }).reset_index()
    
    x_labels = ['Security\nImprovement', 'Data\nPreservation', 'Pattern\nDisruption', 'Efficiency\nGain']
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    sliding_values = [
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['ASR_Reduction_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Total_Preservation_%'].iloc[0] - 99,
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Pattern_Disruption_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Sliding Window Breaking']['Efficiency_Gain_%'].iloc[0]
    ]
    
    adaptive_values = [
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['ASR_Reduction_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Total_Preservation_%'].iloc[0] - 99,
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Pattern_Disruption_%'].iloc[0],
        summary_data[summary_data['Approach'] == 'Adaptive Pattern Breaking']['Efficiency_Gain_%'].iloc[0]
    ]
    
    bars1 = ax12.bar(x_pos - width/2, sliding_values, width, label='Sliding Window', 
                     color='steelblue', alpha=0.8)
    bars2 = ax12.bar(x_pos + width/2, adaptive_values, width, label='Adaptive Pattern', 
                     color='darkorange', alpha=0.8)
    
    ax12.set_xlabel('Performance Metrics')
    ax12.set_ylabel('Average Performance')
    ax12.set_title('Algorithm Comparison Summary\n(All Metrics)')
    ax12.set_xticks(x_pos)
    ax12.set_xticklabels(x_labels)
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    ax12.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i == 1:  # Data preservation (show actual percentage)
                ax12.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height + 99:.2f}%', ha='center', va='bottom', fontsize=8)
            else:
                ax12.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.2),
                         f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_thesis_summary_plot(df, output_prefix='thesis_summary'):
    """Create a clean summary plot for thesis presentation"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Security vs Pattern Disruption
    sliding_data = df[df['Approach'] == 'Sliding Window Breaking']
    adaptive_data = df[df['Approach'] == 'Adaptive Pattern Breaking']
    
    x = np.arange(len(sliding_data))
    width = 0.35
    
    # Security improvement
    bars1 = ax1.bar(x - width/2, sliding_data['ASR_Reduction_%'], width, 
                    label='Sliding Window', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, adaptive_data['ASR_Reduction_%'], width,
                    label='Adaptive Pattern', color='darkorange', alpha=0.8)
    
    ax1.set_xlabel('Tolerance Level (%)')
    ax1.set_ylabel('ASR Reduction (%)')
    ax1.set_title('Security Improvement by Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['5%', '10%', '15%'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold')
    
    # 2. Pattern Disruption Comparison (NEW!)
    ax2.bar(x - width/2, sliding_data['Pattern_Disruption_%'], width, 
            label='Sliding Window', color='purple', alpha=0.8)
    ax2.bar(x + width/2, adaptive_data['Pattern_Disruption_%'], width,
            label='Adaptive Pattern', color='darkviolet', alpha=0.8)
    
    ax2.set_xlabel('Tolerance Level (%)')
    ax2.set_ylabel('Pattern Disruption (%)')
    ax2.set_title('Pattern Breaking Effectiveness')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['5%', '10%', '15%'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for approach_data, color in [(sliding_data, 'purple'), (adaptive_data, 'darkviolet')]:
        for i, (_, row) in enumerate(approach_data.iterrows()):
            offset = -width/2 if color == 'purple' else width/2
            ax2.text(i + offset, row['Pattern_Disruption_%'] + 0.01,
                    f'{row["Pattern_Disruption_%"]:.2f}%', ha='center', va='bottom', 
                    fontweight='bold')
    
    # 3. Multi-Metric Summary (Updated)
    ax3.clear()
    
    # Calculate average performance for each approach
    sliding_avg = sliding_data[['ASR_Reduction_%', 'Total_Preservation_%', 'Pattern_Disruption_%', 'Efficiency_Gain_%']].mean()
    adaptive_avg = adaptive_data[['ASR_Reduction_%', 'Total_Preservation_%', 'Pattern_Disruption_%', 'Efficiency_Gain_%']].mean()
    
    metrics = ['Security\nImprovement', 'Data\nPreservation', 'Pattern\nDisruption', 'Efficiency\nGain']
    sliding_values = [
        sliding_avg['ASR_Reduction_%'],
        sliding_avg['Total_Preservation_%'] - 99,  # Normalize to show variation
        sliding_avg['Pattern_Disruption_%'],
        sliding_avg['Efficiency_Gain_%']
    ]
    adaptive_values = [
        adaptive_avg['ASR_Reduction_%'],
        adaptive_avg['Total_Preservation_%'] - 99,
        adaptive_avg['Pattern_Disruption_%'],
        adaptive_avg['Efficiency_Gain_%']
    ]
    
    x_pos = np.arange(len(metrics))
    bars1 = ax3.bar(x_pos - width/2, sliding_values, width, label='Sliding Window', 
                    color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, adaptive_values, width, label='Adaptive Pattern', 
                    color='darkorange', alpha=0.8)
    
    ax3.set_xlabel('Performance Metrics')
    ax3.set_ylabel('Average Performance')
    ax3.set_title('Overall Algorithm Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        for bar, values in [(bar1, sliding_values), (bar2, adaptive_values)]:
            height = bar.get_height()
            if i == 1:  # Data preservation
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{values[i] + 99:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
            else:
                ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.2),
                        f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                        fontweight='bold', fontsize=8)
    
    # 4. Best Configuration Highlight (Updated)
    best_config = df.loc[df['ASR_Reduction_%'].idxmax()]
    
    metrics = ['Security\nImprovement', 'Pattern\nDisruption', 'Data\nPreservation', 'Efficiency']
    values = [
        best_config['ASR_Reduction_%'],
        best_config['Pattern_Disruption_%'],
        best_config['Total_Preservation_%'] - 99,  # Normalize to show variation
        best_config['Efficiency_Gain_%']
    ]
    
    bars = ax4.bar(metrics, values, color=['red', 'purple', 'green', 'blue'], alpha=0.7)
    ax4.set_ylabel('Performance Metrics')
    ax4.set_title(f'Best Configuration: {best_config["Approach"]}\n({best_config["Tolerance_%"]}% Tolerance)')
    ax4.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 2:  # Data preservation
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{best_config["Total_Preservation_%"]:.2f}%', 
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.2),
                    f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_thesis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load the data
    csv_file = "series_breaking_results_20250809_010343_avg_100_runs.csv"
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
    print(f"  - Pattern Disruption: {best_security['Pattern_Disruption_%']:.2f}%")
    print(f"  - Data Preservation: {best_security['Total_Preservation_%']:.2f}%")
    
    # Check if any configuration achieves both security and pattern disruption
    effective_configs = df[(df['ASR_Reduction_%'] > 0) & (df['Pattern_Disruption_%'] > 0)]
    if len(effective_configs) > 0:
        print(f"\nConfigurations with both security and pattern benefits: {len(effective_configs)}")
        for _, row in effective_configs.iterrows():
            print(f"  - {row['Approach']} ({row['Tolerance_%']}%): {row['ASR_Reduction_%']:.1f}% security, {row['Pattern_Disruption_%']:.2f}% pattern disruption")
    else:
        print(f"\nNOTE: No configurations achieved measurable pattern disruption (all 0.00%)")
        print("This suggests the pattern detection algorithm may need refinement.")