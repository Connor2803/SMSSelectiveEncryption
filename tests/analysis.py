import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_uniqueness_results():
    results_file = "results/combined_results.csv"
    
    if not os.path.exists(results_file):
        print("No results file found. Run the test script first.")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} test results")
    
    # Basic statistics
    print("\n=== ASR Statistics ===")
    print(df.groupby(['Strategy', 'Dataset'])['ASR'].agg(['mean', 'std', 'min', 'max']))
    
    print("\n=== Privacy Statistics ===")
    print(df.groupby(['Strategy', 'Dataset'])['Privacy'].agg(['mean', 'std', 'min', 'max']))
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ASR by strategy
    df.groupby('Strategy')['ASR'].mean().plot(kind='bar', ax=axes[0,0], title='ASR by Strategy')
    axes[0,0].set_ylabel('ASR')
    
    # Privacy by strategy
    df.groupby('Strategy')['Privacy'].mean().plot(kind='bar', ax=axes[0,1], title='Privacy by Strategy')
    axes[0,1].set_ylabel('Privacy')
    
    # ASR vs Encryption Ratio
    df.groupby('Ratio')['ASR'].mean().plot(ax=axes[1,0], title='ASR vs Encryption Ratio', marker='o')
    axes[1,0].set_xlabel('Encryption Ratio (%)')
    axes[1,0].set_ylabel('ASR')
    
    # Privacy vs Encryption Ratio
    df.groupby('Ratio')['Privacy'].mean().plot(ax=axes[1,1], title='Privacy vs Encryption Ratio', marker='s')
    axes[1,1].set_xlabel('Encryption Ratio (%)')
    axes[1,1].set_ylabel('Privacy')
    
    plt.tight_layout()
    plt.savefig('results/uniqueness_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to results/uniqueness_analysis.png")

if __name__ == "__main__":
    analyze_uniqueness_results()