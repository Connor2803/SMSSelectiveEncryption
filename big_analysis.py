import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

def analyze_method_effectiveness():
    """
    Analyze why sliding/adaptive methods appear to work randomly
    Generate per-file results and group datasets by performance patterns
    """
    
    # Load all CSV files
    csv_files = glob("*.csv")
    baseline_files = [f for f in csv_files if 'none' in f or 'baseline' in f or f in ['rerunAVG8.csv', 'resultsAVG8.csv', 'resultsAVG24.csv']]
    sliding_files = [f for f in csv_files if 'sliding' in f and f not in baseline_files]
    adaptive_files = [f for f in csv_files if 'adaptive' in f and f not in baseline_files]
    
    print(f"Found {len(baseline_files)} baseline files, {len(sliding_files)} sliding files, {len(adaptive_files)} adaptive files")
    
    # Load baseline data
    baseline_data = load_baseline_data(baseline_files)
    
    # Analyze each method
    sliding_analysis = analyze_method_vs_baseline(sliding_files, baseline_data, "sliding")
    adaptive_analysis = analyze_method_vs_baseline(adaptive_files, baseline_data, "adaptive")
    
    # Generate comprehensive analysis
    generate_effectiveness_report(sliding_analysis, adaptive_analysis, baseline_data)
    
    # Create per-file performance analysis
    create_per_file_analysis(sliding_files, adaptive_files, baseline_data)
    
    # Identify performance patterns
    identify_performance_patterns(sliding_analysis, adaptive_analysis)

def load_baseline_data(baseline_files):
    """Load and combine baseline data from multiple files"""
    baseline_data = {}
    
    for file in baseline_files:
        try:
            df = pd.read_csv(file)
            
            # Convert columns to numeric, handling errors
            if 'tolerance' in df.columns:
                df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
            if 'ratio' in df.columns:
                df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
            if 'asr' in df.columns:
                df['asr'] = pd.to_numeric(df['asr'], errors='coerce')
            
            # Filter for 'none' approach only
            if 'approach' in df.columns:
                df = df[df['approach'] == 'none']
            
            # Drop rows with NaN values
            df = df.dropna(subset=['tolerance', 'ratio', 'asr'])
            
            for _, row in df.iterrows():
                key = (float(row['tolerance']), float(row['ratio']))
                if key not in baseline_data:
                    baseline_data[key] = []
                baseline_data[key].append({
                    'asr': float(row['asr']),
                    'file': file,
                    'time': row.get('time', 'N/A')
                })
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Calculate average baseline for each tolerance/ratio combination
    baseline_avg = {}
    for key, values in baseline_data.items():
        asrs = [v['asr'] for v in values]
        if asrs:  # Only process if we have valid ASR values
            baseline_avg[key] = {
                'asr_mean': np.mean(asrs),
                'asr_std': np.std(asrs) if len(asrs) > 1 else 0.0,
                'count': len(asrs),
                'files': [v['file'] for v in values]
            }
    
    print(f"Loaded baseline data for {len(baseline_avg)} tolerance/ratio combinations")
    return baseline_avg

def analyze_method_vs_baseline(method_files, baseline_data, method_name):
    """Analyze how each method performs vs baseline"""
    analysis_results = {
        'improvements': [],
        'degradations': [],
        'no_change': [],
        'per_file_results': {},
        'tolerance_patterns': {},
        'ratio_patterns': {}
    }
    
    for file in method_files:
        try:
            df = pd.read_csv(file)
            
            # Convert columns to numeric
            df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
            df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
            df['asr'] = pd.to_numeric(df['asr'], errors='coerce')
            
            # Drop rows with NaN values
            df = df.dropna(subset=['tolerance', 'ratio', 'asr'])
            
            if len(df) == 0:
                print(f"Warning: No valid data in {file}")
                continue
            
            file_results = analyze_single_file(df, baseline_data, method_name, file)
            analysis_results['per_file_results'][file] = file_results
            
            # Categorize overall file performance
            valid_improvements = [r['improvement_pct'] for r in file_results if r['improvement_pct'] is not None]
            
            if not valid_improvements:
                print(f"Warning: No valid improvements found for {file}")
                continue
            
            avg_improvement = np.mean(valid_improvements)
            
            if avg_improvement > 5:  # >5% improvement
                analysis_results['improvements'].append({
                    'file': file,
                    'avg_improvement': avg_improvement,
                    'results': file_results
                })
            elif avg_improvement < -5:  # >5% degradation
                analysis_results['degradations'].append({
                    'file': file,
                    'avg_degradation': avg_improvement,
                    'results': file_results
                })
            else:
                analysis_results['no_change'].append({
                    'file': file,
                    'avg_change': avg_improvement,
                    'results': file_results
                })
                
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
    
    return analysis_results

def analyze_single_file(df, baseline_data, method_name, filename):
    """Analyze a single CSV file against baseline"""
    results = []
    
    for _, row in df.iterrows():
        tolerance = float(row['tolerance'])
        ratio = float(row['ratio'])
        method_asr = float(row['asr'])
        
        baseline_key = (tolerance, ratio)
        if baseline_key in baseline_data:
            baseline_asr = baseline_data[baseline_key]['asr_mean']
            improvement = baseline_asr - method_asr  # Lower ASR is better
            improvement_pct = (improvement / baseline_asr) * 100 if baseline_asr > 0 else 0
            
            results.append({
                'tolerance': tolerance,
                'ratio': ratio,
                'method_asr': method_asr,
                'baseline_asr': baseline_asr,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'file': filename
            })
        else:
            # No baseline data for this combination
            results.append({
                'tolerance': tolerance,
                'ratio': ratio,
                'method_asr': method_asr,
                'baseline_asr': None,
                'improvement': None,
                'improvement_pct': None,
                'file': filename
            })
    
    return results

def generate_effectiveness_report(sliding_analysis, adaptive_analysis, baseline_data):
    """Generate comprehensive effectiveness report"""
    
    print("=" * 80)
    print("SERIES BREAKING METHODS EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    
    # Overall summary
    print(f"\nOVERALL SUMMARY:")
    print(f"Baseline configurations tested: {len(baseline_data)}")
    print(f"Sliding method files: {len(sliding_analysis['per_file_results'])}")
    print(f"Adaptive method files: {len(adaptive_analysis['per_file_results'])}")
    
    # Sliding results
    print(f"\nSLIDING METHOD RESULTS:")
    print(f"  Files showing improvement (>5%): {len(sliding_analysis['improvements'])}")
    print(f"  Files showing degradation (<-5%): {len(sliding_analysis['degradations'])}")
    print(f"  Files with no significant change: {len(sliding_analysis['no_change'])}")
    
    if sliding_analysis['improvements']:
        print(f"  Best improvement: {max([f['avg_improvement'] for f in sliding_analysis['improvements']]):.1f}%")
        best_file = max(sliding_analysis['improvements'], key=lambda x: x['avg_improvement'])
        print(f"    File: {best_file['file']}")
    
    if sliding_analysis['degradations']:
        print(f"  Worst degradation: {min([f['avg_degradation'] for f in sliding_analysis['degradations']]):.1f}%")
        worst_file = min(sliding_analysis['degradations'], key=lambda x: x['avg_degradation'])
        print(f"    File: {worst_file['file']}")
    
    # Adaptive results
    print(f"\nADAPTIVE METHOD RESULTS:")
    print(f"  Files showing improvement (>5%): {len(adaptive_analysis['improvements'])}")
    print(f"  Files showing degradation (<-5%): {len(adaptive_analysis['degradations'])}")
    print(f"  Files with no significant change: {len(adaptive_analysis['no_change'])}")
    
    if adaptive_analysis['improvements']:
        print(f"  Best improvement: {max([f['avg_improvement'] for f in adaptive_analysis['improvements']]):.1f}%")
        best_file = max(adaptive_analysis['improvements'], key=lambda x: x['avg_improvement'])
        print(f"    File: {best_file['file']}")
    
    if adaptive_analysis['degradations']:
        print(f"  Worst degradation: {min([f['avg_degradation'] for f in adaptive_analysis['degradations']]):.1f}%")
        worst_file = min(adaptive_analysis['degradations'], key=lambda x: x['avg_degradation'])
        print(f"    File: {worst_file['file']}")

def create_per_file_analysis(sliding_files, adaptive_files, baseline_data):
    """Create detailed per-file analysis"""
    
    # Create output directory
    os.makedirs("analysis_results", exist_ok=True)
    
    # Analyze each file individually
    all_results = []
    
    # Process sliding files
    for file in sliding_files:
        try:
            df = pd.read_csv(file)
            file_analysis = analyze_file_characteristics(df, file, "sliding", baseline_data)
            if file_analysis:  # Only add if analysis was successful
                all_results.append(file_analysis)
        except Exception as e:
            print(f"Error processing sliding file {file}: {e}")
    
    # Process adaptive files
    for file in adaptive_files:
        try:
            df = pd.read_csv(file)
            file_analysis = analyze_file_characteristics(df, file, "adaptive", baseline_data)
            if file_analysis:  # Only add if analysis was successful
                all_results.append(file_analysis)
        except Exception as e:
            print(f"Error processing adaptive file {file}: {e}")
    
    if not all_results:
        print("No valid results to analyze")
        return None
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("analysis_results/per_file_analysis.csv", index=False)
    
    # Create visualization
    create_per_file_visualizations(results_df)
    
    return results_df

def analyze_file_characteristics(df, filename, method, baseline_data):
    """Analyze characteristics of a single file"""
    
    try:
        # Convert to numeric
        df['tolerance'] = pd.to_numeric(df['tolerance'], errors='coerce')
        df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
        df['asr'] = pd.to_numeric(df['asr'], errors='coerce')
        
        # Drop NaN rows
        df = df.dropna(subset=['tolerance', 'ratio', 'asr'])
        
        if len(df) == 0:
            print(f"No valid data in {filename}")
            return None
        
        # Extract parameters from filename
        tolerance = extract_tolerance_from_filename(filename)
        
        # Calculate performance metrics
        improvements = []
        for _, row in df.iterrows():
            baseline_key = (float(row['tolerance']), float(row['ratio']))
            if baseline_key in baseline_data:
                baseline_asr = baseline_data[baseline_key]['asr_mean']
                improvement = ((baseline_asr - row['asr']) / baseline_asr) * 100 if baseline_asr > 0 else 0
                improvements.append(improvement)
        
        if not improvements:
            print(f"No matching baseline data for {filename}")
            return None
        
        # File characteristics
        avg_improvement = np.mean(improvements)
        improvement_std = np.std(improvements) if len(improvements) > 1 else 0
        max_improvement = max(improvements)
        min_improvement = min(improvements)
        
        # Classify performance
        if avg_improvement > 10:
            performance_class = "Excellent"
        elif avg_improvement > 5:
            performance_class = "Good"
        elif avg_improvement > -5:
            performance_class = "Neutral"
        else:
            performance_class = "Poor"
        
        return {
            'filename': filename,
            'method': method,
            'tolerance': tolerance,
            'avg_improvement_pct': avg_improvement,
            'improvement_std': improvement_std,
            'max_improvement': max_improvement,
            'min_improvement': min_improvement,
            'performance_class': performance_class,
            'data_points': len(df),
            'consistent': improvement_std < 5,  # Low std = consistent performance
        }
        
    except Exception as e:
        print(f"Error analyzing characteristics of {filename}: {e}")
        return None

def extract_tolerance_from_filename(filename):
    """Extract tolerance value from filename"""
    # Extract tolerance from filename patterns like "adaptive0.3.csv", "sliding6.csv", etc.
    import re
    
    # Try different patterns
    patterns = [
        r'(\d+\.\d+)',  # decimal like 0.3
        r'(\d+)',       # integer like 6, 3
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, filename)
        if matches:
            try:
                return float(matches[-1])  # Take the last match
            except:
                continue
    
    return 0.0  # Default

def identify_performance_patterns(sliding_analysis, adaptive_analysis):
    """Identify what factors cause good vs poor performance"""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE PATTERN ANALYSIS")
    print("=" * 80)
    
    # Analyze sliding patterns
    print("\nSLIDING METHOD PATTERNS:")
    analyze_method_patterns(sliding_analysis, "Sliding")
    
    print("\nADAPTIVE METHOD PATTERNS:")
    analyze_method_patterns(adaptive_analysis, "Adaptive")

def analyze_method_patterns(analysis, method_name):
    """Analyze patterns for a specific method"""
    
    # Group by performance
    good_files = analysis['improvements']
    poor_files = analysis['degradations']
    neutral_files = analysis['no_change']
    
    print(f"\n{method_name} - GOOD PERFORMERS ({len(good_files)} files):")
    if good_files:
        for file_info in good_files:
            tolerance = extract_tolerance_from_filename(file_info['file'])
            print(f"  {file_info['file']}: {file_info['avg_improvement']:.1f}% improvement (T={tolerance})")
        
        # Find common characteristics
        tolerances = [extract_tolerance_from_filename(f['file']) for f in good_files]
        if tolerances:
            print(f"  Common tolerance range: {min(tolerances):.1f} - {max(tolerances):.1f}")
    else:
        print("  No good performers found")
    
    print(f"\n{method_name} - POOR PERFORMERS ({len(poor_files)} files):")
    if poor_files:
        for file_info in poor_files:
            tolerance = extract_tolerance_from_filename(file_info['file'])
            print(f"  {file_info['file']}: {file_info['avg_degradation']:.1f}% degradation (T={tolerance})")
        
        # Find common characteristics
        tolerances = [extract_tolerance_from_filename(f['file']) for f in poor_files]
        if tolerances:
            print(f"  Common tolerance range: {min(tolerances):.1f} - {max(tolerances):.1f}")
    else:
        print("  No poor performers found")
    
    print(f"\n{method_name} - NEUTRAL PERFORMERS ({len(neutral_files)} files):")
    if neutral_files:
        tolerances = [extract_tolerance_from_filename(f['file']) for f in neutral_files]
        if tolerances:
            print(f"  Files: {len(neutral_files)} with tolerance range: {min(tolerances):.1f} - {max(tolerances):.1f}")
    else:
        print("  No neutral performers found")

def create_per_file_visualizations(results_df):
    """Create visualizations for per-file analysis"""
    
    if len(results_df) == 0:
        print("No data to visualize")
        return
    
    # 1. Performance by tolerance
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    methods = results_df['method'].unique()
    colors = ['orange', 'blue']
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        plt.scatter(method_data['tolerance'], method_data['avg_improvement_pct'], 
                   alpha=0.7, label=method, s=60, color=colors[i % len(colors)])
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
    plt.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% improvement')
    plt.axhline(y=-5, color='orange', linestyle='--', alpha=0.5, label='5% degradation')
    
    plt.xlabel('Tolerance')
    plt.ylabel('Average Improvement (%)')
    plt.title('Performance vs Tolerance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Performance consistency
    plt.subplot(1, 2, 2)
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        plt.scatter(method_data['improvement_std'], method_data['avg_improvement_pct'], 
                   alpha=0.7, label=method, s=60, color=colors[i % len(colors)])
    
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=5, color='blue', linestyle='--', alpha=0.5, label='Consistency threshold')
    
    plt.xlabel('Improvement Standard Deviation')
    plt.ylabel('Average Improvement (%)')
    plt.title('Performance vs Consistency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_results/per_file_performance_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance classification heatmap
    if len(results_df) > 1:
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            values='avg_improvement_pct', 
            index='tolerance', 
            columns='method', 
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, 
                        fmt='.1f', cbar_kws={'label': 'Average Improvement (%)'})
            
            plt.title('Method Performance Heatmap by Tolerance')
            plt.tight_layout()
            plt.savefig('analysis_results/performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\n✓ Visualizations saved to analysis_results/")

def create_factor_analysis():
    """Create detailed factor analysis to identify what causes good/poor performance"""
    
    print(f"\n" + "=" * 80)
    print("FACTOR ANALYSIS - Why Methods Work 'Randomly'")
    print("=" * 80)
    
    # Load per-file analysis
    if os.path.exists("analysis_results/per_file_analysis.csv"):
        df = pd.read_csv("analysis_results/per_file_analysis.csv")
        
        if len(df) == 0:
            print("No data in per-file analysis")
            return
        
        # Factor 1: Tolerance ranges
        print(f"\nFACTOR 1: TOLERANCE RANGES")
        methods = df['method'].unique()
        
        for method in methods:
            method_data = df[df['method'] == method]
            
            good_performers = method_data[method_data['avg_improvement_pct'] > 5]
            poor_performers = method_data[method_data['avg_improvement_pct'] < -5]
            
            print(f"\n{method.upper()} METHOD:")
            if len(good_performers) > 0:
                print(f"  Good performance tolerance range: {good_performers['tolerance'].min():.1f} - {good_performers['tolerance'].max():.1f}")
                print(f"  Good performance files: {list(good_performers['filename'])}")
            else:
                print("  No good performers found")
            
            if len(poor_performers) > 0:
                print(f"  Poor performance tolerance range: {poor_performers['tolerance'].min():.1f} - {poor_performers['tolerance'].max():.1f}")
                print(f"  Poor performance files: {list(poor_performers['filename'])}")
            else:
                print("  No poor performers found")
        
        # Factor 2: Consistency
        print(f"\nFACTOR 2: CONSISTENCY PATTERNS")
        consistent_files = df[df['consistent'] == True]
        inconsistent_files = df[df['consistent'] == False]
        
        print(f"  Consistent files (std < 5): {len(consistent_files)}")
        if len(consistent_files) > 0:
            print(f"    Average improvement: {consistent_files['avg_improvement_pct'].mean():.1f}%")
        
        print(f"  Inconsistent files (std >= 5): {len(inconsistent_files)}")
        if len(inconsistent_files) > 0:
            print(f"    Average improvement: {inconsistent_files['avg_improvement_pct'].mean():.1f}%")
        
        # Factor 3: Method comparison
        print(f"\nFACTOR 3: METHOD COMPARISON")
        for method in methods:
            method_data = df[df['method'] == method]
            print(f"  {method.upper()}: {len(method_data)} files, avg improvement: {method_data['avg_improvement_pct'].mean():.1f}%")
            
            excellent = len(method_data[method_data['performance_class'] == 'Excellent'])
            good = len(method_data[method_data['performance_class'] == 'Good'])
            neutral = len(method_data[method_data['performance_class'] == 'Neutral'])
            poor = len(method_data[method_data['performance_class'] == 'Poor'])
            
            print(f"    Excellent: {excellent}, Good: {good}, Neutral: {neutral}, Poor: {poor}")
    else:
        print("No per-file analysis found")

def main():
    """Main analysis function"""
    print("Starting comprehensive method effectiveness analysis...")
    
    # Run main analysis
    analyze_method_effectiveness()
    
    # Create factor analysis
    create_factor_analysis()
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS FOR YOUR SUPERVISOR")
    print("=" * 80)
    print(f"""
1. TOLERANCE FACTOR: The methods appear to work best in specific tolerance ranges.
   - Identify the optimal tolerance range for each method
   - Test more densely in those ranges

2. CONSISTENCY ISSUE: Some files show high variance in improvement.
   - This suggests the methods are sensitive to specific data characteristics
   - Need to identify what makes some datasets more suitable

3. METHOD SELECTION: Neither method consistently outperforms across all scenarios.
   - Consider a hybrid approach that selects method based on data characteristics
   - Or develop criteria to predict which method will work better

4. DATASET CHARACTERISTICS: Group datasets by:
   - Data volatility/entropy patterns
   - Sequence uniqueness patterns  
   - Temporal characteristics
   
5. STATISTICAL SIGNIFICANCE: Current results may not be statistically significant.
   - Increase the number of test runs
   - Use proper statistical testing
    """)

if __name__ == "__main__":
    main()