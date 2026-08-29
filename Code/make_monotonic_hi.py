import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_all_parquet_files(folder_path):
    """Load all parquet files from the specified folder"""
    folder = Path(folder_path)
    parquet_files = sorted(folder.glob("*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load first file to understand structure
    print("\n" + "="*80)
    print("LOADING SAMPLE FILE TO UNDERSTAND STRUCTURE")
    print("="*80)
    
    sample_df = pd.read_parquet(parquet_files[0])
    print(f"\nFirst file: {parquet_files[0].name}")
    print(f"Shape: {sample_df.shape}")
    print(f"\nColumns: {list(sample_df.columns)}")
    print(f"\nFirst few rows:")
    print(sample_df.head())
    
    # Load all files
    print("\n" + "="*80)
    print("LOADING ALL FILES (this may take a moment...)")
    print("="*80)
    
    all_data = []
    for i, file in enumerate(parquet_files):
        df = pd.read_parquet(file)
        df['file_id'] = i  # Track which file each row came from
        df['file_name'] = file.stem
        all_data.append(df)
        
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{len(parquet_files)} files...")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined shape: {combined_df.shape}")
    
    return combined_df, sample_df

# ==============================================================================
# 2. BASIC DATA OVERVIEW
# ==============================================================================

def basic_overview(df):
    """Provide basic overview of the dataset"""
    print("\n" + "="*80)
    print("BASIC DATA OVERVIEW")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-"*80)
    print("Column Data Types:")
    print("-"*80)
    print(df.dtypes)
    
    print("\n" + "-"*80)
    print("Missing Values:")
    print("-"*80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    print("\n" + "-"*80)
    print("Statistical Summary:")
    print("-"*80)
    print(df.describe())
    
    return missing_df

# ==============================================================================
# 3. TEMPORAL ANALYSIS
# ==============================================================================

def temporal_analysis(df):
    """Analyze temporal patterns in the data"""
    print("\n" + "="*80)
    print("TEMPORAL ANALYSIS")
    print("="*80)
    
    # Check for time-related columns
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if time_cols:
        print(f"\nTime-related columns found: {time_cols}")
        for col in time_cols:
            print(f"\n{col}:")
            print(f"  Type: {df[col].dtype}")
            print(f"  Range: {df[col].min()} to {df[col].max()}")
            print(f"  Unique values: {df[col].nunique()}")
    else:
        print("\nNo obvious time columns found. Checking index...")
        if df.index.name:
            print(f"Index name: {df.index.name}")
    
    # Analyze sequences per file
    print("\n" + "-"*80)
    print("Sequence Length Analysis (rows per file/pipeline run):")
    print("-"*80)
    
    if 'file_id' in df.columns:
        file_lengths = df.groupby('file_id').size()
        print(f"\nNumber of files: {len(file_lengths)}")
        print(f"Sequence length statistics:")
        print(f"  Mean: {file_lengths.mean():.2f}")
        print(f"  Median: {file_lengths.median():.2f}")
        print(f"  Min: {file_lengths.min()}")
        print(f"  Max: {file_lengths.max()}")
        print(f"  Std: {file_lengths.std():.2f}")
        
        # Plot sequence lengths
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].hist(file_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Sequence Length (rows per file)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Sequence Lengths')
        axes[0].axvline(file_lengths.mean(), color='red', linestyle='--', label=f'Mean: {file_lengths.mean():.1f}')
        axes[0].legend()
        
        axes[1].boxplot(file_lengths)
        axes[1].set_ylabel('Sequence Length')
        axes[1].set_title('Sequence Length Box Plot')
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 4. FEATURE ANALYSIS
# ==============================================================================

def feature_analysis(df):
    """Analyze individual features"""
    print("\n" + "="*80)
    print("FEATURE ANALYSIS")
    print("="*80)
    
    # Identify numeric columns (excluding file_id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'file_id' in numeric_cols:
        numeric_cols.remove('file_id')
    
    print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
    
    # Distribution plots for numeric features
    n_cols = min(len(numeric_cols), 6)  # Limit to 6 features for visibility
    
    if n_cols > 0:
        fig, axes = plt.subplots((n_cols + 2) // 3, 3, figsize=(18, 5 * ((n_cols + 2) // 3)))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols[:n_cols]):
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
            axes[idx].legend()
        
        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Box plots to identify outliers
        fig, axes = plt.subplots((n_cols + 2) // 3, 3, figsize=(18, 5 * ((n_cols + 2) // 3)))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols[:n_cols]):
            axes[idx].boxplot(df[col].dropna())
            axes[idx].set_ylabel(col)
            axes[idx].set_title(f'Box Plot: {col}')
        
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 5. CORRELATION ANALYSIS
# ==============================================================================

def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'file_id' in numeric_cols:
        numeric_cols.remove('file_id')
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        
        # Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Find high correlations
        print("\n" + "-"*80)
        print("High Correlations (|r| > 0.7):")
        print("-"*80)
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', ascending=False)
            print(high_corr_df)
        else:
            print("No high correlations found.")

# ==============================================================================
# 6. TIME SERIES PATTERNS
# ==============================================================================

def time_series_patterns(df, sample_files=5):
    """Analyze time series patterns for sample pipeline runs"""
    print("\n" + "="*80)
    print("TIME SERIES PATTERNS (Sample Pipeline Runs)")
    print("="*80)
    
    if 'file_id' not in df.columns:
        print("file_id column not found. Skipping time series analysis.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'file_id' in numeric_cols:
        numeric_cols.remove('file_id')
    
    # Sample random files
    unique_files = df['file_id'].unique()
    sample_file_ids = np.random.choice(unique_files, min(sample_files, len(unique_files)), replace=False)
    
    print(f"\nPlotting {len(sample_file_ids)} sample pipeline runs...")
    
    for col in numeric_cols[:3]:  # Limit to first 3 features
        plt.figure(figsize=(15, 6))
        
        for file_id in sample_file_ids:
            file_data = df[df['file_id'] == file_id][col].reset_index(drop=True)
            plt.plot(file_data, alpha=0.7, label=f'File {file_id}')
        
        plt.xlabel('Time Step')
        plt.ylabel(col)
        plt.title(f'{col} Over Time - Sample Pipeline Runs to Rupture')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ==============================================================================
# 7. RUPTURE CHARACTERISTICS
# ==============================================================================

def rupture_characteristics(df):
    """Analyze characteristics at or near rupture"""
    print("\n" + "="*80)
    print("RUPTURE CHARACTERISTICS")
    print("="*80)
    
    if 'file_id' not in df.columns:
        print("file_id column not found. Skipping rupture analysis.")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'file_id' in numeric_cols:
        numeric_cols.remove('file_id')
    
    # Get last row of each file (rupture point)
    rupture_points = df.groupby('file_id').tail(1)[numeric_cols]
    
    print("\nStatistics at Rupture Point:")
    print(rupture_points.describe())
    
    # Compare early vs late stage values
    print("\n" + "-"*80)
    print("Comparison: Early Stage (first 10%) vs Late Stage (last 10%):")
    print("-"*80)
    
    early_stage = df.groupby('file_id').head(int(df.groupby('file_id').size().mean() * 0.1))
    late_stage = df.groupby('file_id').tail(int(df.groupby('file_id').size().mean() * 0.1))
    
    comparison_data = []
    for col in numeric_cols:
        comparison_data.append({
            'Feature': col,
            'Early_Mean': early_stage[col].mean(),
            'Late_Mean': late_stage[col].mean(),
            'Change_%': ((late_stage[col].mean() - early_stage[col].mean()) / early_stage[col].mean() * 100)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    
    # Visualize rupture values distribution
    n_cols = min(len(numeric_cols), 4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:n_cols]):
        axes[idx].hist(rupture_points[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='red')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'Distribution of {col} at Rupture')
        axes[idx].axvline(rupture_points[col].mean(), color='darkred', linestyle='--', 
                         label=f'Mean: {rupture_points[col].mean():.2f}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    # Define the data folder path
    data_folder = r"D:\Pipeline RUL Data\out_parquet_20k"
    
    print("="*80)
    print("PIPELINE RUL DATA - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    print(f"\nData Folder: {data_folder}")
    
    # Load data
    df, sample_df = load_all_parquet_files(data_folder)
    
    # Run analyses
    missing_info = basic_overview(df)
    temporal_analysis(df)
    feature_analysis(df)
    correlation_analysis(df)
    time_series_patterns(df, sample_files=5)
    rupture_characteristics(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    print("\nKey Insights Summary:")
    print(f"  • Total samples: {df.shape[0]:,}")
    print(f"  • Number of pipeline runs: {df['file_id'].nunique() if 'file_id' in df.columns else 'N/A'}")
    print(f"  • Number of features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  • Data quality: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}% complete")
    
    return df

# Run the analysis
if __name__ == "__main__":
    df = main()