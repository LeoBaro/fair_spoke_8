import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path

from made.data_pipeline.utils import set_plotting_configuration
set_plotting_configuration()

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--benchmark-files", type=str, nargs='+', required=True, 
                       help="CSV files with the benchmark results (can specify multiple files)")
    parser.add_argument("-t", "--title", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                       help="Output directory for plots (defaults to directory of first benchmark file)")
    parser.add_argument("--figsize", type=str, default="12,8",
                       help="Figure size as 'width,height' (default: '12,8')")
    return parser.parse_args()

def process_benchmark_file(file_path):
    """Process a single benchmark file and return summary statistics."""
    df = pd.read_csv(file_path)
    df["speedup"] = 1 - (df["took"] / max(df["took"]))
    
    summary_df = df.groupby("num_workers").agg({"took": ["mean", "std"], "speedup": ["mean", "std"]}).reset_index()
    summary_df.columns = ["Workers", "Mean Time (s)", "Std Dev", "Mean Speedup", "Std Dev Speedup"]
    # remove the first row
    summary_df = summary_df.iloc[1:]
    print(summary_df)
    return summary_df, Path(file_path).stem

def create_execution_time_plot(summary_dfs, file_names, title, output_path, figsize):
    """Create execution time plot with multiple subplots."""
    n_files = len(summary_dfs)
    
    # Calculate subplot layout
    cols = min(2, n_files)  # Max 3 columns
    rows = (n_files + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle(f"{title} - Execution Time", fontsize=16, y=0.98)
    
    for idx, (summary_df, file_name) in enumerate(zip(summary_dfs, file_names)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ax.errorbar(
            summary_df["Workers"],
            summary_df["Mean Time (s)"],
            yerr=summary_df["Std Dev"],
            fmt="o-",
            capsize=3,
            capthick=2,
            elinewidth=1.5,
            markersize=4,
            color="b",
            label="Mean Execution Time ± Std Dev"
        )
        
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Mean Execution Time (s)")
        ax.set_title(f"{file_name.replace('_', ' ').title()}")
        ax.set_xticks(summary_df["Workers"])
        ax.set_ylim(0)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_files, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close()

def create_speedup_plot(summary_dfs, file_names, title, output_path, figsize):
    """Create speedup plot with multiple subplots."""
    n_files = len(summary_dfs)
    
    # Calculate subplot layout
    cols = min(2, n_files)  # Max 3 columns
    rows = (n_files + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle(f"{title} - Speedup", fontsize=16, y=0.98)
    
    for idx, (summary_df, file_name) in enumerate(zip(summary_dfs, file_names)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ax.errorbar(
            summary_df["Workers"],
            summary_df["Mean Speedup"],
            yerr=summary_df["Std Dev Speedup"],
            fmt="o-",
            capsize=3,
            capthick=2,
            elinewidth=1.5,
            markersize=4,
            color="g",
            label="Mean Speedup ± Std Dev"
        )
        
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Mean Speedup (%)")
        ax.set_title(f"{file_name.replace('_', ' ').title()}")
        ax.set_xticks(summary_df["Workers"])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_files, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    
    plt.close()

def main(args):
    # Parse figure size
    figsize = tuple(map(int, args.figsize.split(',')))
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.benchmark_files[0]).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all benchmark files
    summary_dfs = []
    file_names = []
    
    for benchmark_file in args.benchmark_files:
        summary_df, file_name = process_benchmark_file(benchmark_file)
        summary_dfs.append(summary_df)
        file_names.append(file_name)
    
    # Create execution time plot
    execution_time_output = output_dir / f"scalability_plot_execution_time_{args.title.replace(' ', '_')}.png"
    create_execution_time_plot(summary_dfs, file_names, args.title, execution_time_output, figsize)
    
    # Create speedup plot
    speedup_output = output_dir / f"scalability_plot_speedup_{args.title.replace(' ', '_')}.png"
    create_speedup_plot(summary_dfs, file_names, args.title, speedup_output, figsize)
    
    print(f"Plots saved to:")
    print(f"  Execution time: {execution_time_output}")
    print(f"  Speedup: {speedup_output}")

if __name__=='__main__':
    main(cli())