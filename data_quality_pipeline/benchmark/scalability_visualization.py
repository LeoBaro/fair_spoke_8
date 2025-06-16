import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path

# Set global matplotlib rcParams for scientific plots
plt.rcParams.update({
    # Font settings
    'font.size': 16,                # Base font size
    'axes.titlesize': 18,           # Title size
    'axes.labelsize': 16,           # Axis label size
    'xtick.labelsize': 14,          # X-tick label size
    'ytick.labelsize': 14,          # Y-tick label size
    'legend.fontsize': 14,          # Legend font size


    # Figure & Axes
    'figure.figsize': [6.4, 4.8],   # Default figure size (in inches)
    'figure.dpi': 300,              # High resolution for reports
    'savefig.dpi': 300,             # High resolution for saved images
    'axes.grid': True,              # Enable grid
    'grid.alpha': 0.3,              # Grid transparency
    'axes.spines.top': False,       # Remove top spine
    'axes.spines.right': False,     # Remove right spine


    # Lines
    'lines.linewidth': 2,           # Thicker lines
    'lines.markersize': 6,          # Moderate marker size


    # Legend
    'legend.frameon': False,        # Remove legend frame


    # Text rendering
    'text.usetex': False,           # Set to True if using LaTeX rendering
})


# Optional: use tight_layout by default
plt.rcParams['figure.autolayout'] = True

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--benchmark-file", type=str, required=True, help="CSV file with the benchmark results")
    parser.add_argument("-t", "--title", type=str, required=True)

    return parser.parse_args()

def main(args):

    df = pd.read_csv(args.benchmark_file)
    df["speedup"] = 1 - (df["took"] / max(df["took"]))
    
    summary_df = df.groupby(f"num_workers").agg({"took": ["mean", "std"], "speedup": ["mean", "std"]}).reset_index()
    summary_df.columns = ["Workers", "Mean Time (s)", "Std Dev", "Mean Speedup", "Std Dev Speedup"]
    

    # Set Seaborn style
    sns.set(style="whitegrid")

    fig = plt.errorbar(
        summary_df["Workers"],
        summary_df["Mean Time (s)"],
        yerr=summary_df["Std Dev"],  # Error bars using standard deviation
        fmt="o-",  # Circle markers with solid line
        capsize=3,  # Caps on error bars
        capthick=2,  # Thickness of caps
        elinewidth=1.5,  # Error bar line width
        markersize=4,  # Marker size
        color="b",  # Line color
        label="Mean Execution Time ± Std Dev"
    )
    # Customize the plot
    plt.xlabel(f"Number of Workers")
    plt.ylabel("Mean Execution Time (s)")
    plt.title(args.title)
    plt.xticks(summary_df["Workers"])  # Ensure x-axis has correct worker values
    plt.ylim(0)
    plt.savefig(Path(args.benchmark_file).parent / f"scalability_plot_{args.title.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")

    plt.close()  # Close the plot to free memory
    
    # speedup bar plot
    fig = plt.errorbar(
        summary_df["Workers"],
        summary_df["Mean Speedup"],
        yerr=summary_df["Std Dev Speedup"],
        fmt="o-",
        capsize=3,  # Caps on error bars
        capthick=2,  # Thickness of caps
        elinewidth=1.5,  # Error bar line width
        markersize=4,  # Marker size
        color="b",  # Line color
        label="Mean Speedup ± Std Dev"
    )
    plt.xlabel(f"Number of Workers")
    plt.ylabel("Mean Speedup (%)")
    plt.title(args.title)
    plt.xticks(summary_df["Workers"])  # Ensure x-axis has correct worker values
    plt.ylim(0,1)
    plt.savefig(Path(args.benchmark_file).parent / f"scalability_plot_speedup_{args.title}.png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory



if __name__=='__main__':
    main(cli())