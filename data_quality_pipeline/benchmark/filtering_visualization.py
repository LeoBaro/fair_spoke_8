import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from made.data_pipeline.utils import set_plotting_configuration
set_plotting_configuration()

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--metric-summary", type=str, required=True, help="JSON file with the metric summary")
    parser.add_argument("-t", "--title", type=str, required=True)
    return parser.parse_args()

def main(args):

    df = pd.read_csv(args.benchmark_file)
    df["speedup"] = 1 - (df["took"] / max(df["took"]))

    summary_df = df.groupby("num_workers").agg({"took": ["mean", "std"], "speedup": ["mean", "std"]}).reset_index()
    summary_df.columns = ["Workers", "Mean Time (s)", "Std Dev", "Mean Speedup", "Std Dev Speedup"]
    
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
    plt.xlabel("Number of Workers", fontsize=12)
    plt.ylabel("Mean Execution Time (s)", fontsize=12)
    plt.title(args.title, fontsize=14)
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
    plt.xlabel("Number of Workers", fontsize=12)
    plt.ylabel("Mean Speedup (%)", fontsize=12)
    plt.title(args.title, fontsize=14)
    plt.xticks(summary_df["Workers"])  # Ensure x-axis has correct worker values
    plt.ylim(0,1)
    plt.savefig(Path(args.benchmark_file).parent / f"scalability_plot_speedup_{args.title}.png", dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory



if __name__=='__main__':
    main(cli())