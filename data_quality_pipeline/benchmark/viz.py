import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
if __name__=='__main__':

    results_file = "./benchmark_results.json"

    with open(results_file, "r") as f:
        execution_times_by_worker = json.load(f)

    
    # Prepare data for visualization
    plot_data = []
    for workers, metrics in execution_times_by_worker.items():
        for i, time_value in enumerate(metrics["execution_times"]):
            plot_data.append({
                "Workers": workers,
                "Execution": i+1,
                "Execution Time (s)": time_value
            })
    
    summary_data = []
    for workers, metrics in execution_times_by_worker.items():
        summary_data.append({
            "Workers": workers,
            "Mean Time (s)": metrics["mean_time"],
            "Std Dev": metrics["std_dev"]
        })
    
    df = pd.DataFrame(plot_data)
    summary_df = pd.DataFrame(summary_data)


    # Set Seaborn style
    sns.set(style="whitegrid")


    fig = plt.errorbar(
        summary_df["Workers"],
        summary_df["Mean Time (s)"],
        yerr=summary_df["Std Dev"],  # Error bars using standard deviation
        fmt="o-",  # Circle markers with solid line
        capsize=5,  # Caps on error bars
        capthick=2,  # Thickness of caps
        elinewidth=1.5,  # Error bar line width
        markersize=8,  # Marker size
        color="b",  # Line color
        label="Mean Execution Time ± Std Dev"
    )
    # Customize the plot
    plt.xlabel("Number of Workers", fontsize=12)
    plt.ylabel("Mean Execution Time (s)", fontsize=12)
    plt.title("Mean Execution Time vs. Number of Workers", fontsize=14)
    plt.xticks(summary_df["Workers"])  # Ensure x-axis has correct worker values

    output_path = "benchmark_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()  # Close the plot to free memory
