from typing import Any

import ray

from made.config import Config
from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.data_pipeline.metrics.metrics_decorators import get_time


@get_time
def apply_filter_mask(
        uids: list[str],
        data: list[Any],
        mask: list[bool],
        filter_name: str = "unnamed_filter", 
        batch_id: int = 0
    ) -> tuple[list[str], list[Any], list[str], list[Any], int]:
    """
    Apply a filter mask to items and data, returning both kept and filtered items
    
    Args:
        items: List of identifiers (e.g., UIDs)
        data: List of data items (images or captions)
        mask: Boolean mask for filtering
        filter_name: Name of the filter for logging
        batch_id: Batch identifier for logging
        
    Returns:
        Tuple of (kept_items, kept_data, filtered_items, filtered_data)
    """
    kept_items = []
    kept_data = []
    filtered_items = []
    filtered_data = []
    
    for item, d, m in zip(uids, data, mask):
        if m:  # Keep this item
            kept_items.append(item)
            kept_data.append(d)
        else:  # Filter out this item
            filtered_items.append(item)
            filtered_data.append(d)

    # Log metrics for this filtering operation
    input_count = len(uids)
    output_count = len(kept_items)
    elapsed_time = 0  # This function doesn't time itself
    
    MetricsStore().add_filter_metric(
        filter_name,
        batch_id,
        input_count,
        output_count,
        elapsed_time,
        {"filter_reason": filter_name}
    )

    return kept_items, kept_data, filtered_items, filtered_data

def print_execution_stats():
    """Print execution statistics and filter metrics summaries"""
    summary = MetricsStore().get_summary()
    
    print("\n===== Execution Time Statistics =====")
    for func_name, stats in summary["execution_times"].items():
        print(f"{func_name}:")
        print(f"  Mean = {stats['mean']:.4f}s, StdDev = {stats['stddev']:.4f}s")
        print(f"  Min = {stats['min']:.4f}s, Max = {stats['max']:.4f}s, Calls = {stats['calls']}")
    
    print("\n===== Filter Metrics Summary =====")
    for func_name, stats in summary["filter_metrics"].items():
        print(f"{func_name}:")
        print(f"  Total input: {stats['total_input']}, Total output: {stats['total_output']}")
        print(f"  Total filtered: {stats['total_filtered']} ({stats['avg_filter_rate']*100:.2f}%)")
        print(f"  Batches processed: {stats['batches_processed']}")
    
    # Save metrics to file
    summary_path, details_path = MetricsStore().save_to_file(Config().infrastructure.log_folder)
    print(f"\nMetrics saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Details: {details_path}")