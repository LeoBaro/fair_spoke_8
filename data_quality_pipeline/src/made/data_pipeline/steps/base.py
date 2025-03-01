from typing import Callable, Any
import time
from made.data_pipeline.metrics.metrics_store import MetricsStore

def apply_filtering_step(
        filter_name: Callable,
        batch_id: int,
        uids: list[str],
        samples: list[Any],
        parameters: dict[str, Any]
    ) -> tuple[list[str], list[Any], list[str], list[Any]]:
    """
    Apply a filtering step to the input data, return the filtered data and save metrics.
    """

    start_time = time.time()
    filter_mask = execute_filter(filter_name, samples, parameters)
    elapsed_time = time.time() - start_time

    ok_uids, ok_samples, uids_filtered, samples_filtered = apply_filter_mask(
        uids, samples, filter_mask
    )
    
    MetricsStore().add_filter_metric(
        filter_name.__name__,
        batch_id,
        len(uids),
        len(ok_uids),
        elapsed_time,
        {"parameters": parameters}
    )

    return ok_uids, ok_samples, uids_filtered, samples_filtered

def execute_filter(
        filter_name: Callable,
        samples: list[Any],
        parameters: dict[str, Any]
    ) -> list[bool]:
    """
    Apply a filter to the samples and return a boolean mask
    """
    return filter_name(samples, **parameters)


def apply_filter_mask(
        uids: list[str],
        samples: list[Any],
        mask: list[bool],
    ) -> tuple[list[str], list[Any], list[str], list[Any]]:
    """
    Apply a filter mask to items and data, returning both kept and filtered items
    
    Args:
        items: List of identifiers (e.g., UIDs)
        data: List of data items (images or captions)
        mask: Boolean mask for filtering
    Returns:
        Tuple of (kept_items, kept_data, filtered_items, filtered_data)
    """
    kept_uids = []
    kept_samples = []
    filtered_uids = []
    filtered_samples = []
    
    for item, sample, m in zip(uids, samples, mask):
        if m:  # Keep this item
            kept_uids.append(item)
            kept_samples.append(sample)
        else:  # Filter out this item
            filtered_uids.append(item)
            filtered_samples.append(sample)

    return kept_uids, kept_samples, filtered_uids, filtered_samples
