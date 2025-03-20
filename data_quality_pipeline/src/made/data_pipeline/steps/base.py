from typing import Callable, Any
import time
from made.data_pipeline.metrics.metrics_store import MetricsStore

def apply_filtering_step(
        filter_name: Callable,
        batch_id: int,
        uids: list[str],
        samples: list[Any],
        pipeline_type: str,
        parameters: dict[str, Any]
    ) -> tuple[list[str], list[Any], list[str], list[Any]]:
    """
    Apply a filtering step to the input data, record metrics, and return filtered results.

    This function executes the specified filter function on the provided samples using the
    given parameters. It computes a filter mask by calling the filter function, applies the mask
    to separate valid samples from those filtered out, and records filtering metrics (such as
    elapsed time and counts) via the MetricsStore.

    Args:
        `filter_name (Callable)`: The filter function to apply to the samples.
        `batch_id (int)`: Identifier for the current batch; used for logging metrics.
        `uids (list[str])`: List of unique identifiers corresponding to each sample.
        `samples (list[Any])`: List of samples to be filtered.
        `pipeline_type (str)`: Pipeline mode; if set to `"same_input"`, the original samples are 
                             returned as valid results and no samples are considered filtered out.
                             If set to `"classic"`, the mask is applied to the samples
                             and only the valid samples are returned.
        `parameters (dict[str, Any])`: Dictionary of parameters used to configure the filter function.

    Returns:
        tuple:
        - `list[str]`: Unique identifiers for the samples that passed the filter.
        - `list[Any]`: The samples that passed the filter.
        - `list[str]`: Unique identifiers for the samples that were filtered out.
        - `list[Any]`: The samples that were filtered out.

    Side Effects:
        Records filtering metrics using MetricsStore, including the name of the filter function,
        batch ID, total and passed sample counts, processing time, and the filter parameters.
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
        pipeline_type,
        {"parameters": parameters}
    )

    if pipeline_type == "same_input":
        ok_uids = uids
        ok_samples = samples
        uids_filtered = []
        samples_filtered = []

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
