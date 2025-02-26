import time
from typing import Callable

from made.data_pipeline.metrics.metrics_store import MetricsStore
from made.config import Config

metrics_store = MetricsStore()

def get_time(func):
    def wrapper(*args, **kwargs):
        enable_timing = Config().infrastructure.enable_timing
        if enable_timing:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            metrics_store.add_execution_time(func.__name__, elapsed_time)
        
            print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
            return result
        else:
            return func(*args, **kwargs)  # Run function without timing
    return wrapper


def track_filter(batch_id_arg: int = 0, input_arg: int = 0, output_key: int = 0, extra_info_func: Callable = None):
    """
    Decorator to track filter metrics
    
    Args:
        batch_id_arg: Position of the batch ID argument (or attribute to extract batch ID from)
        input_arg: Position of the input data argument to count
        output_key: Position of the output in the returned data to count
        extra_info_func: Optional function to extract extra info from args/result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            enable_metrics = Config().infrastructure.enable_metrics
            
            if not enable_metrics:
                return func(*args, **kwargs)
            
            # Extract batch ID if available
            batch_id = 0
            if len(args) > batch_id_arg:
                if hasattr(args[batch_id_arg], 'batch_id'):
                    batch_id = args[batch_id_arg].batch_id
                elif isinstance(args[batch_id_arg], (int, str)):
                    batch_id = args[batch_id_arg]
            
            # Count input elements
            input_count = 0
            if len(args) > input_arg:
                if hasattr(args[input_arg], '__len__'):
                    input_count = len(args[input_arg])
                
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            # Count output elements
            output_count = 0
            if isinstance(result, tuple) and len(result) > output_key:
                if hasattr(result[output_key], '__len__'):
                    output_count = len(result[output_key])
            elif hasattr(result, '__len__'):
                output_count = len(result)
            
            # Extract extra info if provided
            extra_info = None
            if extra_info_func:
                extra_info = extra_info_func(args, kwargs, result)
            
            # Store filter metrics
            metrics_store.add_filter_metric(
                func.__name__, 
                batch_id, 
                input_count, 
                output_count, 
                elapsed_time,
                extra_info
            )
            
            print(f"{func.__name__} processed {input_count} items, kept {output_count}, filtered {input_count - output_count} in {elapsed_time:.4f}s")
            
            return result
        return wrapper
    return decorator