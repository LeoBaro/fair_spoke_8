import time
import statistics
import atexit
from made.data_pipeline.config import Config

execution_times = {}

def get_time(func):
    def wrapper(*args, **kwargs):
        enable_timing = Config().infrastructure.enable_timing
        if enable_timing:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Store execution time in the global dictionary
            if func.__name__ not in execution_times:
                execution_times[func.__name__] = []
            execution_times[func.__name__].append(elapsed_time)

            print(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
            return result
        else:
            return func(*args, **kwargs)  # Run function without timing
    return wrapper

def print_execution_stats():
    if execution_times:
        print("\nExecution Time Statistics:")
        for func_name, times in execution_times.items():
            mean_time = statistics.mean(times)
            stddev_time = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"{func_name}: Mean = {mean_time:.4f}s, StdDev = {stddev_time:.4f}s")

atexit.register(print_execution_stats)
