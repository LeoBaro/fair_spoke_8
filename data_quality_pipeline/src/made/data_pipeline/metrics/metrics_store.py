import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict

import ray

from made.data_pipeline.common import Singleton

class MetricsStore(metaclass=Singleton):
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.filter_metrics = defaultdict(list)
        self.worker_id = ray.get_runtime_context().worker_id if ray.is_initialized() else "local"
        
    def add_execution_time(self, func_name: str, elapsed_time: float):
        self.execution_times[func_name].append(elapsed_time)
        
    def add_filter_metric(self, func_name: str, batch_id: int, 
                          input_count: int, output_count: int, 
                          elapsed_time: float, extra_info: Optional[Dict] = None):
        metric = {
            "worker_id": self.worker_id,
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "input_count": input_count,
            "output_count": output_count,
            "filtered_count": input_count - output_count,
            "filter_rate": (input_count - output_count) / input_count if input_count > 0 else 0,
            "elapsed_time": elapsed_time
        }
        
        if extra_info:
            metric.update(extra_info)
            
        self.filter_metrics[func_name].append(metric)
    
    def get_summary(self) -> Dict:
        summary = {
            "worker_id": self.worker_id,
            "execution_times": {},
            "filter_metrics": {}
        }
        
        # Summarize execution times
        for func_name, times in self.execution_times.items():
            mean_time = statistics.mean(times)
            stddev_time = statistics.stdev(times) if len(times) > 1 else 0.0
            summary["execution_times"][func_name] = {
                "mean": mean_time,
                "stddev": stddev_time,
                "min": min(times),
                "max": max(times),
                "calls": len(times)
            }
        
        # Summarize filter metrics
        for func_name, metrics in self.filter_metrics.items():
            input_counts = [m["input_count"] for m in metrics]
            output_counts = [m["output_count"] for m in metrics]
            filter_rates = [m["filter_rate"] for m in metrics]
            
            summary["filter_metrics"][func_name] = {
                "total_input": sum(input_counts),
                "total_output": sum(output_counts),
                "total_filtered": sum(input_counts) - sum(output_counts),
                "avg_filter_rate": statistics.mean(filter_rates) if filter_rates else 0,
                "batches_processed": len(metrics)
            }
            
        return summary
    
    def save_to_file(self, output_dir: str):
        """Save metrics to JSON files in the specified directory"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create a timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        worker_suffix = self.worker_id.replace(":", "_")
        
        # Save summary
        summary_path = output_path / f"metrics_summary_{worker_suffix}_{timestamp}.json"
        with open(summary_path, 'w', encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
            
        # Save detailed metrics
        details_path = output_path / f"metrics_details_{worker_suffix}_{timestamp}.json"
        with open(details_path, 'w', encoding="utf-8") as f:
            json.dump({
                "worker_id": self.worker_id,
                "execution_times": dict(self.execution_times),
                "filter_metrics": dict(self.filter_metrics)
            }, f, indent=2)
        
        return summary_path, details_path
