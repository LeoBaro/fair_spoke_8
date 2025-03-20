import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict

from made.data_pipeline.common import Singleton
import ray

class MetricsStore(metaclass=Singleton):

    def __init__(self):
        self.filter_metrics = defaultdict(list)
        self.worker_id = ray.get_runtime_context().get_worker_id() if ray.is_initialized() else "local"

    def add_filter_metric(
            self, 
            func_name: str,
            batch_id: int,
            input_count: int,
            output_count: int,
            elapsed_time: float,
            pipeline_type: str,
            extra_info: Optional[Dict] = None
        ):
        
        metric = {
            "worker_id": self.worker_id,
            "pipeline_type": pipeline_type,
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "input_count": input_count,
            "output_count": output_count,
            "filtered_count": input_count - output_count,
            "filter_rate": round((input_count - output_count) / input_count if input_count > 0 else 0, 5),
            "elapsed_time": round(elapsed_time, 5)
        }
        
        if extra_info:
            metric.update({p: str(v) for p, v in extra_info.items()})
            
        self.filter_metrics[func_name].append(metric)
    
    def get_summary(self) -> Dict:
        summary = {
            "worker_id": self.worker_id,
            "filter_metrics": {}
        }
        
        for func_name, metrics in self.filter_metrics.items():
            input_counts = [m["input_count"] for m in metrics]
            output_counts = [m["output_count"] for m in metrics]
            filter_rates = [m["filter_rate"] for m in metrics]
            times = [m["elapsed_time"] for m in metrics]
            
            
            summary["filter_metrics"][func_name] = {
                "total_input": sum(input_counts),
                "total_output": sum(output_counts),
                "batches_processed": len(metrics),
                "total_filtered": sum(input_counts) - sum(output_counts),
                "avg_filter_rate": round(statistics.mean(filter_rates) if filter_rates else 0, 5),
                "stddev_filter_rate": round(statistics.stdev(filter_rates) if len(filter_rates) > 1 else 0.0, 5),
                "min_filter_rate": round(min(filter_rates), 5),
                "max_filter_rate": round(max(filter_rates), 5),
                "avg_elapsed_time_seconds": round(statistics.mean(times), 5),
                "stddev_elapsed_time_seconds": round(statistics.stdev(times) if len(times) > 1 else 0.0, 5),
                "min_elapsed_time_seconds": round(min(times), 5),
                "max_elapsed_time_seconds": round(max(times), 5),
                "calls": len(times),
                "parameters": metrics[0]['parameters'],
                "pipeline_type": metrics[0]['pipeline_type']
            }
            
        return summary
    
    def save_to_file(self, output_dir: str):
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
                "filter_metrics": dict(self.filter_metrics)
            }, f, indent=2)
        
        return summary_path, details_path
