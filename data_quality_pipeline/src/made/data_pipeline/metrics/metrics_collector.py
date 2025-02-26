import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import ray

@ray.remote
class MetricsCollector:
    def __init__(self):
        self.worker_metrics = {}
        
    def add_worker_metrics(self, worker_id, metrics):
        self.worker_metrics[worker_id] = metrics
        return True
        
    def get_aggregated_metrics(self):
        # Aggregate metrics across all workers
        aggregated = {
            "execution_times": defaultdict(list),
            "filter_metrics": defaultdict(list),
            "workers": list(self.worker_metrics.keys())
        }
        
        for worker_id, metrics in self.worker_metrics.items():
            # Collect execution times
            for func_name, times in metrics["execution_times"].items():
                aggregated["execution_times"][func_name].extend(times)
            
            # Collect filter metrics
            for func_name, filter_data in metrics["filter_metrics"].items():
                aggregated["filter_metrics"][func_name].extend(filter_data)
        
        return aggregated
    
    def save_aggregated_metrics(self, output_dir):
        """Save aggregated metrics to a file"""
        aggregated = self.get_aggregated_metrics()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated data
        agg_path = output_path / f"metrics_aggregated_{timestamp}.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
            
        return str(agg_path)
