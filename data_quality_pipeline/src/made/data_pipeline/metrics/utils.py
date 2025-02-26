import ray

from made.data_pipeline.metrics.metrics_store import MetricsStore

def send_metrics_central_collector():
    if ray.is_initialized():
        metrics_collector = ray.get_actor("metrics_collector")
        ray.get(metrics_collector.add_worker_metrics.remote(
            ray.get_runtime_context().worker_id,
            {
                "execution_times": dict(MetricsStore().execution_times),
                "filter_metrics": dict(MetricsStore().filter_metrics)
            }
        ))