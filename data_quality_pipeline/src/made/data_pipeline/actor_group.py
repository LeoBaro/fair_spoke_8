from pathlib import Path
import sys
import ray
import logging
import time

from made.data_pipeline.actors.unimodal_text_filtering import UnimodalTextFilter
from made.data_pipeline.actors.unimodal_vision_filtering import UnimodalVisionFilter 
from made.data_pipeline.actors.multimodal_alignment_filtering import MultimodalAlignmentFilter
from made.config import Config

class ActorGroup:
    def __init__(
            self, 
            actor_type: str, 
            config_path: str | Path, 
            log_folder: str | Path, 
            output_folder: str | Path
        ):
        self.logger = logging.getLogger("ray")
        self.config = Config(config_path)

        self.logger.info("Initializing actor group. Actor type: %s with %d workers", actor_type, self.config.infrastructure.num_workers)
        self.actors = [getattr(sys.modules[__name__], actor_type)
            .options(
                name=f"{actor_type}_{i}",
                num_gpus=self.config.infrastructure.num_gpus_per_worker,
                max_concurrency=1
            )
            .remote(config_path, log_folder, output_folder) 
            for i in range(self.config.infrastructure.num_workers)]

        self.futures = None
        self.tar_paths = []
        self.uids_paths = []
        self.actor_type = actor_type

    def run(self, tar_files: list[str | Path]):
        tar_splits = [tar_files[i::len(self.actors)] for i in range(len(self.actors))]
        self.logger.info("Executing %s. Number of tar files per worker: %s", str(self), [len(tar_split) for tar_split in tar_splits])
        self.futures = [
            actor.execute.remote(tar_split) for actor, tar_split in zip(self.actors, tar_splits)
        ]

    def get_results(self, timeout: float = 21600.0, check_interval: float = 5.0) -> tuple[list[str], list[str]]:
        """
        Get results from Ray workers with timeout and proper error handling.

        Args:
            timeout: Maximum time to wait for all workers (seconds)
            check_interval: How often to check for completed workers (seconds)

        Returns:
            Tuple of (tar_paths, uids_paths)

        Raises:
            TimeoutError: If workers don't complete within timeout
            RuntimeError: If workers fail or return unexpected results
        """
        if not self.futures:
            return self.tar_paths, self.uids_paths

        start_time = time.time()
        completed_futures = []
        remaining_futures = self.futures.copy()

        self.logger.info("Waiting for %d workers to complete (timeout: %ds)", len(remaining_futures), timeout)

        while remaining_futures and (time.time() - start_time) < timeout:
            # Check which futures are ready (non-blocking)
            ready_futures, remaining_futures = ray.wait(
                remaining_futures,
                num_returns=len(remaining_futures),  # Check all remaining
                timeout=check_interval
            )

            if ready_futures:
                self.logger.info("%d workers completed, %d remaining", len(ready_futures), len(remaining_futures))
                completed_futures.extend(ready_futures)

            # Small sleep to prevent busy waiting
            if remaining_futures:
                time.sleep(0.1)

        # Handle timeout case
        if remaining_futures:
            self.logger.error("Timeout after %ds. %d workers still running", timeout, len(remaining_futures))

            # Cancel remaining futures
            for future in remaining_futures:
                ray.cancel(future)

            raise TimeoutError(f"Workers did not complete within {timeout} seconds. {len(completed_futures)} completed, {len(remaining_futures)} timed out")

        # Get results from completed futures
        try:
            results = ray.get(completed_futures)
            self.logger.info("Successfully retrieved results from %d workers", len(results))
        except Exception as e:
            self.logger.error("Failed to get results from workers: %s", e)
            raise RuntimeError(f"Worker execution failed: {e}")

        # Process results
        for i, result in enumerate(results):
            try:
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise ValueError(f"Worker {i} returned invalid result format: {type(result)}")

                tar_paths_per_actor, uids_paths_per_actor = result

                if not isinstance(tar_paths_per_actor, list) or not isinstance(uids_paths_per_actor, list):
                    raise ValueError(f"Worker {i} returned non-list results")

                self.tar_paths.extend(tar_paths_per_actor)
                self.uids_paths.extend(uids_paths_per_actor)

            except Exception as e:
                self.logger.error("Failed to process result from worker %d: %s", i, e)
                raise RuntimeError(f"Invalid result from worker {i}: {e}")

        self.logger.info("Processing complete. Total paths: %d tar, %d uids", len(self.tar_paths), len(self.uids_paths))
        return self.tar_paths, self.uids_paths

    def kill_actors(self):
        for actor in self.actors:
            ray.kill(actor)

    def __str__(self):
        return f"ActorGroup(actor_type={self.actor_type}, num_workers={self.config.infrastructure.num_workers})"