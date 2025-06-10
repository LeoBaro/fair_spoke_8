from pathlib import Path
import sys
import ray
import logging

from made.data_pipeline.steps.unimodal_text_filtering import UnimodalTextFilter
from made.data_pipeline.steps.unimodal_vision_filtering import UnimodalVisionFilter 
from made.data_pipeline.steps.multimodal_filtering import MultimodalFilter

class ActorGroup:
    def __init__(
            self, 
            actor_type: str, 
            num_workers: int,
            config_path: str | Path, 
            log_folder: str | Path, 
            output_folder: str | Path
        ):
        self.logger = logging.getLogger("ray")
        self.logger.info("Initializing actor group. Actor type: %s with %d workers", actor_type, num_workers)
        self.actors = [getattr(sys.modules[__name__], actor_type)
            .options(name=f"{actor_type}_{i}")
            .remote(config_path, log_folder, output_folder) for i in range(num_workers)]
        self.futures = None
        self.results = None
        self.actor_type = actor_type
        self.num_workers = num_workers

    def run(self, tar_files: list[str | Path]):
        tar_splits = [tar_files[i::len(self.actors)] for i in range(len(self.actors))]
        self.logger.info("Executing %s. Number of tar files per worker: %s", str(self), [len(tar_split) for tar_split in tar_splits])
        self.futures = [
            actor.execute.remote(tar_split) for actor, tar_split in zip(self.actors, tar_splits)
        ]

    def get_results(self) -> list[str]:
        results = ray.get(self.futures)
        self.results = [element for sublist in results for element in sublist]
        return self.results

    def kill_actors(self):
        for actor in self.actors:
            ray.kill(actor)

    def __str__(self):
        return f"ActorGroup(actor_type={self.actor_type}, num_workers={self.num_workers})"