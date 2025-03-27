from pathlib import Path
import sys
import ray
import logging

from made.data_pipeline.steps.unimodal_text_filtering import UnimodalTextFilter
from made.data_pipeline.steps.unimodal_vision_filtering import UnimodalVisionFilter 
from made.data_pipeline.steps.multimodal_filtering import MultimodalFilter
#from made.data_pipeline.steps.multimodal_filtering import ray_multimodal_filtering
#from made.data_pipeline.steps.vision_deduplication import ray_vision_deduplication
from typing import Any


class ActorGroup:
    def __init__(self, actor_type: str, num_workers: int, config_path: str | Path):
        print(f"Initializing actor group. Actor type: {actor_type}")
        self.actors = [getattr(sys.modules[__name__], actor_type)
            .options(name=f"{actor_type}_{i}")
            .remote(config_path) for i in range(num_workers)]
        self.futures = None
        self.results = None
        self.actor_type = actor_type
        self.num_workers = num_workers

    def execute_parallel(self, tar_files: list[str | Path], log_folder: str | Path, uids: list[str]):
        #print("Executing actor group. Num. actors: ", len(self.actors))
        tar_splits = [tar_files[i::len(self.actors)] for i in range(len(self.actors))]
        #print("Number of tar splits per actor: ", [len(tar_split) for tar_split in tar_splits])
        self.futures = [
            actor.execute.remote(tar_split, log_folder, uids) for actor, tar_split in zip(self.actors, tar_splits)
        ]
        #print("Executed actor group")

    def get_results(self) -> list[str]:
        #print("Getting results")
        results = ray.get(self.futures)
        #print("Number of results per actor: ", [len(result) for result in results])
        self.results = [uid for sublist in results for uid in sublist]
        return self.results

    def kill_actors(self):
        #print("Killing actors")
        for actor in self.actors:
            ray.kill(actor)

    def __str__(self):
        return f"ActorGroup(actor_type={self.actor_type}, num_workers={self.num_workers})"

class ActorGroupPipeline:
   
    def __init__(self):
        self.pipeline_steps = {}

    def add_pipeline_step(self, merge_strategy: str):
        step_index = len(self.pipeline_steps)
        self.pipeline_steps[step_index] = {
            "actor_groups": [],
            "merge_strategy": merge_strategy
        }

    def add_actor_group(self, step_index: int, actor_type: str, num_workers: int, config_path: str | Path):
        if step_index not in self.pipeline_steps:
            raise ValueError(f"Step index {step_index} not found in pipeline. Add a pipeline step first.")
        self.pipeline_steps[step_index]["actor_groups"].append(ActorGroup(actor_type, num_workers, config_path))

    def pretty_print(self):
        """
        "0": {
            "actor_groups": [
                ActorGroup("UnimodalTextFilter", 2, config_path),
                ActorGroup("UnimodalVisionFilter", 2, config_path)
            ],
            "merge_strategy": "intersection"
        },
        "1": {
            "actor_groups": [
                ActorGroup("MultimodalFilter", 2, config_path)
            ],
            "merge_strategy": "union"
        }   
        """
        for step_index, pipeline_step in self.pipeline_steps.items():
            print(f"Pipeline step {step_index}:")
            for actor_group in pipeline_step["actor_groups"]:
                print(f"  Actor group: {actor_group}")
            print(f"  Merge strategy: {pipeline_step['merge_strategy']}")

    def execute(self, tar_files: list[str | Path], log_folder: str | Path):

        print("Executing the pipeline")

        uids = None

        for step_index, pipeline_step in self.pipeline_steps.items():
            print(f"Executing pipeline_step: {step_index}")

            print("Executing parallel actor groups:")
            for actor_group in self.pipeline_steps[step_index]["actor_groups"]:
                print(f"{actor_group}")

            for actor_group in self.pipeline_steps[step_index]["actor_groups"]:
                actor_group.execute_parallel(tar_files, log_folder, uids)

            uids = [actor_group.get_results() for actor_group in self.pipeline_steps[step_index]["actor_groups"]]

            print("Got the following number of samples from actor groups:")
            for uid, actor_group in zip(uids, self.pipeline_steps[step_index]["actor_groups"]):
                print(f"{actor_group}: {len(uid)}")

            if pipeline_step["merge_strategy"] == "intersection":
                print("Merging results with intersection")
                uids = set(uids[0]).intersection(*uids[1:])
            elif pipeline_step["merge_strategy"] == "union":
                print("Merging results with union")
                uids = set(uids[0]).union(*uids[1:])

            print(f"Number of samples after merging ({pipeline_step['merge_strategy']}):", len(uids))
        
        return uids







# def run_pipeline(
#         tar_files: list[str | Path], 
#         log_folder: str | Path,
#         config_path: str | Path
#     ) -> list[str]:
#     """
#     This pipeline will select the subset of samples that will be used for CLIP training.
#     Each sample in the dataset has a unique identifier, which is present in the metadata 
#     parquets, and in the json files inside the .tar tar.
#     """
#     Config(config_path)
#     logger = logging.getLogger("ray")
#     logger.info("Running pipeline")

#     tar_splits = [tar_files[i::num_workers] for i in range(num_workers)]
#     logger.info("Num. workers: %d. Shard split: %s", num_workers, [len(s) for s in tar_splits])



#     unimodal_text_filtering_actors = [
#         UnimodalTextFilter.remote(config_path) for _ in range(num_workers)
#     ]    

#     unimodal_vision_filtering_actors = [
#         UnimodalVisionFilter.remote(config_path) for _ in range(num_workers)
#     ]    

#     multimodal_filtering_actors = [
#         MultimodalFilter.remote(config_path) for _ in range(num_workers)
#     ]

#     # Launch Ray tasks
#     text_filtering_futures = [
#         actor.execute.remote(tar_split, log_folder) for actor, tar_split in zip(unimodal_text_filtering_actors, tar_splits) if tar_split
#     ]

#     vision_filtering_futures = [
#         actor.execute.remote(tar_split, log_folder) for actor, tar_split in zip(unimodal_vision_filtering_actors, tar_splits) if tar_split
#     ]

#     unimodal_text_filtering_results   = ray.get(text_filtering_futures)
#     unimodal_vision_filtering_results = ray.get(vision_filtering_futures)


#     text_uids = set(uid for sublist in unimodal_text_filtering_results for uid in sublist)
#     vision_uids = set(uid for sublist in unimodal_vision_filtering_results for uid in sublist)
#     unimodal_ok_uids = text_uids & vision_uids

#     # Pass results to multimodal filtering along with the original input
#     multimodal_filtering_futures = [
#         actor.execute.remote(unimodal_ok_uids, tar, log_folder) 
#         for actor, tar in zip(multimodal_filtering_actors, tar_splits) if tar
#     ]

#     multimodal_filtering_results = ray.get(multimodal_filtering_futures)

#     multimodal_ok_uids = set(uid for sublist in multimodal_filtering_results for uid in sublist)

#     # vision_deduplication_futures = [
#     #     ray_vision_deduplication.remote(tar, multimodal_filtering_results)
#     #     for tar in tar_splits if tar
#     # ]

#     # vision_deduplication_results = ray.get(vision_deduplication_futures)
    

#     return multimodal_ok_uids   