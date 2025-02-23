import argparse
from pathlib import Path
import ray
#from made.data_pipeline.steps.unimodal_language import filter_caption_length
#from made.data_pipeline.steps.unimodal_vision import filter_image_aspect_ratio
#from made.data_pipeline.steps.multimodal import (
#    compute_clip_score,
#    recover_caption_with_llava,
#    recover_image_with_diffusion
#)
from made.data_pipeline.config import Config

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=False,
        default=Path(__file__).parent.parent / "config.yaml"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=False,
        default="/home/leobaro/Downloads/datasets/web/datacomp/shards"
    )
    parser.add_argument("--output_path", type=str, required=False, default=None)
    return parser.parse_args()

def main(args):
    # https://docs.ray.io/en/latest/data/api/dataset.html
    dataset = ray.data.read_webdataset(Path(args.dataset_path) / "00000000.tar")

    breakpoint()
    config = Config(args.config_path)
    # context = ray.init(ignore_reinit_error=True)
    #print(context.dashboard_url)

    unimodal_config = config.get_unimodal_config()
    multimodal_config = config.get_multimodal_config()

    infrastructure_config = config.get_infrastructure_config()

    # dataset = dataset.map_batches(
    #     filter_caption_length,
    #     batch_size=unimodal_config["batch_size"],
    #     compute="actors"
    # )
    
    # dataset = dataset.map_batches(
    #     filter_image_aspect_ratio,
    #     batch_size=unimodal_config["batch_size"],
    #     compute="actors"
    # )
    
    # dataset = dataset.map_batches(
    #     compute_clip_score,
    #     batch_size=unimodal_config["batch_size"],
    #     compute="actors"
    # )
    
    # # Recovery branch
    # discarded_samples = dataset.filter(
    #     lambda sample: "clip_score" not in sample or sample["clip_score"] < multimodal_config["clip_score_threshold"])
    
    # # Recover captions and images
    # discarded_captions = discarded_samples.filter(
    #     lambda sample: len(sample["caption"].split()) < unimodal_config["caption_min_length"])
    
    # discarded_images = discarded_samples.filter(
    #     lambda sample: "aspect_ratio" in sample and sample["aspect_ratio"] > unimodal_config["aspect_ratio_threshold"])
    
    # recovered_captions = discarded_captions.map_batches(
    #     recover_caption_with_llava,
    #     batch_size=multimodal_config["batch_size"],
    #     compute="actors"
    # )
    # recovered_images = discarded_images.map_batches(
    #     recover_image_with_diffusion,
    #     batch_size=multimodal_config["batch_size"],
    #     compute="actors"
    # )
    
    # # Combine recovered samples with original dataset
    # final_dataset = dataset.union(recovered_captions).union(recovered_images)
    
    # # Save the final dataset
    # final_dataset.write_json("/path/to/output/final_dataset")



if __name__ == "__main__":
    main(cli())
