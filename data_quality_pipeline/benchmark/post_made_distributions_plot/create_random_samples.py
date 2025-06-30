import argparse
from pathlib import Path
import pickle
from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--web-dataset-folder", type=str, required=True)
    parser.add_argument("-d", "--dpi", type=int, required=False, default=300)

    return parser.parse_args()

def create_output_folder(input_dir: str):
    output_dir_name = f"{Path(input_dir).name}_post_made_distributions"
    output_dir = Path(Path(input_dir).parent / output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def sample_random_samples(tar_files, num_samples):
    from made.data_pipeline.common.filtering_result import FilteringResult
    filtering_result = FilteringResult(output_folder=Path("."), dump_every_n_samples=5000)

    import random
    dataset = decode_webdataset(
        tar_files,
        get_images=True,
        get_captions=True,
        batch_size=5000
    )
    dataset = iter(dataset)    
    random_uids = []
    random_images = []
    random_captions = []
    while len(random_uids) < num_samples:
        batch = get_next_batch(dataset)
        if batch is None:
            break
        uids, images, captions = batch
        random_indexes = random.choices(range(len(uids)), k=250)
        random_uids.extend([uids[i] for i in random_indexes])
        random_images.extend([images[i] for i in random_indexes])
        random_captions.extend([captions[i] for i in random_indexes])
        print("New batch! random_samples len:", len(random_uids))
        filtering_result.add_samples(uids, captions, images)
    filtering_result.dump_tar()
    
    return random_uids, random_images, random_captions

def main(args):
    # output_dir = create_output_folder(args.web_dataset_folder)
    random_sample_size = 5000
    tar_files = collect_tar_files(args.web_dataset_folder, recursive=True)

    for i in range(5):
        sample_random_samples(tar_files, random_sample_size)



    # load all pickles

if __name__ == "__main__":
    main(cli())