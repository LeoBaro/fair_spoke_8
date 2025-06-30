import os
import glob
import argparse
from pathlib import Path

from made.data_pipeline.utils import collect_tar_files
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from utils import create_samples_visualization

def get_good_uids(results_folder):
    good_uids_files = glob.glob(os.path.join(results_folder, "*.txt"))
    good_uids = []
    for file in good_uids_files:
        with open(file, "r", encoding="utf-8") as f:
            good_uids.extend(f.read().splitlines())
    return good_uids

def create_txt_file_list(uids: list[str], captions: list[str], output_file: str):
    print("UIDS length: ", len(uids))
    print("Captions length: ", len(captions))
    assert len(uids) == len(captions), "UIDS and Captions must have the same length"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for uid, caption in zip(uids, captions):
            f.write(f"{uid} {caption}\n")
    print(f"Visualization saved to {output_file}")

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--web-dataset-folder", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-i", "--number-of-samples-per-image", type=int, required=False, default=20)
    parser.add_argument("-d", "--dpi", type=int, required=False, default=300)

    return parser.parse_args()

def create_output_folder(input_dir: str):
    output_dir_name = f"{Path(input_dir).name}_good_uids_visualizations"
    output_dir = Path(Path(input_dir).parent / output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def dump_captions_to_txt(uids: list[str], captions: list[str], output_path: str):
    with open(output_path, "a", encoding="utf-8") as f:
        for uid, caption in zip(uids, captions):
            f.write(f"{uid} {caption}\n")

def main(args):
    output_dir = create_output_folder(args.web_dataset_folder)

    dataset = decode_webdataset(
        collect_tar_files(args.web_dataset_folder, recursive=False),
        get_images=True,
        get_captions=True,
        batch_size=args.number_of_samples_per_image
    )
    dataset = iter(dataset)

    count = 0
    while count < args.num_samples:
        try:
            batch_uids, batch_images, batch_captions = get_next_batch(dataset)
            output_path = output_dir / f"good_uids_{count}_{count + args.number_of_samples_per_image}"
            create_samples_visualization(batch_uids, batch_images, batch_captions, f"", output_path, args.dpi)
            dump_captions_to_txt(batch_uids, batch_captions, output_path.parent / "captions.txt")
            count += args.number_of_samples_per_image
        except Exception as e:
            print("Exception!", e)
            break



if __name__ == "__main__":
    main(cli())