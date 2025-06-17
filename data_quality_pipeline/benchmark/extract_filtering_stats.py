import os, json, glob
import argparse
from collections import defaultdict

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True)
    return parser.parse_args()


def main(args):

    metrics_summaries_per_actor = glob.glob(os.path.join(args.folder, "metrics_summary_*.json"))

    filters_order = [
        "_get_filter_captions_by_length_mask",
        "_get_filter_captions_by_language_mask",
        "_get_filter_captions_by_pos_tags_mask",
        "_get_images_by_aspect_ratio_filter_mask",
        "_get_images_by_text_filter_mask",
        "_get_dfn_score_filter_mask"
    ]
    
    metrics_summaries_per_filter = defaultdict(list)

    for mspa in metrics_summaries_per_actor:
        with open(mspa, "r") as mspac:
            content = json.load(mspac)
            for filter_name in filters_order:
                if filter_name in content["filter_metrics"]:
                    metrics_summaries_per_filter[filter_name].append(content["filter_metrics"][filter_name])

    for filter_name, metrics_summaries in metrics_summaries_per_filter.items():
        print(filter_name)
        # print(metrics_summaries)
        total_input = sum([m["total_input"] for m in metrics_summaries])
        total_output = sum([m["total_output"] for m in metrics_summaries])
        total_filtered = sum([m["total_filtered"] for m in metrics_summaries])
        print(f"Total input: {total_input}")
        print(f"Total output: {total_output}")
        print(f"Total filtered: {total_filtered}")
        print(f"Filter rate: {total_filtered / total_input}")

if __name__ == "__main__":
    main(cli())