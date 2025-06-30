from pathlib import Path
from made.paths import MADE_PATH
from made.config import Config
from made.models.meru.nn import model_init
from made.data_pipeline.data.datacomp_handler import decode_webdataset, get_next_batch
from made.data_pipeline.filtering_functions.multimodal_filters import _compute_specificity
from made.data_pipeline.utils import collect_tar_files

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


def compute_scores(tar_files, output_dir, output_file_name, max_samples):
    count = 0
    output_file = output_dir / output_file_name

    with open(output_file, "w") as f:
        dataset = decode_webdataset(
            tar_files,
            get_images=True,
            get_captions=True,
            batch_size=config.infrastructure.batch_size
        )
        dataset_iter = iter(dataset)
        
        while True:    
            print(f"Count: {count}/{max_samples}")
            batch = get_next_batch(dataset_iter) 
            if batch is None:
                break
            if count >= max_samples:
                break
            count += len(batch[0])
            images = batch[1]
            captions = batch[2]

            image_specificity_scores, text_specificity_scores = _compute_specificity(
                captions,
                images,
                model,
                trs,
                tokenizer,
                img_ref,
                txt_ref,
                model.curvature.exp()
            )
            for iss, tss in zip(image_specificity_scores, text_specificity_scores):
                f.write(f"{iss} {tss}\n")

if __name__ == "__main__":
    here = Path(__file__).parent
    tar_files = [
        # here / "20250620_163113_00000000.tar",
        # here / "20250620_163413_00000000.tar",
        # here / "20250620_163657_00000000.tar",
        # here / "20250620_164004_00000000.tar",
        here / "20250620_164249_00000000.tar"
    ]
    max_samples = 50000
    output_dir = here / "specificity_scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(Path("config.yaml"))
    ref = torch.load(
        str(MADE_PATH / config.specificity.reference_path)
    )
    img_ref = ref["img"].to("cuda")
    txt_ref = ref["txt"].to("cuda")
    model, trs = model_init(pretrained=str(MADE_PATH / config.specificity.model_path))
    model = model.to("cuda").eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    for tar_file in tar_files:
        compute_scores([str(tar_file)], output_dir, tar_file.name.replace(".tar", ".txt"), max_samples)

    # compute_scores(collect_tar_files("/home/leobaro/Downloads/datasets/web/datacomp/full_datacomp_multimodal_specificity_filter_output"), output_dir, "full_datacomp_multimodal_specificity_filter_output.txt", max_samples)

            