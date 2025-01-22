import ray
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from diffusers import StableDiffusionPipeline

from spoke8leonardo.data_pipeline.config import Config
from PIL import Image

@ray.remote
def compute_clip_score(batch):
    multimodal_config = Config().get_multimodal_config()
    processor = CLIPProcessor.from_pretrained(multimodal_config["clip_processor"])
    model = CLIPModel.from_pretrained(multimodal_config["clip_model"])
    scored = []
    for sample in batch:
        image = Image.open(sample["image"])
        inputs = processor(
            text=sample["caption"],
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = model(**inputs)
        score = outputs.logits_per_image.item()
        sample["clip_score"] = score
        scored.append(sample)
    return scored

@ray.remote
def recover_caption_with_llava(batch):
    multimodal_config = Config().get_multimodal_config()
    processor = AutoProcessor.from_pretrained(multimodal_config["caption_generation_processor"])
    model = AutoModelForImageTextToText.from_pretrained(multimodal_config["caption_generator_model"])
    recovered = []
    for sample in batch:
        image = Image.open(sample["image"])
        inputs = processor(images=image, return_tensors="pt")
        generated_caption = model.generate(**inputs)
        sample["caption"] = generated_caption
        recovered.append(sample)
    return recovered


@ray.remote
def recover_image_with_diffusion(batch):
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
    recovered = []
    for sample in batch:
        prompt = sample["caption"]
        generated_image = pipeline(prompt, num_inference_steps=50, batch_size=BATCH_SIZE).images[0]
        new_image_path = f"/tmp/recovered_images/{os.path.basename(sample['image'])}"
        generated_image.save(new_image_path)
        sample["image"] = new_image_path
        recovered.append(sample)
    return recovered

