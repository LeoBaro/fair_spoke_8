from made.data_pipeline.filtering_functions.text_filters import filter_captions_by_length, filter_captions_by_language, filter_captions_by_pos_tags
from made.data_pipeline.filtering_functions.vision_filters import filter_images_by_aspect_ratio, filter_images_by_text_detection
from made.data_pipeline.filtering_functions.multimodal_filters import filter_by_clip_similarity
from made.paths import MADE_PATH

import spacy
import easyocr
import fasttext
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def test_filter_captions_by_length():
    captions = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown jumps"
    ]
    mask = filter_captions_by_length(captions, 5, 5)
    assert mask == [True, False]

def test_filter_captions_by_language(config):
    language_detection_model = fasttext.load_model(
        str(MADE_PATH / config.unimodal_text.lang_detection_model_path)
        )
    spacy.require_gpu()
    captions = [
        "The quick brown fox jumps over the lazy dog",
        "敏捷的棕色狐狸跳过了懒狗"
    ]
    mask = filter_captions_by_language(captions, language_detection_model, "en", 0.5)
    assert mask == [True, False]

def test_filter_captions_by_pos_tags(config):
    tagging_model = spacy.load(
        str(config.unimodal_text.tagging_model_name)
    )
    with open(
        str(MADE_PATH / config.unimodal_text.good_captions_pos_distribution_path),
        'r',
        encoding='utf-8'
    ) as file:
        common_pos_patterns = [line.strip() for line in file.readlines()]
    
    captions = [
        "The quick brown fox jumps over the lazy dog",
        "Poor bens"
    ]
    mask = filter_captions_by_pos_tags(captions, tagging_model, common_pos_patterns)
    assert mask == [True, False]

def test_filter_images_by_aspect_ratio(test_images_path_aspect_ratio):
    images = [
        Image.open(str(test_images_path_aspect_ratio / "000000000099.jpg")),
        Image.open(str(test_images_path_aspect_ratio / "000000000338.jpg")),
        Image.open(str(test_images_path_aspect_ratio / "000000000711.jpg"))
    ]
    mask = filter_images_by_aspect_ratio(images, image_min_aspect_ratio=0.6, image_max_aspect_ratio=1.2, image_min_dimension=51)
    assert mask == [False, True, False]

def test_filter_images_by_text_detection(test_images_path_text_detection, config):
    text_detection_model = easyocr.Reader(['en'], gpu=True, user_network_directory=config.unimodal_vision.text_detection_model_path)

    images = [
        Image.open(str(test_images_path_text_detection / "000000000703.jpg")),
        Image.open(str(test_images_path_text_detection / "000000001522.jpg"))
    ]
    mask = filter_images_by_text_detection(images, text_detection_model, text_thresh=0.6, mag_ratio=0.5)
    assert mask == [False, True]

def test_filter_by_clip_similarity(test_images_path_similarity, config):
    dfn_model = CLIPModel.from_pretrained(config.multimodal.dfn_model).eval().to("cuda")
    clip_processor = CLIPProcessor.from_pretrained(config.multimodal.dfn_model)
    captions = [
        "Bell H-13 Sioux",
        "Wooden garden shed with double doors and a charming windowed front."
    ]
    images = [
        Image.open(str(test_images_path_similarity / "000000000139.jpg")),
        Image.open(str(test_images_path_similarity / "000000000081.jpg"))
    ]
    mask = filter_by_clip_similarity(captions, images, dfn_model, clip_processor, dfn_similarity_score_threshold=4, clip_caption_max_length=77)
    assert mask == [False, True]