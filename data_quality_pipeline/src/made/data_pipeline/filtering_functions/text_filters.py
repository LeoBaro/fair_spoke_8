
import spacy
import fasttext
from typing import List

def filter_captions_by_length(
        captions: List[str], 
        min_words: int, 
        min_chars: int
    ) -> List[bool]:
    """Filter captions by minimum word and character count"""
    cleaned_captions = [caption.strip().replace('\n', ' ') for caption in captions]
    
    return [
        len(caption.split()) >= min_words and len(caption) >= min_chars
        for caption in cleaned_captions
    ]

def filter_captions_by_language(
        captions: List[str], 
        model: fasttext.FastText,
        target_language: str, 
        threshold: float
    ) -> List[bool]:
    """Filter captions by language detection"""
    cleaned_captions = [caption.strip().replace('\n', ' ') for caption in captions]
    predictions, scores = model.predict(cleaned_captions)
    
    return [
        (pred[0] == f"__label__{target_language}") and (score[0] > threshold)
        for pred, score in zip(predictions, scores)
    ]

def filter_captions_by_pos_tags(
        captions: List[str], 
        model: spacy.Language,
        target_pos_tags: List[str]
    ) -> List[bool]:
    """Filter captions by POS tag patterns"""
    mask = []
    for caption in captions:
        doc = model(caption)
        pos_tags = [token.pos_ for token in doc]
        sorted_unique_tags = sorted(set(pos_tags))
        pos_string = "_".join(sorted_unique_tags)
        mask.append(pos_string in target_pos_tags)
    return mask
