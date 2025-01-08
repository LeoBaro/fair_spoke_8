from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

def save_field_to_txt(dataset_split, filename, field: str):
    with open(filename, 'w', encoding='utf-8') as f:
        size = len(dataset_split)
        for entry in tqdm(dataset_split, total=size):
            f.write(entry[field] + '\n')

def main():
    ds = load_dataset("yuvalkirstain/laion-hd-subset", split=None)

    if not Path('train_urls.txt').exists():
        save_field_to_txt(ds['train'], 'train_urls.txt', 'url')
    if not Path('train_cap.txt').exists():
        save_field_to_txt(ds['train'], 'train_cap.txt', 'caption')
    if not Path('test_urls.txt').exists():
        save_field_to_txt(ds['test'], 'test_urls.txt', 'url')
    if not Path('test_cap.txt').exists():
        save_field_to_txt(ds['test'], 'test_cap.txt', 'caption')

    print(ds["train"][0])
    
if __name__ == "__main__":
    main()
