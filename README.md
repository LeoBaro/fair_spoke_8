
# Required models

```
mkdir models
```

## Unimodal Text
Fasttext:
```
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
mv lid.176.bin models
```
Spacy:
```
python -m spacy download em_core_web_trf
```
Pos tags:
```
wget <link drive>
mv common_pos_patterns.txt models
```

## Unimodal Vision
```python
import easyocr
easyocr.Reader(['en'], gpu=True, user_network_directory="./models")
```

## Multimodal
```python
from transformers import CLIPProcessor, CLIPModel
CLIPModel.from_pretrained("leobaro/DFN-public")
CLIPProcessor.from_pretrained("leobaro/DFN-public")
```

## Specificy
```
wget https://drive.google.com/file/d/1Ab6z1uVvNQM7X3sc1WQenKhHopv03La4/view?usp=drive_link reference.pt
wget https://drive.google.com/file/d/1MM41wBaifh3YKMDU964qMC6v9x1rfPYo/view?usp=drive_link ckpt.pt
mv reference.pt models
mv ckpt.pt models
```

## Deduplication
```python
from transformers import CLIPProcessor, CLIPModel
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```