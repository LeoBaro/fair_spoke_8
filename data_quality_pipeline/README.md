# fair_spoke_8

## Download on DaVinci
```bash
git clone https://${GITHUB_USER}:${GITHUB_TOKEN}:@github.com/${GITHUB_REPOSITORY}
```

## Setup

This project has been setup with uv
```bash
uv init --lib --name made --build-backend hatch
```
To get started:
```bash
uv python install 3.10
uv venv
```
To add dependencies:
```bash
uv add <name> 
```

## Ray

### Interactive multi-node setup

Obtain n nodes:
```bash
qsub -I -l select=2:ncpus=24:ngpus=4 -l place=scatter -q gpu
```
Start the ray cluster:
```bash
source start_ray_multimode.sh
```
Run the pipeline:
```bash
python src/made/bin/made.py --shards_path --ray-address --log-folder --output-folder --config-path
```

### Automatic multi-node setup
```bash
qsub -l select=2:ncpus=24:ngpus=4 -l place=scatter -q gpu start_ray_cluster_and_pipeline.sh
```

## Connecting to the Ray dashboard
Get the dashboard port
```python
import ray
context = ray.init()
print(context.dashboard_url)
```
Make an ssh tunnel login node -- gpu node
```bash
ssh -N -L 8888:localhost:<rayport> <gpu-node-ip>
```

Make an ssh tunnel vdi -- login node
```bash
ssh -N -L 8888:localhost:8888 lbaroncelli@10.122.0.6
```
Go to: localhost:8888



## Ray webdataset loader
```bash
{
    '__key__': '000000000084', 
    'jpg': array( [[]], dtype=uint8), 
    'json': {
        'uid': 'd84cefb32975139db7f9cfa4e9f00fb0', 
        'face_bboxes': [], 
        'caption': 'Picture 13', 
        'url': 'http://media.rightmove.co.uk/94k/93860/60569927/93860_9gallipothill_IMG_12_0000_max_135x100.JPG', 
        'key': '000000000084', 
        'status': 'success', 
        'error_message': None, 
        'width': 135, 
        'height': 90, 
        'original_width': 135, 
        'original_height': 90, 
        'exif': '{"Image Make": "Canon", "Image Model": "Canon EOS 5D Mark II", "Image Orientation": "Horizontal (normal)", "Image XResolution": "72", "Image YResolution": "72", "Image ResolutionUnit": "Pixels/Inch", "Image Software": "Adobe Photoshop CS3 Macintosh", "Image DateTime": "2016:07:12 08:20:16", "Image Artist": "Photographer: Mark Wood", "Image YCbCrPositioning": "Co-sited", "Image Copyright": "Copyright: mandnwood@aol.com", "Image ExifOffset": "316", "GPS GPSVersionID": "[0, 0, 0, 0]", "Image GPSInfo": "1084", "Thumbnail Compression": "JPEG (old-style)", "Thumbnail XResolution": "72", "Thumbnail YResolution": "72", "Thumbnail ResolutionUnit": "Pixels/Inch", "Thumbnail JPEGInterchangeFormat": "1198", "Thumbnail JPEGInterchangeFormatLength": "6328", "EXIF ExposureTime": "1/13", "EXIF FNumber": "14", "EXIF ExposureProgram": "Aperture Priority", "EXIF ISOSpeedRatings": "250", "EXIF ExifVersion": "", "EXIF DateTimeOriginal": "2016:07:11 11:13:09", "EXIF DateTimeDigitized": "2016:07:11 11:13:09", "EXIF ComponentsConfiguration": "", "EXIF ShutterSpeedValue": "29/8", "EXIF ApertureValue": "61/8", "EXIF ExposureBiasValue": "-2/3", "EXIF MeteringMode": "Pattern", "EXIF Flash": "Flash fired, compulsory flash mode", "EXIF FocalLength": "17", "EXIF SubSecTime": "92", "EXIF SubSecTimeOriginal": "92", "EXIF SubSecTimeDigitized": "92", "EXIF FlashPixVersion": "", "EXIF ColorSpace": "sRGB", "EXIF ExifImageWidth": "1024", "EXIF ExifImageLength": "683", "Interoperability InteroperabilityIndex": "R98", "Interoperability InteroperabilityVersion": "[0, 0, 0, 0]", "EXIF InteroperabilityOffset": "1052", "EXIF FocalPlaneXResolution": "4080000/1459", "EXIF FocalPlaneYResolution": "1360000/479", "EXIF FocalPlaneResolutionUnit": "2", "EXIF CustomRendered": "Normal", "EXIF ExposureMode": "Auto Exposure", "EXIF WhiteBalance": "Auto", "EXIF SceneCaptureType": "Standard"}', 'sha256': '99c960b6158fb886335484749fe40e86ada268d0f4bd1e7e4c8506723c98ae18'}, 
    'txt': 'Picture 13', 
    '__url__': '/home/leobaro/Downloads/datasets/web/datacomp/shards/00000000.tar'
}
```


# Using `img2dataset` and Choosing Image Size for CLIP Training  

## **Understanding `resize_mode` in `img2dataset`**  
The `resize_mode` parameter controls how images are resized when downloading them.  

### **Default Value: `keep_ratio_largest`**  
- Preserves the aspect ratio.  
- Resizes the **largest side** (width or height) to match `image_size`.  
- The smaller side scales proportionally (no cropping or padding).  

### **Other Options:**  
| Resize Mode          | Description |
|----------------------|-------------|
| **`no`**            | No resizing. |
| **`border`**        | Resizes to `image_size × image_size` with padding. |
| **`keep_ratio`**    | The **smallest side** is resized to `image_size`, making the other side larger. |
| **`keep_ratio_largest`** (default) | The **largest side** is resized to `image_size`, making the other side smaller. |
| **`center_crop`**   | Crops the largest side to make the image square (`image_size × image_size`). |

---

## **Best Image Size for Training CLIP**  
The ideal image size depends on compute resources and the CLIP model architecture.  

### **Commonly Used Image Sizes:**  
1. **224×224 (Standard, Efficient)**
   - Used in OpenAI’s original CLIP.
   - Works well with **ViT and ResNet-based models**.
   - Fast and memory-efficient.  

2. **336×336 (More Detail, Slightly Heavier)**
   - Used in some CLIP variants.
   - Captures finer details, improving accuracy.  

3. **448×448 or Higher (High-Resolution)**
   - Useful for **large ViT models** (e.g., ViT-L/14).
   - Computationally expensive but **better for detailed tasks**.  

### **Trade-Offs Between Sizes:**
| Image Size | Pros | Cons |
|------------|------|------|
| **224×224** | Efficient, widely used | May lose fine details |
| **336×336** | More detail, slightly better accuracy | Higher compute/memory cost |
| **448×448** | Best detail, useful for large models | Very slow, expensive |

### **Recommendations:**
- **For standard CLIP training:** **224×224** is the best balance.  
- **If you have more resources:** Try **336×336**.  
- **For large-scale models:** Consider **448×448**, but ensure **high-quality data and strong compute**.  

---

## **How Image Size Relates to Different CLIP Models**  
CLIP models differ based on their **Vision Transformer (ViT) architecture**.  

### **CLIP Model Variants:**
- **ViT-B/32** → Base model, **32×32 patch size**.  
- **ViT-B/16** → Base model, **16×16 patch size**.  
- **ViT-L/14** → Large model, **14×14 patch size**.  
- **ViT-H/14** → Huge model, **14×14 patch size**.  

Since ViTs split images into patches, the **image size must be divisible by the patch size** for efficiency.  

### **Recommended Image Sizes per CLIP Model:**
| CLIP Model  | Patch Size | Best Image Size |
|-------------|------------|----------------|
| **ViT-B/32** | 32×32      | **224×224** (7×7 patches) |
| **ViT-B/16** | 16×16      | **224×224 or 336×336** |
| **ViT-L/14** | 14×14      | **336×336 or 448×448** |
| **ViT-H/14** | 14×14      | **448×448** |

### **Final Recommendations:**
- **ViT-B/32 or ViT-B/16** → **224×224** is sufficient.  
- **ViT-L/14** → **336×336** provides better accuracy.  
- **ViT-H/14** → **448×448**, but requires **powerful compute**.  

---

## **Difference in Parameters Between CLIP Models**  
Larger CLIP models have more parameters due to increased **layers, hidden dimensions, and attention heads**.  

### **Parameter Counts for CLIP Models**
| Model       | Layers (L) | Hidden Dim (D) | Attention Heads | Patch Size | **ViT Only** (M) | **ViT + Text Encoder** (M) |
|------------|-----------|---------------|----------------|------------|------------------|------------------------------|
| **ViT-B/32** | 12        | 768           | 12             | 32×32       | **86M**          | **149M**                      |
| **ViT-B/16** | 12        | 768           | 12             | 16×16       | **86M**          | **149M**                      |
| **ViT-L/14** | 24        | 1024          | 16             | 14×14       | **304M**         | **428M**                      |
| **ViT-H/14** | 32        | 1280          | 16             | 14×14       | **632M**         | **831M**                      |

### **Key Differences:**
- **More Layers** → Deeper models capture **more complex features**.  
- **Larger Hidden Dimensions** → Each layer processes **more information**.  
- **More Attention Heads** → Improves **multi-head self-attention**.  
- **Smaller Patch Size** → More patches = **higher-resolution representation**.  

### **Computational Considerations:**
| Model       | **Memory Usage** | **Training Speed** | **Performance** |
|------------|-----------------|------------------|----------------|
| **ViT-B/32** | Low             | Fast             | Good baseline |
| **ViT-B/16** | Moderate        | Medium           | Better details |
| **ViT-L/14** | High            | Slower           | Higher accuracy |
| **ViT-H/14** | Very High       | Very Slow        | Best, but expensive |

---

## **Final Thoughts**
- **For efficiency** → **ViT-B/32 with 224×224**.  
- **For accuracy improvement** → **ViT-B/16 or ViT-L/14 with 336×336**.  
- **For best performance** → **ViT-H/14 with 448×448**, but requires **high-end compute**.  

Would you like help setting up your `img2dataset` config for a specific CLIP model? 🚀  


# Implementations

* SemDeDup: 
  * https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/semdedup.html
  * https://github.com/Guhaifudeng/SemDeDup