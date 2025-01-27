# fair_spoke_8

## Download on DaVinci
```bash
git clone https://${GITHUB_USER}:${GITHUB_TOKEN}:@github.com/${GITHUB_REPOSITORY}
```

## Setup

This project has been setup with uv
```bash
uv init --lib --name spoke8leonardo --build-backend hatch
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