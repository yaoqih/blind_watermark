


# blind-watermark

Blind watermark based on DWT-DCT-SVD.


[![PyPI](https://img.shields.io/pypi/v/blind_watermark)](https://pypi.org/project/blind_watermark/)
[![Build Status](https://travis-ci.com/guofei9987/blind_watermark.svg?branch=master)](https://travis-ci.com/guofei9987/blind_watermark)
[![codecov](https://codecov.io/gh/guofei9987/blind_watermark/branch/master/graph/badge.svg)](https://codecov.io/gh/guofei9987/blind_watermark)
[![License](https://img.shields.io/pypi/l/blind_watermark.svg)](https://github.com/guofei9987/blind_watermark/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python->=3.5-green.svg)
![Platform](https://img.shields.io/badge/platform-windows%20|%20linux%20|%20macos-green.svg)
[![stars](https://img.shields.io/github/stars/guofei9987/blind_watermark.svg?style=social)](https://github.com/guofei9987/blind_watermark/)
[![fork](https://img.shields.io/github/forks/guofei9987/blind_watermark?style=social)](https://github.com/guofei9987/blind_watermark/fork)
[![Downloads](https://pepy.tech/badge/blind-watermark)](https://pepy.tech/project/blind-watermark)
[![Discussions](https://img.shields.io/badge/discussions-green.svg)](https://github.com/guofei9987/blind_watermark/discussions)


- **Documentation:** [https://BlindWatermark.github.io/blind_watermark/#/en/](https://BlindWatermark.github.io/blind_watermark/#/en/)
- **文档：** [https://BlindWatermark.github.io/blind_watermark/#/zh/](https://BlindWatermark.github.io/blind_watermark/#/zh/)  
- **中文 readme** [README_cn.md](README_cn.md)
- **Source code:** [https://github.com/guofei9987/blind_watermark](https://github.com/guofei9987/blind_watermark)



# install
```bash
pip install blind-watermark
```

For the current developer version:
```bach
git clone git@github.com:guofei9987/blind_watermark.git
cd blind_watermark
pip install .
```

# How to use


## Use in bash


```bash
# embed watermark into image:
blind_watermark --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
# extract watermark from image:
blind_watermark --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png
```



## Use in Python

Original Image + Watermark = Watermarked Image

![origin_image](docs/原图.jpeg) + '@guofei9987 开源万岁！' = ![打上水印的图](docs/打上水印的图.jpg)


See the [codes](/examples/example_str.py)

Embed watermark:
```python
from blind_watermark import WaterMark

bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_img('pic/ori_img.jpg')
wm = '@guofei9987 开源万岁！'
bwm1.read_wm(wm, mode='str')
bwm1.embed('output/embedded.png')
len_wm = len(bwm1.wm_bit)
print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))
```

Extract watermark:
```python
bwm1 = WaterMark(password_img=1, password_wm=1)
wm_extract = bwm1.extract('output/embedded.png', wm_shape=len_wm, mode='str')
print(wm_extract)
```
Output:
>@guofei9987 开源万岁！


## Text Watermark with Metadata

We provide a simpler API for embedding and extracting text watermarks. These methods also support storing watermarks in image metadata for better robustness.

### Embed text watermark:

```python
from blind_watermark import WaterMark

# Initialize watermark object
wm = WaterMark()

# Embed text watermark
wm.embed_text(
    text="This is a watermark with metadata support",
    filename="pic/ori_img.jpg",
    out_filename="output/embedded_with_metadata.png",
    use_metadata=True  # Enable metadata watermarking (default is True)
)
```

This function will:
1. Embed the text watermark into image pixels using robust algorithms
2. Store the same watermark in image metadata (if use_metadata=True)
3. Support various image formats:
   - PNG, TIFF: Uses standard metadata
   - JPG/JPEG: Uses EXIF metadata (requires `piexif` library)

### Extract text watermark:

```python
from blind_watermark import WaterMark

# Initialize watermark object
wm = WaterMark()

# Extract text watermark
extracted_text = wm.extract_text(
    filename="output/embedded_with_metadata.png",
    check_metadata=True  # Try metadata first, then pixel watermark (default is True)
)

print(extracted_text)
```

This function will:
1. First try to extract the watermark from metadata if check_metadata=True
2. If metadata extraction fails or is unavailable, fall back to extracting from pixels
3. Support various image formats including PNG, TIFF, JPG/JPEG

### Features and Benefits:

- **Dual Protection**: Stores watermark in both pixels and metadata
- **Format Support**: Works with PNG, TIFF, JPG/JPEG image formats
- **Error Correction**: Uses Reed-Solomon coding for error correction
- **Compression**: Applies zlib compression to store more information
- **Automatic Recovery**: Falls back to pixel extraction if metadata is damaged

### Dependencies for JPEG Metadata Support:

To enable metadata watermarking in JPEG images, install the piexif library:
```bash
pip install piexif
```

### attacks on Watermarked Image


|attack method|image after attack|extracted watermark|
|--|--|--|
|Rotate 45 Degrees|![旋转攻击](docs/旋转攻击.jpg)|'@guofei9987 开源万岁！'|
|Random crop|![截屏攻击](docs/截屏攻击2_还原.jpg)|'@guofei9987 开源万岁！'|
|Masks| ![多遮挡攻击](docs/多遮挡攻击.jpg) |'@guofei9987 开源万岁！'|
|Vertical cut|![横向裁剪攻击](docs/横向裁剪攻击_填补.jpg)|'@guofei9987 开源万岁！'|
|Horizontal cut|![纵向裁剪攻击](docs/纵向裁剪攻击_填补.jpg)|'@guofei9987 开源万岁！'|
|Resize|![缩放攻击](docs/缩放攻击.jpg)|'@guofei9987 开源万岁！'|
|Pepper Noise|![椒盐攻击](docs/椒盐攻击.jpg)|'@guofei9987 开源万岁！'|
|Brightness 10% Down|![亮度攻击](docs/亮度攻击.jpg)|'@guofei9987 开源万岁！'|






### embed images

embed watermark:
```python
from blind_watermark import WaterMark

bwm1 = WaterMark(password_wm=1, password_img=1)
# read original image
bwm1.read_img('pic/ori_img.jpg')
# read watermark
bwm1.read_wm('pic/watermark.png')
# embed
bwm1.embed('output/embedded.png')
```


Extract watermark:
```python
bwm1 = WaterMark(password_wm=1, password_img=1)
# notice that wm_shape is necessary
bwm1.extract(filename='output/embedded.png', wm_shape=(128, 128), out_wm_name='output/extracted.png', )
```


|attack method|image after attack|extracted watermark|
|--|--|--|
|Rotate 45 Degrees|![旋转攻击](docs/旋转攻击.jpg)|![](docs/旋转攻击_提取水印.png)|
|Random crop|![截屏攻击](docs/截屏攻击2_还原.jpg)|![多遮挡_提取水印](docs/多遮挡攻击_提取水印.png)|
|Mask| ![多遮挡攻击](docs/多遮挡攻击.jpg) |![多遮挡_提取水印](docs/多遮挡攻击_提取水印.png)|


### embed array of bits

See it [here](/examples/example_bit.py)


As demo, we embed 6 bytes data:
```python
wm = [True, False, True, True, True, False]
```

Embed:
```python
from blind_watermark import WaterMark

bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_ori_img('pic/ori_img.jpg')
bwm1.read_wm([True, False, True, True, True, False], mode='bit')
bwm1.embed('output/embedded.png')
```

Extract:
```python
bwm1 = WaterMark(password_img=1, password_wm=1, wm_shape=6)
wm_extract = bwm1.extract('output/打上水印的图.png', mode='bit')
print(wm_extract)
```
Notice that `wm_shape` (shape of watermark) is necessary

The output `wm_extract` is an array of float. set a threshold such as 0.5.


# Concurrency

```python
WaterMark(..., processes=None)
```
- `processes` number of processes, can be integer. Default `None`, which means using all processes.  

## Related Project

- text_blind_watermark (Embed message into text): [https://github.com/guofei9987/text_blind_watermark](https://github.com/guofei9987/text_blind_watermark)  
- HideInfo（hide as image, hide as sounds, hide as text）：[https://github.com/guofei9987/HideInfo](https://github.com/guofei9987/HideInfo)
