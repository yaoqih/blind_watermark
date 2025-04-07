# blind-watermark

基于频域的数字盲水印  


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
- **English readme** [README.md](README.md)
- **Source code:** [https://github.com/guofei9987/blind_watermark](https://github.com/guofei9987/blind_watermark)

# 安装
```bash
pip install blind-watermark
```

或者安装最新开发版本
```bach
git clone git@github.com:guofei9987/blind_watermark.git
cd blind_watermark
pip install .
```

# 如何使用

## 命令行中使用

```bash
# 嵌入水印：
blind_watermark --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
# 提取水印：
blind_watermark --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png
```



## Python 中使用

原图 + 水印 = 打上水印的图

![origin_image](docs/原图.jpeg) + '@guofei9987 开源万岁！' = ![打上水印的图](docs/打上水印的图.jpg)



参考 [代码](/examples/example_str.py)


嵌入水印
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


提取水印
```python
bwm1 = WaterMark(password_img=1, password_wm=1)
wm_extract = bwm1.extract('output/embedded.png', wm_shape=len_wm, mode='str')
print(wm_extract)
```
Output:
>@guofei9987 开源万岁！


## 带元数据的文本水印

我们提供了更简单的API用于嵌入和提取文本水印。这些方法还支持将水印存储在图像元数据中，以提高水印的鲁棒性。

### 嵌入文本水印：

```python
from blind_watermark import WaterMark

# 初始化水印对象
wm = WaterMark()

# 嵌入文本水印
wm.embed_text(
    text="这是一个支持元数据的水印",
    filename="pic/ori_img.jpg",
    out_filename="output/embedded_with_metadata.png",
    use_metadata=True  # 启用元数据水印（默认为True）
)
```

该函数将：
1. 使用鲁棒算法将文本水印嵌入到图像像素中
2. 同时将相同的水印存储在图像元数据中（如果use_metadata=True）
3. 支持多种图像格式：
   - PNG, TIFF：使用标准元数据
   - JPG/JPEG：使用EXIF元数据（需要`piexif`库）

### 提取文本水印：

```python
from blind_watermark import WaterMark

# 初始化水印对象
wm = WaterMark()

# 提取文本水印
extracted_text = wm.extract_text(
    filename="output/embedded_with_metadata.png",
    check_metadata=True  # 首先尝试元数据，然后是像素水印（默认为True）
)

print(extracted_text)
```

该函数将：
1. 如果check_metadata=True，首先尝试从元数据中提取水印
2. 如果元数据提取失败或不可用，回退到从像素中提取
3. 支持多种图像格式，包括PNG, TIFF, JPG/JPEG

### 功能和优势：

- **双重保护**：水印同时存储在像素和元数据中
- **格式支持**：适用于PNG, TIFF, JPG/JPEG图像格式
- **错误纠正**：使用Reed-Solomon编码进行错误纠正
- **压缩**：应用zlib压缩以存储更多信息
- **自动恢复**：如果元数据损坏，会回退到像素提取

### JPEG元数据支持依赖：

要在JPEG图像中启用元数据水印，需要安装piexif库：
```bash
pip install piexif
```

### 各种攻击后的效果

|攻击方式|攻击后的图片|提取的水印|
|--|--|--|
|旋转攻击45度|![旋转攻击](docs/旋转攻击.jpg)|'@guofei9987 开源万岁！'|
|随机截图|![截屏攻击](docs/截屏攻击2_还原.jpg)|'@guofei9987 开源万岁！'|
|多遮挡| ![多遮挡攻击](docs/多遮挡攻击.jpg) |'@guofei9987 开源万岁！'|
|纵向裁剪|![横向裁剪攻击](docs/横向裁剪攻击_填补.jpg)|'@guofei9987 开源万岁！'|
|横向裁剪|![纵向裁剪攻击](docs/纵向裁剪攻击_填补.jpg)|'@guofei9987 开源万岁！'|
|缩放攻击|![缩放攻击](docs/缩放攻击.jpg)|'@guofei9987 开源万岁！'|
|椒盐攻击|![椒盐攻击](docs/椒盐攻击.jpg)|'@guofei9987 开源万岁！'|
|亮度攻击|![亮度攻击](docs/亮度攻击.jpg)|'@guofei9987 开源万岁！'|



### 嵌入图片

参考 [代码](/examples/example_str.py)


嵌入：
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

提取：
```python
bwm1 = WaterMark(password_wm=1, password_img=1)
# notice that wm_shape is necessary
bwm1.extract(filename='output/embedded.png', wm_shape=(128, 128), out_wm_name='output/extracted.png', )
```

|攻击方式|攻击后的图片|提取的水印|
|--|--|--|
|旋转攻击45度|![旋转攻击](docs/旋转攻击.jpg)|![](docs/旋转攻击_提取水印.png)|
|随机截图|![截屏攻击](docs/截屏攻击2_还原.jpg)|![](docs/旋转攻击_提取水印.png)|
|多遮挡| ![多遮挡攻击](docs/多遮挡攻击.jpg) |![多遮挡_提取水印](docs/多遮挡攻击_提取水印.png)|



### 隐水印还可以是二进制数据

参考 [代码](/examples/example_bit.py)


作为 demo， 如果要嵌入是如下长度为6的二进制数据
```python
wm = [True, False, True, True, True, False]
```

嵌入水印

```python
# 除了嵌入图片，也可以嵌入比特类数据
from blind_watermark import WaterMark

bwm1 = WaterMark(password_img=1, password_wm=1)
bwm1.read_ori_img('pic/ori_img.jpg')
bwm1.read_wm([True, False, True, True, True, False], mode='bit')
bwm1.embed('output/打上水印的图.png')
```

解水印：（注意设定水印形状 `wm_shape`）
```python
bwm1 = WaterMark(password_img=1, password_wm=1, wm_shape=6)
wm_extract = bwm1.extract('output/打上水印的图.png', mode='bit')
print(wm_extract)
```

解出的水印是一个0～1之间的实数，方便用户自行卡阈值。如果水印信息量远小于图片可容纳量，偏差极小。

# 并行计算

```python
WaterMark(..., processes=None)
```
- `processes`: 整数，指定线程数。默认为 `None`, 表示使用全部线程。


## 相关项目

- text_blind_watermark (文本盲水印，把信息隐秘地打入文本): [https://github.com/guofei9987/text_blind_watermark](https://github.com/guofei9987/text_blind_watermark)  
- HideInfo（藏物于图、藏物于音、藏图于文）：[https://github.com/guofei9987/HideInfo](https://github.com/guofei9987/HideInfo)
# 水印嵌入测试工具

这个项目提供了一组工具，用于测试不同分辨率图像嵌入不同长度文本的水印效果，并生成图像分辨率与可嵌入文本长度的关系模型。

## 主要功能

1. **批量测试不同分辨率图像嵌入不同长度文本的错误率**
2. **可视化测试结果，包括错误率与文本长度、图像分辨率的关系**
3. **自动生成推荐公式，用于预测图像能够安全嵌入的最大文本长度**
4. **提供单独的预测工具，快速计算图像可嵌入的安全文本长度**

## 文件说明

- `watermark_resolution_test.py`: 主要测试脚本，包含测试、分析和可视化相关函数
- `run_watermark_test.py`: 简化版测试脚本，调用主脚本进行测试
- `watermark_prediction.py`: 独立的预测工具，根据训练结果预测图像可嵌入的最大文本长度

## 使用方法

### 1. 运行测试

执行以下命令运行默认测试：

```bash
python run_watermark_test.py
```

这将使用预定义的图像和文本长度进行测试，结果保存在 `watermark_test_results` 目录中。

### 2. 查看分析结果

测试完成后，可以在 `watermark_test_results` 目录中找到以下文件：

- `watermark_test_results.csv`: 所有测试结果的CSV文件
- `max_viable_text_length.csv`: 每个图像的最大可用文本长度
- `recommendation_formula.txt`: 生成的推荐公式
- `error_rate_by_text_length.png`: 错误率与文本长度关系图
- `max_length_by_resolution.png`: 最大可用文本长度与图像分辨率关系图
- `safe_length_prediction.png`: 推荐公式拟合效果图

### 3. 使用预测工具

通过以下命令可以预测指定图像能够安全嵌入的最大文本长度：

```bash
python watermark_prediction.py --images image1.jpg image2.png --ratio 0.01 --output predictions
```

参数说明：
- `--images`: 需要预测的图像路径，可以指定多个
- `--ratio`: 每像素可安全嵌入的字符数，默认为0.01（即每100像素可嵌入1个字符）
- `--output`: 输出目录，默认为 'watermark_predictions'

## 注意事项

1. 测试过程可能比较耗时，尤其是对于高分辨率图像和长文本
2. 推荐公式是基于测试数据拟合的，实际效果可能会有差异
3. 建议针对特定场景进行测试，以获得更准确的推荐值

## 示例

以下是典型的测试流程：

1. 修改 `run_watermark_test.py` 中的测试图像和文本长度
2. 运行测试脚本生成测试结果
3. 根据生成的推荐公式更新 `watermark_prediction.py` 中的默认 `safe_ratio` 值
4. 使用预测工具快速估算新图像的安全文本长度
