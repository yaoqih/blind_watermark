# 水印嵌入测试工具

这个项目提供了一组工具，用于测试不同分辨率图像嵌入不同长度文本的水印效果，并生成图像分辨率与可嵌入文本长度的关系模型。

## 主要功能

1. **批量测试不同分辨率图像嵌入不同长度文本的错误率**
2. **测试同一图像缩放到不同分辨率后的水印嵌入效果**
3. **可视化测试结果，包括错误率与文本长度、图像分辨率的关系**
4. **自动生成推荐公式，用于预测图像能够安全嵌入的最大文本长度**
5. **提供单独的预测工具，快速计算图像可嵌入的安全文本长度**

## 文件说明

- `watermark_resolution_test.py`: 主要测试脚本，包含测试、分析和可视化相关函数
- `run_watermark_test.py`: 简化版测试脚本，调用主脚本进行测试
- `resize_image_test.py`: 用于测试同一图像缩放到不同分辨率后的水印嵌入效果
- `watermark_prediction.py`: 独立的预测工具，根据训练结果预测图像可嵌入的最大文本长度

## 使用方法

### 1. 运行不同图像测试

执行以下命令测试多个不同图像的水印嵌入效果：

```bash
python run_watermark_test.py
```

这将使用预定义的图像和文本长度进行测试，结果保存在 `watermark_test_results` 目录中。

### 2. 运行图像缩放测试

执行以下命令测试同一图像缩放到不同分辨率后的水印嵌入效果：

```bash
python resize_image_test.py
```

这将使用高分辨率源图像，缩放到不同分辨率后进行测试，结果保存在 `watermark_resize_test` 目录中。

支持两种缩放模式：
- 保持原图宽高比（默认）：只需指定目标宽度，高度会根据原图比例自动计算
- 不保持宽高比：指定目标宽度和高度，强制缩放到指定分辨率

### 3. 查看分析结果

测试完成后，可以在输出目录中找到以下文件：

- `watermark_test_results.csv` / `resized_watermark_test_results.csv`: 所有测试结果的CSV文件
- `max_viable_text_length.csv` / `resized_max_viable_text_length.csv`: 每个分辨率的最大可用文本长度
- `recommendation_formula.txt`: 生成的推荐公式
- `error_rate_by_text_length.png` / `resized_error_rate_by_text_length.png`: 错误率与文本长度关系图
- `max_length_by_resolution.png` / `resized_max_length_by_resolution.png`: 最大可用文本长度与图像分辨率关系图
- `safe_length_prediction.png`: 推荐公式拟合效果图

### 4. 使用预测工具

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
4. 在图像缩放测试中，建议保持原图的宽高比，以避免图像失真对测试结果的影响

## 示例

以下是典型的测试流程：

1. 运行 `resize_image_test.py` 测试同一图像缩放到不同分辨率的效果
2. 查看生成的 `resized_watermark_test_results.csv` 和可视化图表
3. 根据生成的推荐公式更新 `watermark_prediction.py` 中的默认 `safe_ratio` 值
4. 使用预测工具快速估算新图像的安全文本长度 