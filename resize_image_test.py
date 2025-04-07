from watermark_resolution_test import test_resized_images
import numpy as np

# 源图像路径（选择一个高分辨率图像作为源图像）
original_image_path = "image_test/1.1.png"  # 高分辨率图像

# 是否保持原图宽高比例
keep_aspect_ratio = True

# 使用指数递增的图像宽度，这样可以更均匀地测试不同像素大小的图像
# 从320到3840，共11个点
min_width = 320
max_width = 3840
num_points = 11
widths = np.round(np.exp(np.linspace(np.log(min_width), np.log(max_width), num_points))).astype(int)

# 根据是否保持宽高比决定高度计算方式
if keep_aspect_ratio:
    # 只需要提供宽度，高度会根据原图比例自动计算
    # 使用dummy值作为高度占位符
    resolutions = [(width, 0) for width in widths]
else:
    # 如果不保持宽高比，定义标准的宽高比例
    # 常见的宽高比：4:3, 16:9, 3:2
    aspect_ratio = 16/9  # 使用16:9宽高比
    resolutions = [(width, int(width / aspect_ratio)) for width in widths]

# 定义测试文本长度
# 也使用指数递增的方式，更均匀地测试不同长度
min_length = 10
max_length = 5000
num_lengths = 12
text_lengths = np.round(np.exp(np.linspace(np.log(min_length), np.log(max_length), num_lengths))).astype(int).tolist()

# 设置输出目录
output_dir = "watermark_resize_test"

# 打印测试配置
print("测试配置:")
print(f"- 源图像: {original_image_path}")
print(f"- 保持原图宽高比: {keep_aspect_ratio}")
print(f"- 测试分辨率: {len(resolutions)} 种")
for i, (width, height) in enumerate(resolutions):
    print(f"  {i+1}. {width}x{'自动计算' if keep_aspect_ratio else height}")
print(f"- 测试文本长度: {len(text_lengths)} 种")
print(f"  {text_lengths}")
print()

# 执行缩放图像批量测试
print(f"开始测试同一图像缩放到不同分辨率后的水印嵌入效果...")
results = test_resized_images(
    original_image_path, 
    resolutions, 
    text_lengths, 
    output_dir,
    keep_aspect_ratio=keep_aspect_ratio
)
print(f"测试完成，结果已保存到 {output_dir} 目录") 