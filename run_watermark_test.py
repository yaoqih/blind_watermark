from watermark_resolution_test import batch_test_watermark

# 定义测试图像
test_images = [
    "image_test/1.png",     # 低分辨率 (456x1689)
    "image_test/3.png",     # 中低分辨率
    "image_test/20.2.png",  # 中分辨率
    "image_test/5.1.jpg",   # 中高分辨率
    "image_test/1.1.png",   # 高分辨率 (8.8MB)
]

# 定义测试文本长度
# 从短到长逐步增加，可以更精确地找出临界点
text_lengths = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000]

# 设置输出目录
output_dir = "watermark_test_results"

# 执行批量测试
print("开始测试不同分辨率图像嵌入不同长度文本的错误率...")
results = batch_test_watermark(test_images, text_lengths, output_dir)
print(f"测试完成，结果已保存到 {output_dir} 目录") 