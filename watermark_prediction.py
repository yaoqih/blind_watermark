import os
import argparse
from PIL import Image
from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_image_resolution(image_path: str) -> Tuple[int, int]:
    """获取图像分辨率"""
    img = Image.open(image_path)
    return img.size


def calculate_recommended_text_length(width: int, height: int, safe_ratio: float = 0.01) -> int:
    """根据图像分辨率计算推荐的最大文本长度
    
    Args:
        width: 图像宽度
        height: 图像高度
        safe_ratio: 每像素可安全嵌入的字符数，默认为0.01（即每100像素可嵌入1个字符）
        
    Returns:
        推荐的最大文本长度
    """
    # 计算图像像素总数
    pixel_count = width * height
    
    # 应用一个安全系数(0.8)，以确保推荐值偏保守
    recommended_length = int(pixel_count * safe_ratio * 0.8)
    
    return max(10, recommended_length)  # 确保至少返回10


def predict_safe_text_length(image_paths: List[str], safe_ratio: float = 0.01, output_dir: str = None) -> pd.DataFrame:
    """预测多个图像能够安全嵌入的最大文本长度
    
    Args:
        image_paths: 图像路径列表
        safe_ratio: 每像素可安全嵌入的字符数，默认为0.01
        output_dir: 输出目录，如果提供则保存结果
        
    Returns:
        包含预测结果的DataFrame
    """
    results = []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"图像不存在: {image_path}")
            continue
            
        try:
            # 获取图像分辨率
            width, height = get_image_resolution(image_path)
            image_name = os.path.basename(image_path)
            image_size = width * height
            
            # 计算推荐文本长度
            recommended_length = calculate_recommended_text_length(width, height, safe_ratio)
            
            results.append({
                "image_name": image_name,
                "image_path": image_path,
                "width": width,
                "height": height,
                "image_size": image_size,
                "recommended_text_length": recommended_length
            })
            
            print(f"图像 {image_name} (分辨率: {width}x{height}, 大小: {image_size} 像素)")
            print(f"推荐最大文本长度: {recommended_length} 字符")
            print("------")
            
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {str(e)}")
    
    # 创建DataFrame
    result_df = pd.DataFrame(results)
    
    # 如果提供了输出目录，保存结果
    if output_dir and result_df.shape[0] > 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存CSV结果
        result_file = os.path.join(output_dir, "predicted_safe_lengths.csv")
        result_df.to_csv(result_file, index=False, encoding='utf-8')
        
        # 可视化结果
        if len(result_df) > 1:
            plt.figure(figsize=(12, 8))
            plt.scatter(result_df['image_size'], result_df['recommended_text_length'])
            plt.xlabel('图像大小 (像素数)')
            plt.ylabel('推荐最大文本长度')
            plt.title('图像大小与推荐最大文本长度的关系')
            plt.grid(True)
            
            # 添加图像名称标签
            for i, row in result_df.iterrows():
                plt.annotate(row['image_name'], 
                           (row['image_size'], row['recommended_text_length']),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "predicted_safe_lengths.png"))
        
        print(f"预测结果已保存到 {result_file}")
    
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预测图像能够安全嵌入的最大文本长度')
    parser.add_argument('--images', nargs='+', required=True, help='图像路径列表')
    parser.add_argument('--ratio', type=float, default=0.01, help='每像素可安全嵌入的字符数，默认为0.01')
    parser.add_argument('--output', default='watermark_predictions', help='输出目录')
    
    args = parser.parse_args()
    
    predict_safe_text_length(args.images, args.ratio, args.output) 