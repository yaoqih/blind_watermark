import os
import cv2
import numpy as np
from blind_watermark import WaterMark
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import time


def get_image_resolution(image_path: str) -> Tuple[int, int]:
    """获取图像分辨率"""
    img = Image.open(image_path)
    return img.size


def resize_image(image_path: str, target_width: int, target_height: int, output_dir: str) -> str:
    """将图像缩放到指定分辨率
    
    Args:
        image_path: 原始图像路径
        target_width: 目标宽度
        target_height: 目标高度
        output_dir: 输出目录
        
    Returns:
        缩放后的图像路径
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 读取图像
    img = Image.open(image_path)
    
    # 缩放图像
    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # 生成输出文件名
    image_name = os.path.basename(image_path)
    name, ext = os.path.splitext(image_name)
    output_name = f"{name}_{target_width}x{target_height}{ext}"
    output_path = os.path.join(output_dir, output_name)
    
    # 保存缩放后的图像
    resized_img.save(output_path)
    
    return output_path


def calculate_error_rate(original_text: str, extracted_text: str) -> float:
    """计算文本提取的错误率"""
    if not original_text or len(original_text) == 0:
        return 1.0
    
    # 计算编辑距离
    original_len = len(original_text)
    extracted_len = len(extracted_text)
    
    # 截断较长的文本以匹配较短的文本长度
    min_len = min(original_len, extracted_len)
    original_text = original_text[:min_len]
    extracted_text = extracted_text[:min_len]
    
    # 计算字符不匹配的数量
    errors = sum(a != b for a, b in zip(original_text, extracted_text))
    
    # 错误率 = 错误字符数 / 原始文本长度
    return errors / original_len


def calculate_recommended_text_length(width: int, height: int) -> int:
    """根据图像分辨率计算推荐的最大文本长度
    
    这个函数根据测试结果拟合一个线性关系，来估算给定分辨率的图像能够安全嵌入的最大文本长度。
    错误率低于10%被认为是可接受的。
    
    Args:
        width: 图像宽度
        height: 图像高度
        
    Returns:
        推荐的最大文本长度
    """
    # 计算图像像素总数
    pixel_count = width * height
    
    # 基于经验公式计算推荐长度
    # 这里使用一个简单的线性关系，可能需要根据实际测试结果进行调整
    # 假设每10000像素可以安全嵌入约100个字符
    safe_ratio = 100 / 10000  # 每像素可安全嵌入的字符数
    
    # 应用一个安全系数(0.8)，以确保推荐值偏保守
    recommended_length = int(pixel_count * safe_ratio * 0.8)
    
    return max(10, recommended_length)  # 确保至少返回10


def test_watermark(image_path: str, text_lengths: List[int], output_dir: str = "watermark_test_results") -> pd.DataFrame:
    """测试不同长度文本在特定图像上的水印嵌入和提取效果"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取图像分辨率
    width, height = get_image_resolution(image_path)
    image_name = os.path.basename(image_path)
    image_size = width * height
    
    # 计算推荐文本长度
    recommended_length = calculate_recommended_text_length(width, height)
    
    results = []
    
    for length in text_lengths:
        # 生成指定长度的测试文本
        test_text = f"测试水印文本_{length}" * (length // 10 + 1)
        test_text = test_text[:length]
        
        # 输出文件名
        embedded_file = os.path.join(output_dir, f"{image_name}_len{length}_embedded.png")
        
        try:
            # 嵌入水印
            start_time = time.time()
            bwm = WaterMark(password_img=1, password_wm=1)
            bwm.read_img(image_path)
            bwm.read_wm(test_text, mode='str')
            bwm.embed(embedded_file)
            embedding_time = time.time() - start_time
            
            len_wm = len(bwm.wm_bit)
            
            # 提取水印
            start_time = time.time()
            bwm = WaterMark(password_img=1, password_wm=1)
            wm_extract = bwm.extract(embedded_file, wm_shape=len_wm, mode='str')
            extraction_time = time.time() - start_time
            
            # 计算错误率
            error_rate = calculate_error_rate(test_text, wm_extract)
            
            results.append({
                "image_name": image_name,
                "width": width,
                "height": height,
                "image_size": image_size,
                "text_length": length,
                "watermark_bit_length": len_wm,
                "error_rate": error_rate,
                "embedding_time": embedding_time,
                "extraction_time": extraction_time,
                "recommended_length": recommended_length,
                "status": "成功" if error_rate < 0.1 else "失败"
            })
            print(f"图像 {image_name} (分辨率: {width}x{height}) 嵌入文本长度: {length}, 错误率: {error_rate:.4f}")
            
        except Exception as e:
            print(f"错误：图像 {image_name} 嵌入文本长度 {length} 失败: {str(e)}")
            results.append({
                "image_name": image_name,
                "width": width,
                "height": height,
                "image_size": image_size,
                "text_length": length,
                "watermark_bit_length": 0,
                "error_rate": 1.0,
                "embedding_time": 0,
                "extraction_time": 0,
                "recommended_length": recommended_length,
                "status": f"失败: {str(e)}"
            })
    
    return pd.DataFrame(results)


def test_resized_images(original_image_path: str, 
                       resolutions: List[Tuple[int, int]], 
                       text_lengths: List[int],
                       output_dir: str = "watermark_test_results",
                       keep_aspect_ratio: bool = False) -> pd.DataFrame:
    """对同一张图片缩放到不同分辨率后进行水印测试
    
    Args:
        original_image_path: 原始图像路径
        resolutions: 目标分辨率列表，每个元素为(宽度,高度)元组
        text_lengths: 测试文本长度列表
        output_dir: 输出目录
        keep_aspect_ratio: 是否保持原图的宽高比例
        
    Returns:
        测试结果DataFrame
    """
    # 创建临时目录存放缩放后的图像
    resized_dir = os.path.join(output_dir, "resized_images")
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    
    # 如果需要保持宽高比，获取原图比例
    if keep_aspect_ratio:
        orig_img = Image.open(original_image_path)
        orig_width, orig_height = orig_img.size
        aspect_ratio = orig_width / orig_height
        print(f"原图宽高比为: {aspect_ratio:.4f} ({orig_width}x{orig_height})")
    
    all_results = []
    
    # 对每个分辨率进行测试
    for target_width, target_height in resolutions:
        try:
            # 如果保持宽高比，则根据目标宽度计算新的高度
            if keep_aspect_ratio:
                # 使用目标宽度作为基准，计算对应的等比例高度
                adjusted_height = int(target_width / aspect_ratio)
                print(f"调整分辨率: {target_width}x{target_height} -> {target_width}x{adjusted_height}（保持宽高比）")
                target_height = adjusted_height
            
            # 缩放图像
            resized_image_path = resize_image(original_image_path, target_width, target_height, resized_dir)
            
            # 测试水印
            print(f"测试分辨率 {target_width}x{target_height} 的图像...")
            result_df = test_watermark(resized_image_path, text_lengths, output_dir)
            all_results.append(result_df)
            
        except Exception as e:
            print(f"处理分辨率 {target_width}x{target_height} 时出错: {str(e)}")
    
    # 合并所有结果
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # 保存结果到CSV
        result_file = os.path.join(output_dir, "resized_watermark_test_results.csv")
        combined_results.to_csv(result_file, index=False, encoding='utf-8')
        
        # 生成可视化结果
        visualize_resized_results(combined_results, output_dir)
        
        # 生成推荐公式
        generate_recommendation_formula(combined_results, os.path.join(output_dir, "resized"))
        
        return combined_results
    
    return pd.DataFrame()


def visualize_resized_results(results: pd.DataFrame, output_dir: str) -> None:
    """可视化不同分辨率测试结果"""
    # 为不同分辨率创建错误率与文本长度的关系图
    plt.figure(figsize=(12, 8))
    
    for resolution in results[['width', 'height']].drop_duplicates().itertuples():
        width, height = resolution.width, resolution.height
        resolution_data = results[(results['width'] == width) & (results['height'] == height)]
        
        if not resolution_data.empty:
            plt.plot(resolution_data['text_length'], resolution_data['error_rate'], 
                    marker='o', label=f"{width}x{height} ({width*height} 像素)")
    
    plt.xlabel('文本长度')
    plt.ylabel('错误率')
    plt.title('不同分辨率图像嵌入不同长度文本的错误率')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resized_error_rate_by_text_length.png"))
    
    # 分辨率与最大可用文本长度的关系图
    max_viable_length = []
    
    for resolution in results[['width', 'height']].drop_duplicates().itertuples():
        width, height = resolution.width, resolution.height
        resolution_data = results[(results['width'] == width) & (results['height'] == height)]
        
        # 找出错误率低于10%的最大文本长度
        viable_data = resolution_data[resolution_data['error_rate'] < 0.1]
        
        if not viable_data.empty:
            max_length = viable_data['text_length'].max()
        else:
            max_length = 0
        
        max_viable_length.append({
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'image_size': width * height,
            'max_viable_length': max_length
        })
    
    max_viable_df = pd.DataFrame(max_viable_length)
    max_viable_df = max_viable_df.sort_values('image_size')
    
    # 保存最大可用文本长度结果
    max_viable_df.to_csv(os.path.join(output_dir, "resized_max_viable_text_length.csv"), index=False, encoding='utf-8')
    
    # 绘制图像大小与最大可用文本长度的关系图
    plt.figure(figsize=(12, 8))
    plt.scatter(max_viable_df['image_size'], max_viable_df['max_viable_length'])
    plt.xlabel('图像大小 (像素数)')
    plt.ylabel('最大可用文本长度')
    plt.title('图像大小与最大可用文本长度的关系')
    plt.grid(True)
    
    # 添加分辨率标签
    for i, row in max_viable_df.iterrows():
        plt.annotate(row['resolution'], 
                    (row['image_size'], row['max_viable_length']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "resized_max_length_by_resolution.png"))


def batch_test_watermark(image_paths: List[str], text_lengths: List[int], output_dir: str = "watermark_test_results") -> pd.DataFrame:
    """批量测试多个图像和多个文本长度的水印效果"""
    all_results = []
    
    for image_path in image_paths:
        result_df = test_watermark(image_path, text_lengths, output_dir)
        all_results.append(result_df)
    
    # 合并所有结果
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # 保存结果到CSV
        result_file = os.path.join(output_dir, "watermark_test_results.csv")
        combined_results.to_csv(result_file, index=False, encoding='utf-8')
        
        # 生成可视化结果
        visualize_results(combined_results, output_dir)
        
        # 生成推荐公式
        generate_recommendation_formula(combined_results, output_dir)
        
        return combined_results
    
    return pd.DataFrame()


def generate_recommendation_formula(results: pd.DataFrame, output_dir: str) -> None:
    """基于测试结果生成推荐公式"""
    # 只选择成功的测试结果
    successful_results = results[results['error_rate'] < 0.1]
    
    if successful_results.empty:
        print("没有找到成功的测试结果，无法生成推荐公式")
        return
    
    # 对每个图像，找出错误率小于10%的最大文本长度
    max_lengths = []
    for image_name in successful_results['image_name'].unique():
        image_data = successful_results[successful_results['image_name'] == image_name]
        max_length = image_data['text_length'].max()
        image_size = image_data['image_size'].iloc[0]
        max_lengths.append({
            'image_name': image_name,
            'image_size': image_size,
            'max_safe_length': max_length
        })
    
    max_lengths_df = pd.DataFrame(max_lengths)
    
    # 尝试拟合一个线性关系: max_safe_length = a * image_size + b
    # 只有当有足够的数据点时才进行拟合
    if len(max_lengths_df) >= 3:
        try:
            x = max_lengths_df['image_size'].values
            y = max_lengths_df['max_safe_length'].values
            coefficients = np.polyfit(x, y, 1)
            a, b = coefficients
            
            # 创建预测函数
            def predict_safe_length(image_size):
                return max(10, int(a * image_size + b))
            
            # 计算预测值
            max_lengths_df['predicted_length'] = max_lengths_df['image_size'].apply(predict_safe_length)
            
            # 可视化拟合结果
            plt.figure(figsize=(12, 8))
            plt.scatter(max_lengths_df['image_size'], max_lengths_df['max_safe_length'], label='实际最大安全长度')
            
            # 绘制拟合线
            x_range = np.linspace(min(x), max(x), 100)
            y_pred = a * x_range + b
            plt.plot(x_range, y_pred, 'r-', label=f'拟合曲线: length = {a:.6f} * pixels + {b:.2f}')
            
            # 添加图像名称标签
            for i, row in max_lengths_df.iterrows():
                plt.annotate(row['image_name'], 
                           (row['image_size'], row['max_safe_length']),
                           textcoords="offset points",
                           xytext=(0, 10),
                           ha='center')
            
            plt.xlabel('图像大小 (像素数)')
            plt.ylabel('最大安全文本长度')
            plt.title('图像大小与最大安全文本长度的关系')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "safe_length_prediction.png"))
            
            # 将公式和预测结果保存到CSV
            formula_file = os.path.join(output_dir, "recommendation_formula.txt")
            with open(formula_file, 'w', encoding='utf-8') as f:
                f.write(f"推荐公式: 最大安全文本长度 = {a:.6f} * 图像像素数 + {b:.2f}\n")
                f.write(f"简化公式: 最大安全文本长度 ≈ {a:.6f} * 图像像素数\n")
                f.write("\n每个图像的推荐值:\n")
                for _, row in max_lengths_df.iterrows():
                    f.write(f"{row['image_name']} ({row['image_size']} 像素): 实际最大长度 = {row['max_safe_length']}, 预测长度 = {row['predicted_length']}\n")
            
            print(f"推荐公式已生成并保存到 {formula_file}")
            
            # 更新calculate_recommended_text_length函数中的系数
            print(f"建议更新推荐函数中的系数为: safe_ratio = {a:.8f}")
            
        except Exception as e:
            print(f"生成推荐公式时出错: {str(e)}")
    else:
        print("数据点不足，无法进行可靠的线性拟合")


def visualize_results(results: pd.DataFrame, output_dir: str) -> None:
    """可视化测试结果"""
    # 为每个图像创建错误率与文本长度的关系图
    plt.figure(figsize=(12, 8))
    
    for image_name in results['image_name'].unique():
        image_data = results[results['image_name'] == image_name]
        resolution = f"{image_data['width'].iloc[0]}x{image_data['height'].iloc[0]}"
        plt.plot(image_data['text_length'], image_data['error_rate'], marker='o', label=f"{image_name} ({resolution})")
    
    plt.xlabel('文本长度')
    plt.ylabel('错误率')
    plt.title('不同分辨率图像嵌入不同长度文本的错误率')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_rate_by_text_length.png"))
    
    # 分辨率与最大可用文本长度的关系图
    max_viable_length = []
    
    for image_name in results['image_name'].unique():
        image_data = results[results['image_name'] == image_name]
        # 找出错误率低于10%的最大文本长度
        viable_data = image_data[image_data['error_rate'] < 0.1]
        if not viable_data.empty:
            max_length = viable_data['text_length'].max()
        else:
            max_length = 0
        
        max_viable_length.append({
            'image_name': image_name,
            'width': image_data['width'].iloc[0],
            'height': image_data['height'].iloc[0],
            'image_size': image_data['image_size'].iloc[0],
            'max_viable_length': max_length
        })
    
    max_viable_df = pd.DataFrame(max_viable_length)
    max_viable_df = max_viable_df.sort_values('image_size')
    
    plt.figure(figsize=(12, 8))
    plt.scatter(max_viable_df['image_size'], max_viable_df['max_viable_length'])
    plt.xlabel('图像分辨率 (像素数)')
    plt.ylabel('最大可用文本长度')
    plt.title('图像分辨率与最大可用文本长度的关系')
    plt.grid(True)
    
    # 添加图像名称标签
    for i, row in max_viable_df.iterrows():
        plt.annotate(row['image_name'], 
                    (row['image_size'], row['max_viable_length']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "max_length_by_resolution.png"))
    
    # 保存最大可用文本长度结果
    max_viable_df.to_csv(os.path.join(output_dir, "max_viable_text_length.csv"), index=False, encoding='utf-8')


if __name__ == "__main__":
    # 测试不同分辨率的图像
    test_images = [
        "image_test/1.png",  # 低分辨率
        "image_test/2.png",
        "image_test/3.png",
        "image_test/20.1.png",
        "image_test/20.2.png",  # 中分辨率
        "image_test/5.1.jpg",
        "image_test/9.1.png",
        "image_test/1.1.png",  # 高分辨率
    ]
    
    # 测试不同长度的文本
    text_lengths = [100, 200, 500, 1000, 2000, 5000]
    
    # 执行批量测试
    results = batch_test_watermark(test_images, text_lengths)
    
    print("测试完成，结果已保存到 watermark_test_results 目录") 