#!/usr/bin/env python3
# coding=utf-8
# 测试blind_watermark库支持PIL Image的功能

from PIL import Image
import io
from blind_watermark import WaterMark

# 初始化水印对象
bwm = WaterMark(password_wm=1, password_img=1)

# 测试文本
test_text = "这是一个测试文本水印，支持中文和English以及特殊字符!@#$%^&*()"

def test_file_to_file():
    """测试从文件到文件的水印嵌入和提取"""
    print("测试从文件到文件...")
    in_file = "image_test/1.1.png"
    out_file = "output_file_to_file.png"
    
    # 嵌入水印
    bwm.embed_text(text=test_text, filename=in_file, out_filename=out_file)
    
    # 提取水印
    extracted_text = bwm.extract_text(filename=out_file)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

def test_pil_to_bytes():
    """测试从PIL Image到图像字节数据的水印嵌入和提取"""
    print("测试从PIL Image到图像字节数据...")
    in_file = "image_test/1.1.png"
    
    # 读取图像
    pil_img = Image.open(in_file)
    
    # 嵌入水印并获取字节数据
    watermarked_bytes = bwm.embed_text(text=test_text, img=pil_img, return_bytes=True)
    
    # 从字节数据创建PIL图像并提取水印
    watermarked_pil = Image.open(io.BytesIO(watermarked_bytes))
    extracted_text = bwm.extract_text(img=watermarked_pil)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

def test_pil_to_file():
    """测试从PIL Image到文件的水印嵌入和提取"""
    print("测试从PIL Image到文件...")
    in_file = "image_test/1.1.png"
    out_file = "output_pil_to_file.png"
    
    # 读取图像
    pil_img = Image.open(in_file)
    
    # 嵌入水印
    bwm.embed_text(text=test_text, img=pil_img, out_filename=out_file)
    
    # 提取水印
    extracted_text = bwm.extract_text(filename=out_file)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

def test_file_to_bytes():
    """测试从文件到图像字节数据的水印嵌入和提取"""
    print("测试从文件到图像字节数据...")
    in_file = "image_test/1.1.png"
    
    # 嵌入水印并获取字节数据
    watermarked_bytes = bwm.embed_text(text=test_text, filename=in_file, return_bytes=True)
    
    # 从字节数据创建PIL图像并提取水印
    watermarked_pil = Image.open(io.BytesIO(watermarked_bytes))
    extracted_text = bwm.extract_text(img=watermarked_pil)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

def test_bytes_to_file():
    """测试从字节数据到文件的过程，验证元数据保留"""
    print("测试从字节数据到文件，验证元数据保留...")
    in_file = "image_test/1.1.png"
    out_file = "output_bytes_to_file.png"
    
    # 嵌入水印并获取字节数据
    watermarked_bytes = bwm.embed_text(text=test_text, filename=in_file, return_bytes=True)
    
    # 将字节数据保存到文件
    with open(out_file, "wb") as f:
        f.write(watermarked_bytes)
    
    # 从文件提取水印文本
    extracted_text = bwm.extract_text(filename=out_file)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

def test_numpy_support():
    """测试numpy数组输入输出"""
    print("测试numpy数组支持...")
    import cv2
    import numpy as np
    
    in_file = "image_test/1.1.png"
    
    # 读取图像为numpy数组
    numpy_img = cv2.imread(in_file)
    
    # 嵌入水印
    watermarked_numpy = bwm.embed_text(text=test_text, img=numpy_img)
    
    # 提取水印
    extracted_text = bwm.extract_text(img=watermarked_numpy, check_metadata=False)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

def test_bytes_metadata():
    """测试bytes输出的元数据是否被正确保留"""
    print("测试bytes输出的元数据是否被正确保留...")
    in_file = "image_test/1.1.png"
    
    # 嵌入水印并获取字节数据
    watermarked_bytes = bwm.embed_text(text=test_text, filename=in_file, return_bytes=True)
    
    # 从字节数据创建PIL图像
    watermarked_pil = Image.open(io.BytesIO(watermarked_bytes))
    
    # 检查元数据是否存在
    has_metadata = 'watermark' in watermarked_pil.info
    print(f"元数据存在: {'是' if has_metadata else '否'}")
    
    # 提取水印文本 - 首先尝试从元数据中提取
    extracted_text = bwm.extract_text(img=watermarked_pil)
    
    # 检查结果
    print(f"原始文本: {test_text}")
    print(f"提取文本: {extracted_text}")
    print(f"匹配结果: {'成功' if test_text == extracted_text else '失败'}")
    print("-" * 50)

if __name__ == "__main__":
    print("开始测试blind_watermark库的图像字节数据支持...\n")
    
    # 运行所有测试
    test_file_to_file()
    test_pil_to_bytes()
    test_pil_to_file()
    test_file_to_bytes()
    test_bytes_to_file()
    test_numpy_support()
    test_bytes_metadata()
    
    print("测试完成!") 