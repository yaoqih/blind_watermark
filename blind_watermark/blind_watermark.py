#!/usr/bin/env python3
# coding=utf-8
# @Time    : 2020/8/13
# @Author  : github.com/guofei9987
import warnings

import numpy as np
import cv2

from .bwm_core import WaterMarkCore
from .version import bw_notes
import zlib
import reedsolo
from PIL import Image, PngImagePlugin

class WaterMark:
    def __init__(self, password_wm=1, password_img=1, block_shape=(4, 4), mode='common', processes=None):

        self.bwm_core = WaterMarkCore(password_img=password_img, mode=mode, processes=processes)

        self.password_wm = password_wm

        self.wm_bit = None
        self.wm_size = 0
        self.rs_coder = reedsolo.RSCodec()

    def read_img(self, filename=None, img=None):
        if img is None:
            # 从文件读入图片
            img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            assert img is not None, "image file '{filename}' not read".format(filename=filename)

        self.bwm_core.read_img_arr(img=img)
        return img

    def read_wm(self, wm_content, mode='img'):
        assert mode in ('img', 'str', 'bit'), "mode in ('img','str','bit')"
        if mode == 'img':
            wm = cv2.imread(filename=wm_content, flags=cv2.IMREAD_GRAYSCALE)
            assert wm is not None, 'file "{filename}" not read'.format(filename=wm_content)

            # 读入图片格式的水印，并转为一维 bit 格式，抛弃灰度级别
            self.wm_bit = wm.flatten() > 128

        elif mode == 'str':
            byte = bin(int(wm_content.encode('utf-8').hex(), base=16))[2:]
            self.wm_bit = (np.array(list(byte)) == '1')
        else:
            self.wm_bit = np.array(wm_content)

        self.wm_size = self.wm_bit.size

        # 水印加密:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

        self.bwm_core.read_wm(self.wm_bit)

    def embed(self, filename=None, compression_ratio=None):
        '''
        :param filename: string
            Save the image file as filename
        :param compression_ratio: int or None
            If compression_ratio = None, do not compression,
            If compression_ratio is integer between 0 and 100, the smaller, the output file is smaller.
        :return:
        '''
        embed_img = self.bwm_core.embed()
        if filename is not None:
            if compression_ratio is None:
                cv2.imwrite(filename=filename, img=embed_img)
            elif filename.endswith('.jpg'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_JPEG_QUALITY, compression_ratio])
            elif filename.endswith('.png'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_PNG_COMPRESSION, compression_ratio])
            else:
                cv2.imwrite(filename=filename, img=embed_img)
        return embed_img

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password_wm).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

    def extract(self, filename=None, embed_img=None, wm_shape=None, out_wm_name=None, mode='img'):
        assert wm_shape is not None, 'wm_shape needed'

        if filename is not None:
            embed_img = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
            assert embed_img is not None, "{filename} not read".format(filename=filename)

        self.wm_size = np.array(wm_shape).prod()

        if mode in ('str', 'bit'):
            wm_avg = self.bwm_core.extract_with_kmeans(img=embed_img, wm_shape=wm_shape)
        else:
            wm_avg = self.bwm_core.extract(img=embed_img, wm_shape=wm_shape)

        # 解密：
        wm = self.extract_decrypt(wm_avg=wm_avg)

        # 转化为指定格式：
        if mode == 'img':
            wm = 255 * wm.reshape(wm_shape[0], wm_shape[1])
            cv2.imwrite(out_wm_name, wm)
        elif mode == 'str':
            byte = ''.join(str((i >= 0.5) * 1) for i in wm)
            wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode('utf-8', errors='replace')

        return wm

    def encode_int32_array(int_array):
        # 创建一个空列表存储所有整数的二进制表示
        all_bits = []
        
        for num in int_array:
            # 将每个int32转换为32位二进制表示，去掉'0b'前缀
            binary = bin(num & 0xFFFFFFFF)[2:].zfill(32)
            # 将二进制字符串转换为单个字符的列表并添加到总列表
            all_bits.extend(list(binary))
        
        # 转换为numpy数组，并将'1'字符转换为True，其他转换为False
        bit_array = (np.array(all_bits) == '1')
    
        return bit_array
    
    def decode_to_int32_array(bit_array):
        # 确保比特数组长度是32的倍数
        if len(bit_array) % 32 != 0:
            raise ValueError("比特数组长度必须是32的倍数")
        
        # 将布尔数组转换为'0'和'1'字符
        bit_chars = ['1' if bit else '0' for bit in bit_array]
        
        # 创建一个空列表存储恢复的整数
        int_array = []
        
        # 每32位转换为一个整数
        for i in range(0, len(bit_chars), 32):
            # 获取32位的二进制字符串
            binary = ''.join(bit_chars[i:i+32])
            # 将二进制字符串转换为整数
            num = int(binary, 2)
            # 处理有符号整数（如果最高位是1，则为负数）
            if binary[0] == '1':
                num = num - (1 << 32)
            int_array.append(num)
        
        # 截取到指定长度
        return int_array[0]
    
    def embed_text(self, text, filename=None, out_filename=None, use_metadata=True, img=None, return_bytes=False, only_meta_data=False):
        """
        在图像中嵌入文本水印
        
        参数:
        text: str - 要嵌入的文本内容
        filename: str - 输入图像文件路径（如果img为None）
        out_filename: str - 输出图像文件路径（如果需要保存到文件）
        use_metadata: bool - 是否在图像元数据中也存储水印信息
        img: PIL.Image 或 numpy.ndarray - 输入图像对象（优先于filename）
        return_bytes: bool - 是否返回图像字节数据(True)或numpy数组(False)
        only_meta_data: bool - 如果为True，则只嵌入元数据而不嵌入像素水印
        
        返回:
        bytes或numpy.ndarray - 根据return_bytes参数返回不同类型的嵌入水印后的图像
        """

        
        # 处理输入图像
        if img is not None:
            if isinstance(img, Image.Image):
                # 如果输入是PIL.Image，转换为numpy数组
                numpy_img = np.array(img.convert('RGB'))
                # 调整通道顺序从RGB到BGR (PIL->OpenCV)
                numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
            else:
                # 如果已经是numpy数组，则直接使用
                numpy_img = img
        elif filename is not None:
            # 从文件读取图像
            numpy_img = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError("必须提供img或filename中的一个")
        
        # 确保图像已正确加载
        assert numpy_img is not None, "无法读取图像"
        
        # 1. 将文本转换为字节
        text_bytes = text.encode('utf-8')
        
        # 2. 使用zlib进行压缩
        compressed_data = zlib.compress(text_bytes, level=9)
        
        embed_img = numpy_img # 默认使用原图，除非进行像素嵌入
        
        if not only_meta_data:
            # 计算安全水印长度
            pixel_num = numpy_img.shape[0] * numpy_img.shape[1]
            safe_bit_num = int((pixel_num * 0.0025 + 190) * 0.9)
            
            # 3. 应用Reed-Solomon编码添加纠错能力
            rs_encoded = self.rs_coder.encode(compressed_data)
            
            # 4. 转换为位数组
            bit_string = ''
            for byte in rs_encoded:
                # 将每个字节转换为8位二进制
                bits = format(byte, '08b')
                bit_string += bits
            
            # 5. 转换为NumPy数组
            wm_bit = (np.array(list(bit_string)) == '1')
            if len(wm_bit) > safe_bit_num - 32:
                raise ValueError(f"水印长度超过安全阈值，请调整水印长度,当前水印长度为：{len(wm_bit)}，安全阈值为：{safe_bit_num-32}")
            
            wm_all = wm_bit.tolist() + [0] * (safe_bit_num - len(wm_bit) - 32) + WaterMark.encode_int32_array([len(wm_bit)]).tolist()
            
            # 使用已读取的图像数组而不是再次读取文件
            self.bwm_core.read_img_arr(img=numpy_img)
            self.read_wm(wm_all, mode='bit')
            
            # 嵌入水印到像素
            embed_img = self.embed()
        
        # 准备元数据 (即使only_meta_data=True也要准备)
        import base64
        metadata_text = base64.b64encode(compressed_data).decode('ascii')
        
        # 根据需要返回带元数据的字节数据或保存到文件
        if use_metadata and (return_bytes or out_filename is not None):
            import io
            # 确保图像数据类型正确 (使用 embed_img，它可能是原图或修改后的图)
            embed_img_uint8 = np.clip(embed_img, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(cv2.cvtColor(embed_img_uint8, cv2.COLOR_BGR2RGB))
            
            # 添加元数据
            pil_image.info['watermark'] = metadata_text
            
            # 确定图像格式
            if out_filename is not None:
                img_format = out_filename.split('.')[-1].upper()
                if img_format == 'JPG':
                    img_format = 'JPEG'
            else:
                img_format = 'PNG'  # 默认格式
            
            # 保存到内存或文件
            byte_io = io.BytesIO()
            
            # 针对不同格式设置保存参数
            if img_format == 'JPEG':
                pil_image.save(byte_io, format=img_format, quality=100)
                
                # 如果需要保存到文件，且需要元数据
                if out_filename is not None and use_metadata:
                    pil_image.save(out_filename, quality=100)
                    
                    # 尝试使用piexif添加EXIF元数据
                    try:
                        import piexif
                        import json
                        
                        # 读取EXIF数据
                        try:
                            exif_dict = piexif.load(out_filename)
                        except:
                            exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}
                        
                        # 将水印信息添加到用户注释字段
                        exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
                            json.dumps({"watermark": metadata_text}),
                            encoding="unicode"
                        )
                        
                        # 保存修改后的EXIF数据
                        exif_bytes = piexif.dump(exif_dict)
                        piexif.insert(exif_bytes, out_filename)
                        print(f"水印已同时嵌入图像的像素和EXIF元数据中" if not only_meta_data else f"水印已嵌入图像的EXIF元数据中")
                    except ImportError:
                        print("警告：未安装piexif库，无法在JPEG中添加元数据水印，仅保留像素水印")
                        print("安装方法：pip install piexif")
            else:
                # 对于PNG和其他支持标准元数据的格式
                # Create PngInfo object to explicitly pass metadata
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text('watermark', metadata_text)
                
                # Save to memory with explicit pnginfo
                pil_image.save(byte_io, format=img_format, pnginfo=png_info)
                
                # 如果需要保存到文件
                if out_filename is not None:
                    # Save to file with explicit pnginfo
                    pil_image.save(out_filename, pnginfo=png_info)
                    if use_metadata:
                         print(f"水印已同时嵌入图像的像素和元数据中" if not only_meta_data else f"水印已嵌入图像的元数据中")
                    
            # 如果需要返回字节，获取字节数据并返回
            if return_bytes:
                return byte_io.getvalue()
        
        # 如果仅嵌入元数据，但不需要返回字节或保存文件，则原始numpy数组可能已经满足要求
        # 但为了统一，如果use_metadata为False，或者既不返回字节也不保存文件，则返回处理过的embed_img
        # 如果use_metadata为True且进行了保存或需要字节返回，则之前的if块已经处理并返回
        # 此处返回适用于不使用元数据或仅在内存中操作的场景
        return embed_img
    
    def extract_text(self, filename=None, check_metadata=True, img=None):
        """
        从图像中提取文本水印
        
        参数:
        filename: str - 输入图像文件路径（如果img为None）
        check_metadata: bool - 是否尝试从图像元数据中提取水印
        img: PIL.Image 或 numpy.ndarray - 输入图像对象（优先于filename）
        
        返回:
        str - 提取的文本水印，如果提取失败则返回None
        """
        from PIL import Image
        import numpy as np
        
        # 首先尝试从元数据中提取水印（仅当提供文件名且check_metadata为True）
        if filename is not None and check_metadata:
            try:
                # 根据文件类型选择不同的元数据提取方法
                if filename.lower().endswith(('.png', '.tiff', '.tif')):
                    import base64
                    
                    # 尝试打开图像并检查是否有水印元数据
                    with Image.open(filename) as pil_img:
                        # print(f"DEBUG: After PNG open in extract_text, pil_img.info: {pil_img.info}") # Removed debug print
                        if 'watermark' in pil_img.info:
                            # 从元数据中提取并解码水印
                            metadata_text = pil_img.info['watermark']
                            compressed_data = base64.b64decode(metadata_text)
                            try:
                                # 尝试解压缩
                                decompressed = zlib.decompress(compressed_data)
                                return decompressed.decode('utf-8')
                            except zlib.error:
                                print("元数据中的水印解压缩失败，尝试从像素中提取...")
                
                elif filename.lower().endswith(('.jpg', '.jpeg')):
                    # 处理JPEG元数据
                    try:
                        import piexif
                        import json
                        import base64
                        
                        # 读取EXIF数据
                        exif_dict = piexif.load(filename)
                        
                        # 检查用户注释字段中是否有水印
                        if piexif.ExifIFD.UserComment in exif_dict["Exif"]:
                            user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
                            # 尝试解析JSON
                            try:
                                comment_data = json.loads(user_comment)
                                if "watermark" in comment_data:
                                    # 获取并解码水印数据
                                    metadata_text = comment_data["watermark"]
                                    compressed_data = base64.b64decode(metadata_text)
                                    # 尝试解压缩
                                    try:
                                        decompressed = zlib.decompress(compressed_data)
                                        return decompressed.decode('utf-8')
                                    except zlib.error:
                                        print("EXIF元数据中的水印解压缩失败，尝试从像素中提取...")
                            except json.JSONDecodeError:
                                print("EXIF元数据解析失败，尝试从像素中提取...")
                    except ImportError:
                        print("未安装piexif库，无法从JPEG中提取元数据水印，尝试从像素中提取...")
            except Exception as e:
                print(f"从元数据中提取水印失败: {e}，尝试从像素中提取...")
        
        # 如果元数据中没有水印或提取失败，则从像素中提取
        # 处理输入图像
        if img is not None:
            if isinstance(img, Image.Image):
                # 如果输入是PIL.Image，转换为numpy数组
                numpy_img = np.array(img.convert('RGB'))
                # 调整通道顺序从RGB到BGR (PIL->OpenCV)
                numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
                
                # 计算安全位数
                pixel_num = numpy_img.shape[0] * numpy_img.shape[1]
                safe_bit_num = int((pixel_num * 0.0025 + 190) * 0.9)
                
                # 使用类的标准API提取水印
                self.bwm_core.read_img_arr(img=numpy_img)
                wm_bit = self.extract(embed_img=numpy_img, wm_shape=safe_bit_num, mode='bit')
            else:
                # 如果已经是numpy数组，则直接使用
                pixel_num = img.shape[0] * img.shape[1]
                safe_bit_num = int((pixel_num * 0.0025 + 190) * 0.9)
                
                # 使用类的标准API提取水印
                wm_bit = self.extract(embed_img=img, wm_shape=safe_bit_num, mode='bit')
        elif filename is not None:
            # 从文件提取水印
            img_data = cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)
            pixel_num = img_data.shape[0] * img_data.shape[1]
            safe_bit_num = int((pixel_num * 0.0025 + 190) * 0.9)
            wm_bit = self.extract(filename=filename, wm_shape=safe_bit_num, mode='bit')
        else:
            raise ValueError("必须提供img或filename中的一个")
        
        # 解析水印长度
        wm_num = WaterMark.decode_to_int32_array(wm_bit[-32:])
        wm_bit = wm_bit[:wm_num]
        bit_string = ''.join(['1' if bit else '0' for bit in wm_bit])
        
        # 2. 将位字符串转换为字节数组
        bytes_data = bytearray()
        for i in range(0, len(bit_string), 8):
            if i + 8 <= len(bit_string):
                byte = int(bit_string[i:i+8], 2)
                bytes_data.append(byte)
        
        # 3. 应用Reed-Solomon解码（带纠错）
        try:
            decoded_data, corrected, error_pos = self.rs_coder.decode(bytes_data)
            num_errors = len(corrected)
            if num_errors > 0:
                print(f"已修正 {num_errors} 个错误")
        except reedsolo.ReedSolomonError as e:
            print(f"Reed-Solomon解码失败: {e}")
            return None
        
        # 4. 解压缩数据
        try:
            decompressed = zlib.decompress(decoded_data)
            
        # 5. 转换回字符串
            return decompressed.decode('utf-8')
        except zlib.error:
            print("解压缩失败，数据可能损坏")
            return None

