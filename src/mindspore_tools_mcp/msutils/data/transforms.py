"""
msutils.data.transforms - 数据转换工具

提供常用的数据转换功能
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Union


class ToTensor:
    """
    转换为张量
    
    将图像转换为 MindSpore 张量格式
    
    Example:
        >>> transform = ToTensor()
        >>> tensor = transform(image)
    """
    
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            # 转换为 CHW 格式
            if image.ndim == 2:
                return image[np.newaxis, ...]
            elif image.ndim == 3:
                return image.transpose(2, 0, 1)
            return image
        return image


class Normalize:
    """
    归一化
    
    使用均值和标准差归一化图像
    
    Args:
        mean: 均值
        std: 标准差
    
    Example:
        >>> transform = Normalize(mean=[0.5], std=[0.5])
        >>> normalized = transform(image)
    """
    
    def __init__(self, mean: Union[float, List[float]], std: Union[float, List[float]]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - self.mean) / self.std
        return image


class Resize:
    """
    调整大小
    
    调整图像尺寸
    
    Args:
        size: 目标尺寸，可以是数字或 (height, width)
    
    Example:
        >>> transform = Resize(224)
        >>> resized = transform(image)
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size
    
    def __call__(self, image):
        try:
            import cv2
            if isinstance(self.size, int):
                h, w = image.shape[:2]
                if h > w:
                    new_h, new_w = self.size, int(w * self.size / h)
                else:
                    new_h, new_w = int(h * self.size / h), self.size
            else:
                new_h, new_w = self.size
            
            return cv2.resize(image, (new_w, new_h))
        except:
            return image


class CenterCrop:
    """
    中心裁剪
    
    从图像中心裁剪出指定区域
    
    Args:
        size: 裁剪尺寸
    
    Example:
        >>> transform = CenterCrop(224)
        >>> cropped = transform(image)
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size
    
    def __call__(self, image):
        h, w = image.shape[:2]
        
        if isinstance(self.size, int):
            crop_h = crop_w = self.size
        else:
            crop_h, crop_w = self.size
        
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        
        return image[top:top+crop_h, left:left+crop_w]


class RandomPerspective:
    """
    随机透视变换
    
    对图像进行透视变换
    
    Args:
        distortion_scale: 扭曲程度
        p: 应用概率
    
    Example:
        >>> transform = RandomPerspective(distortion_scale=0.2)
        >>> transformed = transform(image)
    """
    
    def __init__(self, distortion_scale: float = 0.2, p: float = 0.5):
        self.distortion_scale = distortion_scale
        self.p = p
    
    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        
        try:
            import cv2
            h, w = image.shape[:2]
            
            # 生成透视变换点
            pts1 = np.float32([
                [0, 0],
                [w, 0],
                [0, h],
                [w, h]
            ])
            
            # 随机偏移
            offset = int(self.distortion_scale * min(h, w))
            pts2 = np.float32([
                [np.random.randint(0, offset), np.random.randint(0, offset)],
                [w - np.random.randint(0, offset), np.random.randint(0, offset)],
                [np.random.randint(0, offset), h - np.random.randint(0, offset)],
                [w - np.random.randint(0, offset), h - np.random.randint(0, offset)]
            ])
            
            # 应用变换
            M = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(image, M, (w, h))
        except:
            return image


class GaussianBlur:
    """
    高斯模糊
    
    对图像进行高斯模糊
    
    Args:
        kernel_size: 卷积核大小
        sigma: 高斯标准差
        p: 应用概率
    
    Example:
        >>> transform = GaussianBlur(kernel_size=5, sigma=1.0)
        >>> blurred = transform(image)
    """
    
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, p: float = 0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        
        try:
            import cv2
            return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        except:
            return image


class RandomAffine:
    """
    随机仿射变换
    
    对图像进行随机仿射变换（旋转、缩放、平移、剪切）
    
    Args:
        degrees: 旋转角度范围
        translate: 平移范围
        scale: 缩放范围
        shear: 剪切范围
        p: 应用概率
    
    Example:
        >>> transform = RandomAffine(degrees=15, translate=(0.1, 0.1))
        >>> transformed = transform(image)
    """
    
    def __init__(
        self,
        degrees: float = 10,
        translate: Tuple[float, float] = (0.1, 0.1),
        scale: Tuple[float, float] = (0.9, 1.1),
        shear: float = 0,
        p: float = 0.5
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p
    
    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        
        try:
            import cv2
            h, w = image.shape[:2]
            
            # 旋转
            angle = np.random.uniform(-self.degrees, self.degrees)
            
            # 缩放
            scale = np.random.uniform(self.scale[0], self.scale[1])
            
            # 平移
            tx = np.random.uniform(self.translate[0], self.translate[1]) * w
            ty = np.random.uniform(self.translate[0], self.translate[1]) * h
            
            # 计算变换矩阵
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            
            return cv2.warpAffine(image, M, (w, h))
        except:
            return image


class Pad:
    """
    填充
    
    对图像边缘进行填充
    
    Args:
        padding: 填充大小
        fill: 填充值
        mode: 填充模式
    
    Example:
        >>> transform = Pad(padding=4, fill=0)
        >>> padded = transform(image)
    """
    
    def __init__(
        self,
        padding: Union[int, Tuple[int, int]],
        fill: int = 0,
        mode: str = 'constant'
    ):
        self.padding = padding
        self.fill = fill
        self.mode = mode
    
    def __call__(self, image):
        if isinstance(self.padding, int):
            pad_h = pad_w = self.padding
        else:
            pad_h, pad_w = self.padding
        
        if self.mode == 'constant':
            return np.pad(
                image,
                ((pad_h, pad_h), (pad_w, pad_w), (0, 0) if image.ndim == 3 else (0, 0)),
                mode=self.mode,
                constant_values=self.fill
            )
        return image


class Lambda:
    """
    自定义变换
    
    使用自定义函数进行变换
    
    Args:
        func: 变换函数
    
    Example:
        >>> transform = Lambda(lambda x: x / 255.0)
        >>> transformed = transform(image)
    """
    
    def __init__(self, func: Callable):
        self.func = func
    
    def __call__(self, image):
        return self.func(image)


# 导出所有类
__all__ = [
    'ToTensor',
    'Normalize',
    'Resize',
    'CenterCrop',
    'RandomPerspective',
    'GaussianBlur',
    'RandomAffine',
    'Pad',
    'Lambda'
]
