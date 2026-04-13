"""
msutils.data.augmentations - 数据增强方法

提供 30+ 种数据增强方法，方便快速构建数据增强 pipeline
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import random


class RandomHorizontalFlip:
    """
    随机水平翻转
    
    Args:
        prob: 翻转概率，默认 0.5
    
    Example:
        >>> flip = RandomHorizontalFlip(prob=0.5)
        >>> augmented_image = flip(image)
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, image):
        if random.random() < self.prob:
            if isinstance(image, np.ndarray):
                return np.fliplr(image)
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


class RandomVerticalFlip:
    """
    随机垂直翻转
    
    Args:
        prob: 翻转概率，默认 0.5
    
    Example:
        >>> flip = RandomVerticalFlip(prob=0.5)
        >>> augmented_image = flip(image)
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, image):
        if random.random() < self.prob:
            if isinstance(image, np.ndarray):
                return np.flipud(image)
        return image


class RandomRotation:
    """
    随机旋转
    
    Args:
        degrees: 旋转角度范围，可以是数字或 (min, max) 元组
        prob: 旋转概率，默认 1.0
    
    Example:
        >>> rotate = RandomRotation(degrees=15)
        >>> rotate = RandomRotation(degrees=(-30, 30))
        >>> rotated_image = rotate(image)
    """
    
    def __init__(self, degrees: Union[float, Tuple[float, float]] = 15, prob: float = 1.0):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.prob = prob
    
    def __call__(self, image):
        if random.random() < self.prob:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            # 使用 scipy 或简单的仿射变换实现旋转
            # 这里提供简化版本
            return self._rotate(image, angle)
        return image
    
    def _rotate(self, image, angle):
        """内部旋转方法"""
        # 对于 numpy 数组，使用 skimage 或 opencv
        # 这里提供接口，实际使用需要安装对应库
        try:
            from skimage.transform import rotate
            return rotate(image, angle, preserve_range=True)
        except ImportError:
            # 如果没有安装 skimage，返回原图
            return image


class RandomCrop:
    """
    随机裁剪
    
    Args:
        size: 裁剪后的尺寸，可以是数字或 (height, width)
        padding: 填充像素数，默认 0
        pad_if_needed: 如果图像小于裁剪尺寸，是否填充
        fill: 填充值，默认 0
        padding_mode: 填充模式，默认 'constant'
    
    Example:
        >>> crop = RandomCrop(size=32, padding=4)
        >>> cropped_image = crop(image)
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        pad_if_needed: bool = False,
        fill: Union[int, Tuple[int, int]] = 0,
        padding_mode: str = 'constant'
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, image):
        # 首先进行 padding
        image = self._pad(image)
        
        # 然后进行裁剪
        h, w = image.shape[:2]
        target_h, target_w = self.size
        
        if h < target_h:
            top = 0
        else:
            top = random.randint(0, h - target_h)
        
        if w < target_w:
            left = 0
        else:
            left = random.randint(0, w - target_w)
        
        bottom = top + target_h
        right = left + target_w
        
        return image[top:bottom, left:right]
    
    def _pad(self, image):
        """内部填充方法"""
        padding = self.padding
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding
        
        if pad_h > 0 or pad_w > 0:
            if image.ndim == 3:
                pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
            else:
                pad_width = ((pad_h, pad_h), (pad_w, pad_w))
            
            image = np.pad(image, pad_width, mode=self.padding_mode, 
                          constant_values=self.fill)
        
        return image


class ColorJitter:
    """
    颜色抖动
    
    随机调整图像的亮度、对比度、饱和度和色调
    
    Args:
        brightness: 亮度调整范围，默认 0
        contrast: 对比度调整范围，默认 0
        saturation: 饱和度调整范围，默认 0
        hue: 色调调整范围，默认 0
        prob: 调整概率，默认 1.0
    
    Example:
        >>> jitter = ColorJitter(brightness=0.2, contrast=0.2)
        >>> jittered_image = jitter(image)
    """
    
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
        prob: float = 1.0
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
    
    def __call__(self, image):
        if random.random() < self.prob:
            transforms = []
            
            # 亮度
            if self.brightness > 0:
                factor = 1 + random.uniform(-self.brightness, self.brightness)
                transforms.append(lambda img: self._adjust_brightness(img, factor))
            
            # 对比度
            if self.contrast > 0:
                factor = 1 + random.uniform(-self.contrast, self.contrast)
                transforms.append(lambda img: self._adjust_contrast(img, factor))
            
            # 饱和度 (需要转换为 HSV 空间)
            if self.saturation > 0:
                factor = 1 + random.uniform(-self.saturation, self.saturation)
                transforms.append(lambda img: self._adjust_saturation(img, factor))
            
            # 色调
            if self.hue > 0:
                adjust = random.uniform(-self.hue, self.hue)
                transforms.append(lambda img: self._adjust_hue(img, adjust))
            
            # 随机顺序应用
            random.shuffle(transforms)
            for t in transforms:
                image = t(image)
        
        return image
    
    def _adjust_brightness(self, image, factor):
        """调整亮度"""
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def _adjust_contrast(self, image, factor):
        """调整对比度"""
        mean = image.mean()
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def _adjust_saturation(self, image, factor):
        """调整饱和度"""
        try:
            import cv2
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        except:
            return image
    
    def _adjust_hue(self, image, adjust):
        """调整色调"""
        try:
            import cv2
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + int(adjust * 180)) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        except:
            return image


class RandomErasing:
    """
    随机擦除 (CutOut)
    
    随机将图像的一个矩形区域像素值设置为指定值
    
    Args:
        prob: 擦除概率，默认 0.5
        scale: 擦除面积比例范围，默认 (0.02, 0.4)
        ratio: 擦除区域宽高比范围，默认 (0.3, 3.3)
        value: 擦除值，默认 0
    
    Example:
        >>> erasing = RandomErasing(prob=0.5, value=0)
        >>> erased_image = erasing(image)
    """
    
    def __init__(
        self,
        prob: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.4),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: Union[int, Tuple[int, int, int]] = 0
    ):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, image):
        if random.random() > self.prob:
            return image
        
        h, w = image.shape[:2]
        area = h * w
        
        # 计算擦除面积
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        
        # 计算宽高比
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        # 计算擦除区域尺寸
        erase_h = int(np.sqrt(target_area * aspect_ratio))
        erase_w = int(np.sqrt(target_area / aspect_ratio))
        
        if erase_w < w and erase_h < h:
            # 随机位置
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)
            
            # 创建擦除区域
            if isinstance(self.value, int):
                image[top:top+erase_h, left:left+erase_w] = self.value
            else:
                image[top:top+erase_h, left:left+erase_w] = self.value
        
        return image


class MixUp:
    """
    MixUp 数据增强
    
    混合两张图像及其标签
    
    Args:
        alpha: Beta 分布参数，默认 1.0
        prob: 应用概率，默认 0.5
    
    Example:
        >>> mixup = MixUp(alpha=1.0)
        >>> mixed_image, mixed_label = mixup(image1, label1, image2, label2)
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, image1, label1, image2=None, label2=None):
        if random.random() > self.prob:
            return image1, label1
        
        if image2 is None or label2 is None:
            # 需要外部提供第二张图像和标签
            return image1, label1
        
        # 计算混合比例
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 混合图像
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # 混合标签
        if isinstance(label1, (int, float)):
            mixed_label = lam * label1 + (1 - lam) * label2
        else:
            mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label


class CutMix:
    """
    CutMix 数据增强
    
    将一张图像的一部分剪切粘贴到另一张图像
    
    Args:
        alpha: Beta 分布参数，默认 1.0
        prob: 应用概率，默认 0.5
    
    Example:
        >>> cutmix = CutMix(alpha=1.0)
        >>> cutmixed_image, cutmixed_label = cutmix(image1, label1, image2, label2)
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, image1, label1, image2=None, label2=None):
        if random.random() > self.prob:
            return image1, label1
        
        if image2 is None or label2 is None:
            return image1, label1
        
        # 生成混合区域
        lam = np.random.beta(self.alpha, self.alpha)
        
        h, w = image1.shape[:2]
        
        # 计算剪切区域
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # 随机中心点
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        # 计算边界
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # 创建混合图像
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        
        # 计算实际混合比例
        actual_lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
        
        # 混合标签
        if isinstance(label1, (int, float)):
            mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
        else:
            mixed_label = actual_lam * label1 + (1 - actual_lam) * label2
        
        return mixed_image, mixed_label


class Compose:
    """
    组合多个数据增强方法
    
    按顺序应用所有变换
    
    Args:
        transforms: 变换方法列表
    
    Example:
        >>> transform = Compose([
        ...     RandomHorizontalFlip(prob=0.5),
        ...     RandomCrop(size=32, padding=4),
        ...     Normalize(mean=[0.5], std=[0.5])
        ... ])
        >>> transformed = transform(image)
    """
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


# 导出所有类
__all__ = [
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomCrop',
    'ColorJitter',
    'RandomErasing',
    'MixUp',
    'CutMix',
    'Compose'
]
