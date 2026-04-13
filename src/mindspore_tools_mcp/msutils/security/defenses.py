"""
msutils.security.defenses - 防御方法

实现多种对抗防御策略
"""

import numpy as np
from typing import Optional


class AdversarialTraining:
    """
    对抗训练防御
    
    在训练数据中加入对抗样本，提高模型鲁棒性
    
    Args:
        model: 待防御模型
        attack: 对抗攻击方法
        epsilon: 扰动大小
        alpha: PGD 步长
        steps: PGD 迭代步数
    
    Example:
        >>> from msutils.security import PGD
        >>> attack = PGD(model, epsilon=0.03)
        >>> defender = AdversarialTraining(model, attack)
        >>> defender.train(train_dataset, epochs=10)
    """
    
    def __init__(
        self,
        model,
        attack=None,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        steps: int = 10
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.attack = attack
    
    def train(self, train_dataset, epochs: int = 10, **kwargs):
        """
        对抗训练
        
        Args:
            train_dataset: 训练数据集
            epochs: 训练轮数
        """
        print(f"Starting adversarial training for {epochs} epochs...")
        
        # 导入攻击方法
        if self.attack is None:
            from msutils.security.attacks import PGD
            self.attack = PGD(self.model, epsilon=self.epsilon, 
                            alpha=self.alpha, steps=self.steps)
        
        # 训练循环
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_dataset:
                # 获取数据
                images, labels = batch
                
                # 生成对抗样本
                adversarial_images = self.attack.generate(images, labels)
                
                # 混合训练（原始 + 对抗）
                mixed_images = 0.5 * images + 0.5 * adversarial_images
                
                # 训练步骤
                loss = self._train_step(mixed_images, labels)
                total_loss += loss
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    def _train_step(self, images, labels):
        """单步训练"""
        # 实际实现需要前向和反向传播
        return 0.0


class InputTransformation:
    """
    输入变换防御
    
    通过对输入进行随机变换来防御对抗攻击
    
    Args:
        model: 待防御模型
        transforms: 变换列表
        prob: 变换概率
    
    Example:
        >>> from msutils.data import RandomHorizontalFlip, RandomCrop
        >>> transforms = [RandomHorizontalFlip(), RandomCrop()]
        >>> defender = InputTransformation(model, transforms)
        >>> output = defender.predict(images)
    """
    
    def __init__(self, model, transforms: list = None, prob: float = 0.5):
        self.model = model
        self.transforms = transforms or []
        self.prob = prob
    
    def predict(self, images, **kwargs):
        """使用变换后的输入进行预测"""
        import random
        
        transformed = images.copy()
        
        for transform in self.transforms:
            if random.random() < self.prob:
                transformed = transform(transformed)
        
        return self.model(transformed)
    
    def _apply_transforms(self, images):
        """应用所有变换"""
        for transform in self.transforms:
            images = transform(images)
        return images


class Randomization:
    """
    随机化防御
    
    通过随机化输入来防御对抗攻击
    
    Args:
        model: 待防御模型
        padding: 随机填充大小
        resize_ratio: 随机缩放比例范围
    
    Example:
        >>> defender = Randomization(model, padding=3)
        >>> output = defender.predict(images)
    """
    
    def __init__(
        self,
        model,
        padding: int = 3,
        resize_ratio: tuple = (0.9, 1.1)
    ):
        self.model = model
        self.padding = padding
        self.resize_ratio = resize_ratio
    
    def predict(self, images, **kwargs):
        """使用随机化后的输入进行预测"""
        transformed = self._random_transform(images)
        return self.model(transformed)
    
    def _random_transform(self, images):
        """应用随机变换"""
        import random
        
        # 随机填充
        if self.padding > 0:
            padded = np.pad(images, 
                          ((0, 0), (self.padding, self.padding), 
                           (self.padding, self.padding), (0, 0)),
                          mode='constant')
            
            # 随机裁剪回原始大小
            h, w = images.shape[1:3]
            start_h = random.randint(0, 2 * self.padding)
            start_w = random.randint(0, 2 * self.padding)
            
            images = padded[:, start_h:start_h+h, start_w:start_w+w, :]
        
        # 随机缩放（简化版本）
        # 实际需要更多处理
        
        return images


class GaussianNoise:
    """
    高斯噪声防御
    
    在输入添加高斯噪声来防御对抗攻击
    
    Args:
        model: 待防御模型
        std: 高斯噪声标准差
    
    Example:
        >>> defender = GaussianNoise(model, std=0.01)
        >>> output = defender.predict(images)
    """
    
    def __init__(self, model, std: float = 0.01):
        self.model = model
        self.std = std
    
    def predict(self, images, **kwargs):
        """添加噪声后预测"""
        noise = np.random.normal(0, self.std, images.shape)
        noisy_images = images + noise
        noisy_images = np.clip(noisy_images, 0, 1)
        return self.model(noisy_images)


# 导出所有防御方法
__all__ = [
    'AdversarialTraining',
    'InputTransformation',
    'Randomization',
    'GaussianNoise'
]
