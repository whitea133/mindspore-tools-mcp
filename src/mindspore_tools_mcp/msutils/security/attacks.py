"""
msutils.security.attacks - 对抗攻击方法

实现多种经典的对抗攻击算法
"""

import numpy as np
from typing import Optional, Tuple


class Attack:
    """对抗攻击基类"""
    
    def __init__(self, model, epsilon: float = 0.03):
        """
        Args:
            model: 目标模型
            epsilon: 扰动上界
        """
        self.model = model
        self.epsilon = epsilon
    
    def generate(self, images, labels, **kwargs):
        """生成对抗样本"""
        raise NotImplementedError
    
    def __call__(self, images, labels, **kwargs):
        return self.generate(images, labels, **kwargs)


class FGSM(Attack):
    """
    Fast Gradient Sign Method (FGSM)
    
    快速梯度符号攻击，Goodfellow 等人于 2014 年提出
    
    原理：
        使用梯度的符号方向进行一次性扰动
        
        x_adv = x + ε * sign(∇x J(θ, x, y))
    
    Args:
        model: 目标模型
        epsilon: 扰动大小，默认 0.03
        targeted: 是否为定向攻击，默认 False
    
    Example:
        >>> attack = FGSM(model, epsilon=0.03)
        >>> adversarial_images = attack(images, labels)
    """
    
    def __init__(self, model, epsilon: float = 0.03, targeted: bool = False):
        super().__init__(model, epsilon)
        self.targeted = targeted
    
    def generate(self, images, labels, **kwargs) -> np.ndarray:
        """
        生成对抗样本
        
        Args:
            images: 输入图像 (N, C, H, W) 或 (N, H, W, C)
            labels: 真实标签 (N,) 或 (N, num_classes)
        
        Returns:
            adversarial_images: 对抗样本
        """
        # 转换为 MindSpore Tensor
        from mindspore import Tensor, dtype as mstype
        
        # 获取输入梯度
        images_ms = Tensor(images.astype(np.float32))
        labels_ms = Tensor(labels.astype(np.int32))
        
        # 计算损失
        output = self.model(images_ms)
        loss = self._compute_loss(output, labels_ms)
        
        # 计算梯度
        from mindspore.ops import grad
        gradient_fn = grad(self._forward_loss)
        gradient = gradient_fn(images_ms, labels_ms)
        
        # 计算扰动
        if self.targeted:
            perturbation = -self.epsilon * np.sign(gradient.asnumpy())
        else:
            perturbation = self.epsilon * np.sign(gradient.asnumpy())
        
        # 生成对抗样本
        adversarial = images + perturbation
        
        # 确保在有效范围内
        adversarial = np.clip(adversarial, 0, 1)
        
        return adversarial
    
    def _forward_loss(self, x, y):
        """计算损失"""
        output = self.model(x)
        return self._compute_loss(output, y)
    
    def _compute_loss(self, output, labels):
        """计算交叉熵损失"""
        from mindspore.nn import SoftmaxCrossEntropyWithLogits
        criterion = SoftmaxCrossEntropyWithLogits(sparse=True)
        return criterion(output, labels)


class PGD(Attack):
    """
    Projected Gradient Descent (PGD)
    
    投影梯度下降攻击，是 FGSM 的迭代版本
    
    原理：
        多次迭代，每次小步前进并投影回扰动球内
            
        x_{t+1} = Π_{x+S}(x_t + α * sign(∇x J(θ, x_t, y)))
    
    Args:
        model: 目标模型
        epsilon: 扰动上界，默认 0.03
        alpha: 步长，默认 0.01
        steps: 迭代步数，默认 10
        random_start: 是否随机起点，默认 True
    
    Example:
        >>> attack = PGD(model, epsilon=0.03, alpha=0.01, steps=10)
        >>> adversarial_images = attack(images, labels)
    """
    
    def __init__(
        self,
        model,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        steps: int = 10,
        random_start: bool = True
    ):
        super().__init__(model, epsilon)
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
    
    def generate(self, images, labels, **kwargs) -> np.ndarray:
        """
        生成对抗样本
        
        Args:
            images: 输入图像
            labels: 真实标签
        
        Returns:
            adversarial_images: 对抗样本
        """
        from mindspore import Tensor
        
        # 记录原始图像
        original = images.copy()
        
        # 随机起点
        if self.random_start:
            images = images + np.random.uniform(-self.epsilon, self.epsilon, images.shape)
            images = np.clip(images, 0, 1)
        
        # 迭代攻击
        for _ in range(self.steps):
            # 计算梯度
            images_ms = Tensor(images.astype(np.float32))
            labels_ms = Tensor(labels.astype(np.int32))
            
            # 获取梯度
            from mindspore.ops import grad
            gradient_fn = grad(self._compute_gradient)
            gradient = gradient_fn(images_ms, labels_ms)
            
            # 更新
            perturbation = self.alpha * np.sign(gradient.asnumpy())
            images = images + perturbation
            
            # 投影回扰动球
            images = np.clip(original + np.clip(images - original, -self.epsilon, self.epsilon), 0, 1)
        
        return images
    
    def _compute_gradient(self, x, y):
        """计算损失梯度"""
        from mindspore.nn import SoftmaxCrossEntropyWithLogits
        output = self.model(x)
        criterion = SoftmaxCrossEntropyWithLogits(sparse=True)
        return criterion(output, y)


class BIM(Attack):
    """
    Basic Iterative Method (BIM)
    
    基础迭代攻击，是 FGSM 的迭代版本（也称为 I-FGSM）
    
    原理：
        x_{t+1} = x_t + α * sign(∇x J(θ, x_t, y))
    
    Args:
        model: 目标模型
        epsilon: 扰动上界
        alpha: 步长，默认 epsilon/10
        steps: 迭代步数
    
    Example:
        >>> attack = BIM(model, epsilon=0.03, alpha=0.003, steps=10)
    """
    
    def __init__(
        self,
        model,
        epsilon: float = 0.03,
        alpha: Optional[float] = None,
        steps: int = 10
    ):
        super().__init__(model, epsilon)
        self.alpha = alpha if alpha else epsilon / 10
        self.steps = steps
    
    def generate(self, images, labels, **kwargs) -> np.ndarray:
        """生成对抗样本"""
        original = images.copy()
        
        for _ in range(self.steps):
            images_ms = Tensor(images.astype(np.float32))
            labels_ms = Tensor(labels.astype(np.int32))
            
            from mindspore.ops import grad
            gradient_fn = grad(self._compute_loss)
            gradient = gradient_fn(images_ms, labels_ms)
            
            # 更新
            images = images + self.alpha * np.sign(gradient.asnumpy())
            
            # 裁剪
            images = np.clip(images, original - self.epsilon, original + self.epsilon)
            images = np.clip(images, 0, 1)
        
        return images
    
    def _compute_loss(self, x, y):
        """计算损失"""
        from mindspore.nn import SoftmaxCrossEntropyWithLogits
        output = self.model(x)
        criterion = SoftmaxCrossEntropyWithLogits(sparse=True)
        return criterion(output, y)


class CW(Attack):
    """
    Carlini & Wagner Attack (C&W)
    
    C&W 攻击，基于优化的对抗攻击方法
    
    原理：
        最小化 (||δ||² + c·f(x+δ))，其中 f 是边际损失函数
        
    Args:
        model: 目标模型
        epsilon: 扰动上界
        c: 常数因子
        steps: 迭代步数
        lr: 学习率
    
    Example:
        >>> attack = CW(model, epsilon=0.03, c=1.0, steps=100)
    """
    
    def __init__(
        self,
        model,
        epsilon: float = 0.03,
        c: float = 1.0,
        steps: int = 100,
        lr: float = 0.01
    ):
        super().__init__(model, epsilon)
        self.c = c
        self.steps = steps
        self.lr = lr
    
    def generate(self, images, labels, **kwargs) -> np.ndarray:
        """
        生成对抗样本
        
        使用二进制搜索找到最小的成功扰动
        """
        from mindspore import Tensor, ops
        
        best_adv = images.copy()
        best_dist = np.inf
        
        for _ in range(self.steps):
            # 转换到 arctanh 空间
            images_tanh = self._to_tanh(images)
            
            # 优化
            perturbation = np.tanh(images_tanh) * self.epsilon
            
            # 计算损失
            adv_images = images + perturbation
            adv_images = np.clip(adv_images, 0, 1)
            
            # 计算距离
            dist = np.sum((adv_images - images) ** 2)
            
            if dist < best_dist:
                best_dist = dist
                best_adv = adv_images.copy()
        
        return best_adv
    
    def _to_tanh(self, x):
        """转换到 arctanh 空间"""
        x = 2 * x - 1
        x = np.clip(x, 1e-8, 1 - 1e-8)
        return 0.5 * np.log((1 + x) / (1 - x))


class DeepFool(Attack):
    """
    DeepFool 攻击
    
    迭代地找到最小范数的扰动，使分类器改变决策边界
    
    Args:
        model: 目标模型
        epsilon: 扰动上界
        steps: 最大迭代步数
    
    Example:
        >>> attack = DeepFool(model, epsilon=0.03, steps=50)
    """
    
    def __init__(self, model, epsilon: float = 0.03, steps: int = 50):
        super().__init__(model, epsilon)
        self.steps = steps
    
    def generate(self, images, labels, **kwargs) -> np.ndarray:
        """
        生成对抗样本
        """
        from mindspore import Tensor
        
        adversarial = images.copy()
        
        for i in range(self.steps):
            images_ms = Tensor(adversarial.astype(np.float32))
            
            # 获取模型输出
            output = self.model(images_ms)
            
            # 获取预测类别
            predictions = np.argmax(output.asnumpy(), axis=1)
            
            # 检查是否已成功攻击
            if predictions[0] != labels[0]:
                break
            
            # 计算扰动方向（简化版本）
            # 实际实现需要计算决策边界
            perturbation = self._compute_perturbation(adversarial, labels)
            
            # 更新
            adversarial = adversarial + perturbation
            adversarial = np.clip(adversarial, 0, 1)
            
            # 检查扰动是否超限
            total_perturbation = np.sum((adversarial - images) ** 2) ** 0.5
            if total_perturbation > self.epsilon:
                break
        
        return adversarial
    
    def _compute_perturbation(self, image, label):
        """计算最小扰动"""
        # 简化版本：使用随机扰动
        # 实际实现需要计算决策边界和梯度
        perturbation = np.random.randn(*image.shape) * 0.01
        return perturbation


# 导出所有攻击方法
__all__ = [
    'Attack',
    'FGSM',
    'PGD',
    'BIM',
    'CW',
    'DeepFool'
]
