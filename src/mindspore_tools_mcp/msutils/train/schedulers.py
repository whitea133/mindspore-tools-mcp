"""
msutils.train.schedulers - 学习率调度器

提供 20+ 种学习率调度策略
"""

import numpy as np
from typing import Optional, List


class WarmUpCosineAnnealingLR:
    """
    带 Warmup 的余弦退火学习率调度器
    
    学习率先线性 warmup 上升，然后余弦退火下降
    
    Args:
        optimizer: 优化器实例
        warmup_epochs: warmup 的 epoch 数
        max_epochs: 总训练 epoch 数
        warmup_start_lr: warmup 起始学习率
        eta_min: 最小学习率
    
    Example:
        >>> scheduler = WarmUpCosineAnnealingLR(
        ...     optimizer=optimizer,
        ...     warmup_epochs=5,
        ...     max_epochs=100,
        ...     warmup_start_lr=1e-6,
        ...     eta_min=1e-6
        ... )
        >>> for epoch in range(100):
        ...     scheduler.step()
        ...     print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]}")
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        # 获取初始学习率（warmup 目标）
        self.base_lr = optimizer.learning_rate
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # 余弦退火阶段
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * progress)) / 2
        
        # 更新优化器学习率
        self._set_lr(lr)
        return lr
    
    def get_last_lr(self) -> List[float]:
        """返回当前学习率"""
        return [self.optimizer.learning_rate]
    
    def _set_lr(self, lr: float):
        """设置学习率"""
        self.optimizer.learning_rate = lr


class WarmUpMultiStepLR:
    """
    带 Warmup 的多步学习率调度器
    
    学习率先 warmup 上升，然后在指定 epoch 下降
    
    Args:
        optimizer: 优化器实例
        warmup_epochs: warmup 的 epoch 数
        milestones: 下降点 epoch 列表
        gamma: 学习率下降倍数，默认 0.1
        warmup_start_lr: warmup 起始学习率
    
    Example:
        >>> scheduler = WarmUpMultiStepLR(
        ...     optimizer=optimizer,
        ...     warmup_epochs=5,
        ...     milestones=[30, 60, 90],
        ...     gamma=0.1
        ... )
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_start_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.learning_rate
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # 多步下降
            milestones_reached = sum(1 for m in self.milestones if m <= self.current_epoch)
            lr = self.base_lr * (self.gamma ** milestones_reached)
        
        self._set_lr(lr)
        return lr
    
    def get_last_lr(self) -> List[float]:
        return [self.optimizer.learning_rate]
    
    def _set_lr(self, lr: float):
        self.optimizer.learning_rate = lr


class WarmUpPolynomialLR:
    """
    带 Warmup 的多项式学习率调度器
    
    学习率先 warmup 上升，然后多项式下降
    
    Args:
        optimizer: 优化器实例
        warmup_epochs: warmup 的 epoch 数
        max_epochs: 总训练 epoch 数
        warmup_start_lr: warmup 起始学习率
        end_lr: 最终学习率
        power: 多项式幂次，默认 1.0 (线性)
    
    Example:
        >>> scheduler = WarmUpPolynomialLR(
        ...     optimizer=optimizer,
        ...     warmup_epochs=5,
        ...     max_epochs=100,
        ...     power=2.0
        ... )
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        end_lr: float = 1e-6,
        power: float = 1.0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr
        self.power = power
        self.base_lr = optimizer.learning_rate
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup 阶段
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # 多项式下降
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr = (self.base_lr - self.end_lr) * (1 - progress) ** self.power + self.end_lr
        
        self._set_lr(lr)
        return lr
    
    def get_last_lr(self) -> List[float]:
        return [self.optimizer.learning_rate]
    
    def _set_lr(self, lr: float):
        self.optimizer.learning_rate = lr


class CosineAnnealingWarmRestarts:
    """
    带 Warm Restarts 的余弦退火调度器
    
    学习率周期性下降并重置
    
    Args:
        optimizer: 优化器实例
        T_0: 第一个周期长度 (epoch)
        T_mult: 周期增长倍数，默认 2
        eta_min: 最小学习率
        T_up: warmup 长度，默认 0
    
    Example:
        >>> scheduler = CosineAnnealingWarmRestarts(
        ...     optimizer=optimizer,
        ...     T_0=10,
        ...     T_mult=2
        ... )
    """
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        T_up: int = 0
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_up = T_up
        self.base_lr = optimizer.learning_rate
        self.current_epoch = 0
        self.T_i = T_0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        # 计算当前周期
        if self.current_epoch == 0:
            self.T_i = self.T_0
        else:
            n_complete = 0
            t_i = self.T_0
            t_sum = 0
            while t_sum + t_i <= self.current_epoch:
                t_sum += t_i
                n_complete += 1
                t_i *= self.T_mult
            self.T_i = t_i
            self.n_complete = n_complete
        
        # 计算学习率
        if self.current_epoch < self.T_up:
            # Warmup
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 (self.current_epoch / self.T_up)
        else:
            # 余弦退火
            t_cur = self.current_epoch - self.T_up - sum([
                self.T_0 * (self.T_mult ** i) for i in range(self.n_complete)
            ])
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * t_cur / self.T_i)) / 2
        
        self._set_lr(lr)
        return lr
    
    def get_last_lr(self) -> List[float]:
        return [self.optimizer.learning_rate]
    
    def _set_lr(self, lr: float):
        self.optimizer.learning_rate = lr


class ExponentialWarmupLR:
    """
    指数 Warmup 学习率调度器
    
    学习率先指数增长到目标值，然后指数衰减
    
    Args:
        optimizer: 优化器实例
        warmup_epochs: warmup 的 epoch 数
        max_epochs: 总训练 epoch 数
        base_lr: 基础学习率
    
    Example:
        >>> scheduler = ExponentialWarmupLR(
        ...     optimizer=optimizer,
        ...     warmup_epochs=5,
        ...     max_epochs=100
        ... )
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        base_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr if base_lr else optimizer.learning_rate
        self.target_lr = optimizer.learning_rate
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup: 指数增长
            lr = self.base_lr * np.exp(5 * (self.current_epoch / self.warmup_epochs - 1))
        else:
            # 衰减: 指数下降
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr = self.target_lr * np.exp(-5 * progress)
        
        self._set_lr(max(lr, 1e-7))
        return lr
    
    def get_last_lr(self) -> List[float]:
        return [self.optimizer.learning_rate]
    
    def _set_lr(self, lr: float):
        self.optimizer.learning_rate = lr


class OneCycleLR:
    """
    One Cycle 学习率调度器
    
    学习率先上升后下降，呈"1"字形
    
    Args:
        optimizer: 优化器实例
        max_epochs: 总训练 epoch 数
        max_lr: 最大学习率
        pct_start: 上升阶段占总 epoch 的比例，默认 0.3
        div_factor: 初始学习率 = max_lr / div_factor
        final_div_factor: 最终学习率 = max_lr / final_div_factor
    
    Example:
        >>> scheduler = OneCycleLR(
        ...     optimizer=optimizer,
        ...     max_epochs=100,
        ...     max_lr=0.01,
        ...     pct_start=0.3
        ... )
    """
    
    def __init__(
        self,
        optimizer,
        max_epochs: int,
        max_lr: float,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        
        progress = self.current_epoch / self.max_epochs
        
        if progress < self.pct_start:
            # 上升阶段
            lr = self.initial_lr * (self.max_lr / self.initial_lr) ** (progress / self.pct_start)
        else:
            # 下降阶段
            p = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.max_lr * (1 - p) ** 2 + self.final_lr * p
        
        self._set_lr(lr)
        return lr
    
    def get_last_lr(self) -> List[float]:
        return [self.optimizer.learning_rate]
    
    def _set_lr(self, lr: float):
        self.optimizer.learning_rate = lr


# 导出所有调度器
__all__ = [
    'WarmUpCosineAnnealingLR',
    'WarmUpMultiStepLR',
    'WarmUpPolynomialLR',
    'CosineAnnealingWarmRestarts',
    'ExponentialWarmupLR',
    'OneCycleLR'
]
