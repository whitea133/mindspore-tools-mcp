"""
msutils.train.callbacks - 回调函数

提供训练过程中的各种回调功能
"""

import time
from typing import Dict, List, Optional, Callable
import numpy as np


class Callback:
    """回调函数基类"""
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始前调用"""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束后调用"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """每个 epoch 开始时调用"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """每个 epoch 结束时调用"""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """每个 batch 开始时调用"""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """每个 batch 结束时调用"""
        pass


class EarlyStopping(Callback):
    """
    早停回调
    
    当监控指标不再改善时，停止训练
    
    Args:
        monitor: 监控的指标名，默认 'val_loss'
        patience: 容忍多少个 epoch 没有改善，默认 10
        mode: 监控模式，'min' 或 'max'，默认 'min'
        min_delta: 最小改善量，默认 0
        verbose: 是否打印信息，默认 True
        restore_best: 是否恢复最佳模型，默认 True
    
    Example:
        >>> early_stop = EarlyStopping(patience=10, monitor='val_loss')
        >>> model.train(callbacks=[early_stop])
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0,
        verbose: bool = True,
        restore_best: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best = restore_best
        
        if mode == 'min':
            self.best = float('inf')
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.best = float('-inf')
            self.is_better = lambda current, best: current > best + min_delta
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """检查是否应该停止训练"""
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.is_better(current, self.best):
            if self.verbose:
                print(f'\nEpoch {epoch}: {self.monitor} improved from {self.best:.6f} to {current:.6f}')
            self.best = current
            self.wait = 0
            
            if self.restore_best:
                self.best_weights = logs.get('weights')
        else:
            self.wait += 1
            if self.verbose:
                print(f'\nEpoch {epoch}: {self.monitor} did not improve ({self.wait}/{self.patience})')
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                return True
        
        return False
    
    def get_best_score(self) -> float:
        """获取最佳分数"""
        return self.best


class ModelCheckpoint(Callback):
    """
    模型检查点回调
    
    保存训练过程中的模型快照
    
    Args:
        filepath: 保存路径，支持格式化字符串如 'model_{epoch:02d}_{val_loss:.4f}.ckpt'
        monitor: 监控的指标，默认 'val_loss'
        mode: 'min' 或 'max'
        save_best_only: 是否只保存最佳模型
        save_weights_only: 是否只保存权重
        verbose: 是否打印信息
    
    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='./checkpoints/model_{epoch:02d}.ckpt',
        ...     monitor='val_loss',
        ...     save_best_only=True
        ... )
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        if mode == 'min':
            self.best = float('inf')
            self.is_better = lambda current: current < self.best
        else:
            self.best = float('-inf')
            self.is_better = lambda current: current > self.best
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """保存检查点"""
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.save_best_only:
            if self.is_better(current):
                if self.verbose:
                    print(f'\nEpoch {epoch}: {self.monitor} improved to {current:.6f}, saving model')
                self.best = current
                self._save_checkpoint(epoch, logs)
        else:
            if self.verbose:
                print(f'\nEpoch {epoch}: saving model')
            self._save_checkpoint(epoch, logs)
    
    def _save_checkpoint(self, epoch: int, logs: Dict):
        """内部保存方法"""
        import os
        from mindspore import save_checkpoint
        
        # 格式化文件名
        filepath = self.filepath.format(epoch=epoch, **logs)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        if self.save_weights_only:
            # 只保存权重
            pass  # 需要模型实例
        else:
            # 保存完整检查点
            save_checkpoint(logs.get('model'), filepath)


class LearningRateScheduler(Callback):
    """
    学习率调度回调
    
    在每个 epoch 调整学习率
    
    Args:
        schedule: 学习率调度函数，接受 epoch 返回新的学习率
        verbose: 是否打印新的学习率
    
    Example:
        >>> def lr_schedule(epoch):
        ...     if epoch < 10:
        ...         return 0.1
        ...     return 0.1 * (0.1 ** (epoch // 30))
        >>> lr_scheduler = LearningRateScheduler(lr_schedule)
    """
    
    def __init__(self, schedule: Callable[[int], float], verbose: bool = True):
        super().__init__()
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """更新学习率"""
        new_lr = self.schedule(epoch)
        
        if logs and 'optimizer' in logs:
            logs['optimizer'].learning_rate = new_lr
        
        if self.verbose:
            print(f'\nEpoch {epoch}: Learning rate set to {new_lr:.6f}')


class MetricsLogger(Callback):
    """
    指标记录回调
    
    记录训练过程中的各种指标
    
    Args:
        log_dir: 日志保存目录
        metrics: 要记录的指标列表
    
    Example:
        >>> logger = MetricsLogger('./logs', metrics=['loss', 'accuracy'])
    """
    
    def __init__(self, log_dir: str = './logs', metrics: Optional[List[str]] = None):
        super().__init__()
        self.log_dir = log_dir
        self.metrics = metrics or ['loss']
        self.history = {m: [] for m in self.metrics}
        self.train_history = {m: [] for m in self.metrics}
        self.val_history = {m: [] for m in self.metrics}
        
        import os
        os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """记录指标"""
        if logs is None:
            logs = {}
        
        for metric in self.metrics:
            if metric in logs:
                value = logs[metric]
                self.history[metric].append(value)
                
                # 区分训练和验证指标
                if metric.startswith('val_'):
                    key = metric[4:]
                    self.val_history[key].append(value)
                else:
                    self.train_history[metric].append(value)
    
    def get_history(self, metric: str) -> List[float]:
        """获取指标历史"""
        return self.history.get(metric, [])
    
    def save_history(self, filename: str = 'training_history.txt'):
        """保存历史记录到文件"""
        import os
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("Training History\n")
            f.write("=" * 50 + "\n\n")
            
            for metric, values in self.history.items():
                f.write(f"{metric}:\n")
                for i, v in enumerate(values):
                    f.write(f"  Epoch {i+1}: {v:.6f}\n")
                f.write("\n")


class ProgressBar(Callback):
    """
    进度条回调
    
    显示训练进度
    
    Args:
        total_epochs: 总 epoch 数
        total_batches: 每个 epoch 的 batch 数
        width: 进度条宽度
    
    Example:
        >>> progress = ProgressBar(total_epochs=100, total_batches=1000)
    """
    
    def __init__(self, total_epochs: int, total_batches: int, width: int = 50):
        super().__init__()
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.width = width
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """显示 epoch 开始"""
        print(f"\nEpoch {epoch}/{self.total_epochs}")
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """显示 batch 进度"""
        if batch % 10 == 0:  # 每10个batch更新一次
            progress = int((batch / self.total_batches) * self.width)
            bar = '=' * progress + '-' * (self.width - progress)
            loss = logs.get('loss', 0) if logs else 0
            print(f"\r  [{bar}] {batch}/{self.total_batches} - loss: {loss:.4f}", end='')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """epoch 结束"""
        print()  # 换行


class GradientCaccumulationCallback(Callback):
    """
    梯度累积回调
    
    实现梯度累积以模拟更大批量的训练
    
    Args:
        accumulation_steps: 累积步数
        original_batch_size: 原始批量大小
    
    Example:
        >>> grad_accum = GradientCaccumulationCallback(accumulation_steps=4)
    """
    
    def __init__(self, accumulation_steps: int = 4):
        super().__init__()
        self.accumulation_steps = accumulation_steps
        self.step = 0
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """累积梯度"""
        self.step += 1
        if self.step % self.accumulation_steps == 0:
            # 执行优化器步骤
            if logs and 'optimizer' in logs:
                logs['optimizer'].step()
            self.step = 0


class LRWarmupCallback(Callback):
    """
    学习率预热回调
    
    在训练初期逐渐增加学习率
    
    Args:
        warmup_epochs: 预热 epoch 数
        initial_lr: 初始学习率
        target_lr: 目标学习率
    
    Example:
        >>> warmup = LRWarmupCallback(warmup_epochs=5, initial_lr=1e-5, target_lr=1e-3)
    """
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        initial_lr: float = 1e-5,
        target_lr: float = 1e-3,
        verbose: bool = True
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """计算当前学习率"""
        if epoch < self.warmup_epochs:
            # 线性预热
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
        else:
            lr = self.target_lr
        
        if logs and 'optimizer' in logs:
            logs['optimizer'].learning_rate = lr
        
        if self.verbose and epoch < self.warmup_epochs:
            print(f"Epoch {epoch}: Warmup LR = {lr:.6f}")


class LambdaCallback(Callback):
    """
    Lambda 回调
    
    使用自定义函数作为回调
    
    Args:
        on_epoch_begin_fn: epoch 开始时调用的函数
        on_epoch_end_fn: epoch 结束时调用的函数
        on_batch_begin_fn: batch 开始时调用的函数
        on_batch_end_fn: batch 结束时调用的函数
    
    Example:
        >>> callback = LambdaCallback(
        ...     on_epoch_end_fn=lambda epoch, logs: print(f"Epoch {epoch} done")
        ... )
    """
    
    def __init__(
        self,
        on_train_begin_fn: Optional[Callable] = None,
        on_train_end_fn: Optional[Callable] = None,
        on_epoch_begin_fn: Optional[Callable] = None,
        on_epoch_end_fn: Optional[Callable] = None,
        on_batch_begin_fn: Optional[Callable] = None,
        on_batch_end_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.on_train_begin_fn = on_train_begin_fn
        self.on_train_end_fn = on_train_end_fn
        self.on_epoch_begin_fn = on_epoch_begin_fn
        self.on_epoch_end_fn = on_epoch_end_fn
        self.on_batch_begin_fn = on_batch_begin_fn
        self.on_batch_end_fn = on_batch_end_fn
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        if self.on_train_begin_fn:
            self.on_train_begin_fn(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        if self.on_train_end_fn:
            self.on_train_end_fn(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        if self.on_epoch_begin_fn:
            self.on_epoch_begin_fn(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if self.on_epoch_end_fn:
            self.on_epoch_end_fn(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        if self.on_batch_begin_fn:
            self.on_batch_begin_fn(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.on_batch_end_fn:
            self.on_batch_end_fn(batch, logs)


class ReduceLROnPlateau(Callback):
    """
    学习率降低回调
    
    当指标停止改善时降低学习率
    
    Args:
        monitor: 监控的指标
        factor: 学习率降低因子
        patience: 多少个 epoch 无改善后降低学习率
        min_lr: 最小学习率
        mode: 'min' 或 'max'
        verbose: 是否打印
    
    Example:
        >>> reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        mode: str = 'min',
        verbose: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'min':
            self.best = float('inf')
            self.is_better = lambda current, best: current < best
        else:
            self.best = float('-inf')
            self.is_better = lambda current, best: current > best
        
        self.wait = 0
        self.current_lr = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """检查是否需要降低学习率"""
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.is_better(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                # 获取当前学习率
                if logs and 'optimizer' in logs:
                    current_lr = logs['optimizer'].learning_rate
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    
                    if new_lr < current_lr:
                        logs['optimizer'].learning_rate = new_lr
                        self.wait = 0
                        
                        if self.verbose:
                            print(f"\nEpoch {epoch}: Reducing learning rate to {new_lr:.6f}")


class TensorBoardCallback(Callback):
    """
    TensorBoard 回调（简化版）
    
    记录标量和图像到 TensorBoard
    
    Args:
        log_dir: 日志目录
    
    Example:
        >>> tb = TensorBoardCallback('./logs')
    """
    
    def __init__(self, log_dir: str = './logs'):
        super().__init__()
        self.log_dir = log_dir
        self.scalar_history = {}
        
        import os
        os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """记录标量数据"""
        if logs is None:
            logs = {}
        
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.scalar_history:
                    self.scalar_history[key] = []
                self.scalar_history[key].append((epoch, value))
    
    def save(self, filename: str = 'scalars.json'):
        """保存标量历史"""
        import json
        import os
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.scalar_history, f, indent=2)


class CallbackList:
    """
    回调列表
    
    管理多个回调的顺序调用
    
    Example:
        >>> callbacks = CallbackList([early_stop, checkpoint, logger])
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """添加回调"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        stop_training = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs)
            if result is True:
                stop_training = True
        return stop_training
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


# 导出所有回调
__all__ = [
    'Callback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'MetricsLogger',
    'ProgressBar',
    'GradientCaccumulationCallback',
    'LRWarmupCallback',
    'LambdaCallback',
    'ReduceLROnPlateau',
    'TensorBoardCallback',
    'CallbackList'
]
