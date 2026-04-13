"""
msutils.distributed.ddp - 分布式数据并行工具

提供 MindSpore 分布式训练的便捷接口
"""

import os
from typing import Optional, Callable, Any
import numpy as np


class DistributedConfig:
    """
    分布式配置
    
    管理分布式训练的配置信息
    
    Example:
        >>> config = DistributedConfig(
        ...     rank=0,
        ...     world_size=4,
        ...     backend='nccl'
        ... )
        >>> print(config.is_master)
    """
    
    def __init__(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        backend: str = 'nccl',
        init_method: Optional[str] = None
    ):
        """
        初始化分布式配置
        
        Args:
            rank: 当前进程的 rank
            world_size: 总进程数
            backend: 通信后端 ('nccl', 'gloo', 'mpi')
            init_method: 初始化方法
        """
        # 从环境变量获取默认值
        self.rank = rank if rank is not None else int(os.environ.get('RANK', 0))
        self.world_size = world_size if world_size is not None else int(os.environ.get('WORLD_SIZE', 1))
        self.backend = backend
        self.init_method = init_method or os.environ.get('MASTER_ADDR', 'localhost')
        
        # 计算属性
        self.is_master = self.rank == 0
        self.is_distributed = self.world_size > 1
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'rank': self.rank,
            'world_size': self.world_size,
            'backend': self.backend,
            'init_method': self.init_method,
            'is_master': self.is_master,
            'is_distributed': self.is_distributed
        }
    
    def __repr__(self) -> str:
        return (
            f"DistributedConfig(rank={self.rank}, world_size={self.world_size}, "
            f"backend={self.backend}, is_master={self.is_master})"
        )


def init_distributed(
    backend: str = 'nccl',
    init_method: Optional[str] = None
) -> DistributedConfig:
    """
    初始化分布式环境
    
    Args:
        backend: 通信后端
        init_method: 初始化方法
    
    Returns:
        分布式配置
    
    Example:
        >>> config = init_distributed()
        >>> print(f"Rank: {config.rank}")
    """
    try:
        import mindspore as ms
        from mindspore.communication import init, get_rank, get_group_size
        
        # 初始化通信组
        init()
        
        rank = get_rank()
        world_size = get_group_size()
        
        return DistributedConfig(
            rank=rank,
            world_size=world_size,
            backend=backend,
            init_method=init_method
        )
    except ImportError:
        # 如果 MindSpore 不可用，返回单进程配置
        return DistributedConfig(rank=0, world_size=1)


def is_distributed() -> bool:
    """
    检查是否在分布式环境中
    
    Returns:
        是否分布式
    """
    try:
        from mindspore.communication import get_group_size
        return get_group_size() > 1
    except:
        return False


def get_rank() -> int:
    """
    获取当前进程的 rank
    
    Returns:
        当前 rank
    """
    try:
        from mindspore.communication import get_rank
        return get_rank()
    except:
        return 0


def get_world_size() -> int:
    """
    获取总进程数
    
    Returns:
        总进程数
    """
    try:
        from mindspore.communication import get_group_size
        return get_group_size()
    except:
        return 1


def is_master() -> bool:
    """
    检查是否为主进程
    
    Returns:
        是否为主进程
    """
    return get_rank() == 0


class DistributedSampler:
    """
    分布式数据采样器
    
    将数据分片到各个进程
    
    Example:
        >>> sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
        >>> indices = sampler.get_indices()
    """
    
    def __init__(
        self,
        dataset_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        初始化采样器
        
        Args:
            dataset_size: 数据集大小
            num_replicas: 副本数（进程数）
            rank: 当前 rank
            shuffle: 是否打乱
            seed: 随机种子
        """
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas or get_world_size()
        self.rank = rank or get_rank()
        self.shuffle = shuffle
        self.seed = seed
        
        # 计算每个进程的样本数
        self.num_samples = int(np.ceil(dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    
    def get_indices(self, epoch: int = 0) -> list:
        """
        获取当前进程的数据索引
        
        Args:
            epoch: 当前 epoch
        
        Returns:
            索引列表
        """
        # 生成所有索引
        indices = list(range(self.dataset_size))
        
        # 打乱
        if self.shuffle:
            g = np.random.RandomState(self.seed + epoch)
            g.shuffle(indices)
        
        # 填充到 total_size
        if len(indices) < self.total_size:
            indices += indices[:(self.total_size - len(indices))]
        
        # 取当前 rank 的部分
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return indices
    
    def __len__(self) -> int:
        return self.num_samples


def all_reduce(
    tensor,
    op: str = 'sum',
    group: Optional[Any] = None
):
    """
    全局归约操作
    
    Args:
        tensor: 输入张量
        op: 归约操作 ('sum', 'avg', 'max', 'min')
        group: 通信组
    
    Returns:
        归约后的张量
    
    Example:
        >>> result = all_reduce(tensor, op='avg')
    """
    try:
        import mindspore.ops as ops
        
        if op == 'sum':
            reduce_op = ops.AllReduce()
        elif op == 'avg':
            reduce_op = ops.AllReduce()
            tensor = tensor / get_world_size()
        else:
            raise ValueError(f"Unknown op: {op}")
        
        return reduce_op(tensor)
    except ImportError:
        return tensor


def all_gather(
    tensor,
    group: Optional[Any] = None
):
    """
    全局收集操作
    
    Args:
        tensor: 输入张量
        group: 通信组
    
    Returns:
        收集后的张量列表
    
    Example:
        >>> tensors = all_gather(tensor)
    """
    try:
        import mindspore.ops as ops
        
        all_gather_op = ops.AllGather()
        return all_gather_op(tensor)
    except ImportError:
        return tensor


def broadcast(
    tensor,
    src: int = 0,
    group: Optional[Any] = None
):
    """
    广播操作
    
    Args:
        tensor: 输入张量
        src: 源 rank
        group: 通信组
    
    Returns:
        广播后的张量
    
    Example:
        >>> tensor = broadcast(tensor, src=0)
    """
    try:
        import mindspore.ops as ops
        
        broadcast_op = ops.Broadcast(src)
        return broadcast_op(tensor)
    except ImportError:
        return tensor


def barrier():
    """
    同步屏障
    
    等待所有进程到达
    
    Example:
        >>> barrier()
    """
    try:
        from mindspore.communication import barrier as ms_barrier
        ms_barrier()
    except ImportError:
        pass


class DistributedTrainer:
    """
    分布式训练器
    
    封装分布式训练的常用操作
    
    Example:
        >>> trainer = DistributedTrainer(model, config)
        >>> trainer.train(train_dataset, epochs=10)
    """
    
    def __init__(
        self,
        model,
        config: Optional[DistributedConfig] = None,
        optimizer=None,
        loss_fn=None
    ):
        """
        初始化分布式训练器
        
        Args:
            model: 模型
            config: 分布式配置
            optimizer: 优化器
            loss_fn: 损失函数
        """
        self.model = model
        self.config = config or DistributedConfig()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def is_master(self) -> bool:
        """是否为主进程"""
        return self.config.is_master
    
    def save_checkpoint(self, path: str, **kwargs):
        """
        保存检查点（仅主进程）
        
        Args:
            path: 保存路径
            **kwargs: 其他保存内容
        """
        if self.is_master():
            try:
                import mindspore as ms
                ms.save_checkpoint(self.model, path)
            except ImportError:
                pass
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        try:
            import mindspore as ms
            param_dict = ms.load_checkpoint(path)
            ms.load_param_into_net(self.model, param_dict)
        except ImportError:
            pass


# 导出所有函数和类
__all__ = [
    'DistributedConfig',
    'init_distributed',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'is_master',
    'DistributedSampler',
    'all_reduce',
    'all_gather',
    'broadcast',
    'barrier',
    'DistributedTrainer'
]