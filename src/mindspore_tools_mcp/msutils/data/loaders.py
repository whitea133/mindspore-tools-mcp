"""
msutils.data.loaders - 数据加载器

提供常用数据集的快速加载功能
"""

import os
from typing import Tuple, Optional, Callable, List
import numpy as np


class MnistLoader:
    """
    MNIST 数据加载器
    
    快速加载 MNIST 数据集
    
    Args:
        data_dir: 数据存储目录
        train: 是否加载训练集
        transform: 数据转换函数
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
    
    Example:
        >>> from msutils.data import MnistLoader
        >>> loader = MnistLoader('./data', train=True)
        >>> train_dataset = loader.get_dataset()
        >>> for images, labels in train_dataset:
        ...     print(images.shape)  # (batch_size, 1, 28, 28)
    """
    
    def __init__(
        self,
        data_dir: str = './mnist_data',
        train: bool = True,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_dataset(self):
        """获取 MindSpore 数据集"""
        import mindspore.dataset as ds
        
        usage = 'train' if self.train else 'test'
        dataset = ds.MnistDataset(self.data_dir, usage=usage)
        
        if self.transform:
            dataset = dataset.map(operations=self.transform, input_columns='image')
        
        dataset = dataset.batch(self.batch_size, shuffle=self.shuffle)
        
        return dataset
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        return {
            'dataset': 'MNIST',
            'train': self.train,
            'num_classes': 10,
            'image_size': (28, 28),
            'channels': 1,
            'train_samples': 60000 if self.train else None,
            'test_samples': None if self.train else 10000
        }


class Cifar10Loader:
    """
    CIFAR-10 数据加载器
    
    快速加载 CIFAR-10 数据集
    
    Args:
        data_dir: 数据存储目录
        train: 是否加载训练集
        transform: 数据转换函数
        batch_size: 批大小
        shuffle: 是否打乱数据
    
    Example:
        >>> from msutils.data import Cifar10Loader
        >>> loader = Cifar10Loader('./data', train=True)
        >>> train_dataset = loader.get_dataset()
    """
    
    CIFAR10_LABELS = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(
        self,
        data_dir: str = './cifar10_data',
        train: bool = True,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_dataset(self):
        """获取 MindSpore 数据集"""
        import mindspore.dataset as ds
        
        usage = 'train' if self.train else 'test'
        dataset = ds.Cifar10Dataset(self.data_dir, usage=usage)
        
        if self.transform:
            dataset = dataset.map(operations=self.transform, input_columns='image')
        
        dataset = dataset.batch(self.batch_size, shuffle=self.shuffle)
        
        return dataset
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        return {
            'dataset': 'CIFAR-10',
            'train': self.train,
            'num_classes': 10,
            'image_size': (32, 32),
            'channels': 3,
            'labels': self.CIFAR10_LABELS,
            'train_samples': 50000 if self.train else None,
            'test_samples': None if self.train else 10000
        }


class Cifar100Loader:
    """
    CIFAR-100 数据加载器
    
    快速加载 CIFAR-100 数据集
    
    Args:
        data_dir: 数据存储目录
        train: 是否加载训练集
        transform: 数据转换函数
        batch_size: 批大小
        shuffle: 是否打乱数据
    
    Example:
        >>> from msutils.data import Cifar100Loader
        >>> loader = Cifar100Loader('./data', train=True)
        >>> train_dataset = loader.get_dataset()
    """
    
    CIFAR100_LABELS = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crane', 'crocodile', 'crow',
        'cucumber', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
        'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
        'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
        'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
        'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'refrigerator', 'road',
        'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'shrimp', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'starfish', 'streetcar',
        'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
        'toaster', 'tortoise', 'tractor', 'truck', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    def __init__(
        self,
        data_dir: str = './cifar100_data',
        train: bool = True,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_dataset(self):
        """获取 MindSpore 数据集"""
        import mindspore.dataset as ds
        
        usage = 'train' if self.train else 'test'
        dataset = ds.Cifar100Dataset(self.data_dir, usage=usage)
        
        if self.transform:
            dataset = dataset.map(operations=self.transform, input_columns='image')
        
        dataset = dataset.batch(self.batch_size, shuffle=self.shuffle)
        
        return dataset
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        return {
            'dataset': 'CIFAR-100',
            'train': self.train,
            'num_classes': 100,
            'image_size': (32, 32),
            'channels': 3,
            'train_samples': 50000 if self.train else None,
            'test_samples': None if self.train else 10000
        }


class ImageNetLoader:
    """
    ImageNet 数据加载器
    
    快速加载 ImageNet 数据集
    
    Args:
        data_dir: 数据存储目录
        train: 是否加载训练集
        transform: 数据转换函数
        batch_size: 批大小
        shuffle: 是否打乱数据
    
    Example:
        >>> from msutils.data import ImageNetLoader
        >>> loader = ImageNetLoader('./imagenet_data', train=True)
        >>> train_dataset = loader.get_dataset()
    """
    
    def __init__(
        self,
        data_dir: str = './imagenet_data',
        train: bool = True,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8
    ):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def get_dataset(self):
        """获取 MindSpore 数据集"""
        import mindspore.dataset as ds
        
        # ImageNet 通常使用 ImageFolderDataset
        # 这里简化处理
        if self.train:
            dataset = ds.ImageFolderDataset(
                os.path.join(self.data_dir, 'train')
            )
        else:
            dataset = ds.ImageFolderDataset(
                os.path.join(self.data_dir, 'val')
            )
        
        if self.transform:
            dataset = dataset.map(operations=self.transform, input_columns='image')
        
        dataset = dataset.batch(self.batch_size, shuffle=self.shuffle)
        
        return dataset
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        return {
            'dataset': 'ImageNet',
            'train': self.train,
            'num_classes': 1000,
            'image_size': (224, 224),
            'channels': 3,
            'train_samples': 1281167 if self.train else None,
            'test_samples': None if self.train else 50000
        }


class Flowers102Loader:
    """
    Oxford 102 Flowers 数据加载器
    
    快速加载 Flowers-102 数据集
    
    Args:
        data_dir: 数据存储目录
        train: 是否加载训练集
        transform: 数据转换函数
        batch_size: 批大小
    
    Example:
        >>> from msutils.data import Flowers102Loader
        >>> loader = Flowers102Loader('./flowers_data')
        >>> dataset = loader.get_dataset()
    """
    
    def __init__(
        self,
        data_dir: str = './flowers102_data',
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def get_dataset(self):
        """获取 MindSpore 数据集"""
        import mindspore.dataset as ds
        
        # 使用通用数据集加载
        dataset = ds.Flowers102Dataset(self.data_dir)
        
        if self.transform:
            dataset = dataset.map(operations=self.transform, input_columns='image')
        
        dataset = dataset.batch(self.batch_size, shuffle=self.shuffle)
        
        return dataset


class VOCLoader:
    """
    VOC 数据集加载器
    
    加载 Pascal VOC 数据集
    
    Args:
        data_dir: 数据存储目录
        year: 数据集年份，如 '2012', '2007'
        task: 任务类型，如 'detection', 'segmentation'
        transform: 数据转换函数
        batch_size: 批大小
    
    Example:
        >>> from msutils.data import VOCLoader
        >>> loader = VOCLoader('./voc_data', year='2012', task='detection')
        >>> dataset = loader.get_dataset()
    """
    
    def __init__(
        self,
        data_dir: str = './voc_data',
        year: str = '2012',
        task: str = 'detection',
        transform: Optional[Callable] = None,
        batch_size: int = 8
    ):
        self.data_dir = data_dir
        self.year = year
        self.task = task
        self.transform = transform
        self.batch_size = batch_size
    
    def get_dataset(self):
        """获取 MindSpore 数据集"""
        import mindspore.dataset as ds
        
        # VOC 数据集加载
        dataset = ds.VOCDataset(
            self.data_dir,
            task=self.task,
            year=self.year,
            decode=True
        )
        
        if self.transform:
            dataset = dataset.map(operations=self.transform, input_columns='image')
        
        dataset = dataset.batch(self.batch_size)
        
        return dataset


def create_loader(
    dataset_name: str,
    data_dir: str = './data',
    train: bool = True,
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    **kwargs
):
    """
    通用数据加载器工厂函数
    
    Args:
        dataset_name: 数据集名称
        data_dir: 数据存储目录
        train: 是否加载训练集
        transform: 数据转换函数
        batch_size: 批大小
        **kwargs: 其他参数
    
    Returns:
        数据加载器实例
    
    Example:
        >>> from msutils.data import create_loader
        >>> loader = create_loader('mnist', './data', train=True)
        >>> dataset = loader.get_dataset()
    """
    loaders = {
        'mnist': MnistLoader,
        'cifar10': Cifar10Loader,
        'cifar100': Cifar100Loader,
        'imagenet': ImageNetLoader,
        'flowers102': Flowers102Loader,
        'voc': VOCLoader
    }
    
    dataset_name = dataset_name.lower()
    
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(loaders.keys())}"
        )
    
    loader_class = loaders[dataset_name]
    
    return loader_class(
        data_dir=data_dir,
        train=train,
        transform=transform,
        batch_size=batch_size,
        **kwargs
    )


# 导出所有加载器
__all__ = [
    'MnistLoader',
    'Cifar10Loader',
    'Cifar100Loader',
    'ImageNetLoader',
    'Flowers102Loader',
    'VOCLoader',
    'create_loader'
]
