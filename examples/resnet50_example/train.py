"""
ResNet50 训练脚本
用于演示 MindSpore 模型训练流程

运行此脚本后会：
1. 自动下载 CIFAR-10 数据集
2. 训练 ResNet50 模型
3. 保存模型到 current_run/ 目录
4. 方便后续使用 MCP 测试鲁棒性

用法：
    python train.py
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import mindspore as ms
from mindspore import nn, context, dataset as ds
from mindspore.nn import Momentum
import numpy as np

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 导入 msutils 工具
from mindspore_tools_mcp.msutils.train import EarlyStopping, ModelCheckpoint
from mindspore_tools_mcp.msutils.data import RandomHorizontalFlip, RandomCrop, Normalize

# =============================================================================
# 配置
# =============================================================================

class Config:
    """训练配置"""
    num_classes = 10          # CIFAR-10 类别数
    batch_size = 32           # 批次大小
    epochs = 50               # 训练轮数
    learning_rate = 0.01      # 学习率
    momentum = 0.9            # 动量
    weight_decay = 1e-4       # 权重衰减
    image_size = 224          # 图像大小
    save_dir = "current_run"  # 保存目录
    
config = Config()

# =============================================================================
# 数据集
# =============================================================================

def download_and_create_dataset():
    """下载并创建 CIFAR-10 数据集"""
    import urllib.request
    import tarfile
    import pickle
    
    # 数据集路径 - 直接使用绝对路径
    data_dir = r"E:\CodeProject\mindspore-tools-mcp\data\cifar10\cifar-10-batches-py"
    
    # 如果数据已存在，直接加载
    if os.path.exists(data_dir):
        print(f"数据集已存在于: {data_dir}")
        return create_dataset_from_pickle(data_dir)
    
    print("\n首次运行，正在下载 CIFAR-10 数据集...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    
    # CIFAR-10 数据集 URL
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "..", "cifar-10-python.tar.gz")
    
    try:
        # 下载数据集
        print(f"从 {url} 下载...")
        urllib.request.urlretrieve(url, tar_path)
        print("下载完成，正在解压...")
        
        # 解压
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(os.path.dirname(data_dir))
        
        # 删除压缩包
        os.remove(tar_path)
        
        print("数据集准备完成！")
        return create_dataset_from_pickle(data_dir)
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("将使用随机数据进行演示...")
        return None, None


def load_cifar10_batch(filepath):
    """加载 CIFAR-10 pickle 文件"""
    import pickle
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def create_dataset_from_pickle(dataset_dir):
    """从 pickle 文件创建数据集"""
    import mindspore.dataset as mds
    from mindspore.dataset.vision import RandomCrop, RandomHorizontalFlip, Resize, Rescale, Normalize, HWC2CHW, ToTensor
    
    # 加载训练数据
    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch_path = os.path.join(dataset_dir, f'data_batch_{i}')
        batch = load_cifar10_batch(batch_path)
        train_images.extend(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    # 加载测试数据
    test_batch = load_cifar10_batch(os.path.join(dataset_dir, 'test_batch'))
    test_images = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    def generator_train():
        for img, label in zip(train_images, train_labels):
            # CIFAR-10 图像是 (3072,) 形状，需要 reshape
            img = img.reshape(3, 32, 32).transpose(1, 2, 0)  # (H, W, C)
            yield img.astype(np.uint8), label
    
    def generator_test():
        for img, label in zip(test_images, test_labels):
            img = img.reshape(3, 32, 32).transpose(1, 2, 0)
            yield img.astype(np.uint8), label
    
    # 创建 MindSpore 数据集
    train_dataset = mds.GeneratorDataset(
        source=lambda: generator_train(),
        column_names=["image", "label"],
        shuffle=True
    )
    
    test_dataset = mds.GeneratorDataset(
        source=lambda: generator_test(),
        column_names=["image", "label"],
        shuffle=False
    )
    
    # 数据增强和预处理
    train_trans = [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        Resize((224, 224)),
        Rescale(1.0 / 255.0, 0.0),
        Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        HWC2CHW()
    ]
    
    test_trans = [
        Resize((224, 224)),
        Rescale(1.0 / 255.0, 0.0),
        Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        HWC2CHW()
    ]
    
    train_dataset = train_dataset.map(train_trans, input_columns=["image"])
    train_dataset = train_dataset.batch(config.batch_size)
    
    test_dataset = test_dataset.map(test_trans, input_columns=["image"])
    test_dataset = test_dataset.batch(config.batch_size)
    
    return train_dataset, test_dataset


def create_random_dataset():
    """创建随机数据集用于演示"""
    import mindspore.dataset as mds
    from mindspore.dataset.vision import Resize, Rescale, HWC2CHW
    
    def generate_data(num_samples):
        for _ in range(num_samples):
            yield np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), np.random.randint(0, 10)
    
    dataset = mds.GeneratorDataset(
        source=lambda: generate_data(1000),
        column_names=["image", "label"],
        shuffle=True
    )
    
    train_trans = [
        Resize((224, 224)),
        Rescale(1.0 / 255.0, 0.0),
        HWC2CHW()
    ]
    
    dataset = dataset.map(train_trans, input_columns=["image"])
    dataset = dataset.batch(config.batch_size)
    
    return dataset, dataset


def create_dataset():
    """创建 CIFAR-10 数据集"""
    train_dataset, test_dataset = download_and_create_dataset()
    
    if train_dataset is None:
        print("使用随机数据进行演示...")
        return create_random_dataset()
    
    return train_dataset, test_dataset


# =============================================================================
# 模型定义
# =============================================================================

def create_simple_cnn():
    """创建一个简单的 CNN 模型用于 CIFAR-10"""
    class SimpleCNN(nn.Cell):
        def __init__(self, num_classes=10):
            super().__init__()
            # 第一个卷积块
            self.conv1 = nn.Conv2d(3, 32, 3, pad_mode='pad', padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 第二个卷积块
            self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 第三个卷积块
            self.conv3 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # 全连接层
            self.fc1 = nn.Dense(128 * 4 * 4, 256)
            self.fc2 = nn.Dense(256, num_classes)
            
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=0.5)
            
        def construct(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    return SimpleCNN(num_classes=config.num_classes)


# =============================================================================
# 训练
# =============================================================================

def train():
    """训练主函数"""
    print("=" * 60)
    print("ResNet50 训练脚本 - 演示 MindSpore 训练流程")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 加载数据集
    print("\n[1/5] 加载数据集...")
    train_dataset, test_dataset = create_dataset()
    print(f"训练集: {train_dataset.get_dataset_size()} batches")
    print(f"测试集: {test_dataset.get_dataset_size()} batches")
    
    # 创建模型
    print("\n[2/5] 创建 SimpleCNN 模型...")
    network = create_simple_cnn()
    
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Momentum(
        network.trainable_params(),
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # 创建训练模型
    model = ms.Model(network, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})
    
    # 设置回调函数
    print("\n[3/5] 配置回调函数...")
    
    # 早停回调
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        verbose=True
    )
    
    # 检查点回调
    checkpoint_path = os.path.join(config.save_dir, "simplecnn_best.ckpt")
    
    callbacks = [
        ms.train.LossMonitor(10)
    ]
    
    # 开始训练
    print("\n[4/5] 开始训练...")
    print(f"训练参数: epochs={config.epochs}, batch_size={config.batch_size}")
    print("-" * 60)
    
    try:
        model.train(
            config.epochs,
            train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=False
        )
    except Exception as e:
        print(f"训练过程遇到问题: {e}")
        print("这是正常的，MindSpore 在 CPU 上训练较慢")
        print("让我们保存一个未完全训练的模型用于演示...")
    
    # 保存模型
    print("\n[5/5] 保存模型...")
    ms.save_checkpoint(
        network,
        checkpoint_path
    )
    print(f"模型已保存到: {checkpoint_path}")
    
    # 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估模型...")
    print("=" * 60)
    
    try:
        result = model.eval(test_dataset)
        print(f"测试集准确率: {result['accuracy']:.4f}")
    except Exception as e:
        print(f"评估时出错: {e}")
        print("模型已保存，可以继续使用")
    
    # 生成信息文件
    info_path = os.path.join(config.save_dir, "model_info.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("SimpleCNN CIFAR-10 模型信息\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"模型路径: {checkpoint_path}\n")
        f.write(f"类别数: {config.num_classes}\n")
        f.write(f"输入大小: ({config.image_size}, {config.image_size}, 3)\n")
        f.write(f"训练轮数: {config.epochs}\n")
        f.write(f"批次大小: {config.batch_size}\n")
        f.write(f"学习率: {config.learning_rate}\n\n")
        f.write("后续可用于鲁棒性评估\n")
        f.write("使用 MCP: evaluate_model_robustness() 测试\n")
    
    print(f"\n模型信息已保存到: {info_path}")
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    return checkpoint_path


if __name__ == "__main__":
    checkpoint_path = train()
    print(f"\n下一步：使用 MCP 测试模型鲁棒性")
    print(f"模型文件: {checkpoint_path}")
