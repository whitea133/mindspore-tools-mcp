"""
ResNet50 鲁棒性评估脚本
用于测试模型在对抗攻击下的表现

使用方法：
1. 先运行 train.py 训练模型
2. 然后运行此脚本评估模型鲁棒性
    python evaluate_robustness.py
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits

# 导入 msutils 安全模块
from mindspore_tools_mcp.msutils.security import FGSM, PGD

# =============================================================================
# 配置
# =============================================================================

class Config:
    """评估配置"""
    model_path = "../../current_run/simplecnn_best.ckpt"  # 模型路径
    num_classes = 10                                  # CIFAR-10 类别数
    batch_size = 32                                   # 批次大小
    num_samples = 100                                 # 评估样本数
    epsilon = 0.03                                    # 攻击扰动大小
    
config = Config()

# =============================================================================
# 模型加载
# =============================================================================

def load_model():
    """加载训练好的模型"""
    print("加载模型...")
    from mindspore import nn
    
    # 定义与训练时相同的模型结构
    class SimpleCNN(nn.Cell):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, pad_mode='pad', padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='pad', padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv3 = nn.Conv2d(64, 128, 3, pad_mode='pad', padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
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
    
    network = SimpleCNN(num_classes=config.num_classes)
    
    # 加载权重
    if os.path.exists(config.model_path):
        ms.load_param_into_net(network, ms.load_checkpoint(config.model_path))
        print(f"模型加载成功: {config.model_path}")
    else:
        print(f"警告: 模型文件不存在: {config.model_path}")
        print("将使用随机初始化的模型进行演示...")
    
    network.set_train(False)
    return network


# =============================================================================
# 对抗攻击测试
# =============================================================================

def test_fgsm_attack(model, images, labels):
    """测试 FGSM 攻击"""
    attack = FGSM(model, epsilon=config.epsilon)
    
    # 转换为 numpy
    images_np = images.asnumpy()
    labels_np = labels.asnumpy()
    
    # 生成对抗样本
    adversarial = attack.generate(images_np, labels_np)
    
    # 计算原始准确率和攻击后准确率
    clean_output = model(Tensor(images_np))
    clean_pred = np.argmax(clean_output.asnumpy(), axis=1)
    clean_acc = np.mean(clean_pred == labels_np)
    
    adversarial_output = model(Tensor(adversarial.astype(np.float32)))
    adv_pred = np.argmax(adversarial_output.asnumpy(), axis=1)
    adv_acc = np.mean(adv_pred == labels_np)
    
    return clean_acc, adv_acc


def test_pgd_attack(model, images, labels):
    """测试 PGD 攻击"""
    pgd_attack = PGD(model, epsilon=config.epsilon, alpha=0.01, steps=10)
    
    # 转换为 numpy
    images_np = images.asnumpy()
    labels_np = labels.asnumpy()
    
    # 生成对抗样本
    adversarial = pgd_attack.generate(images_np, labels_np)
    
    # 计算准确率
    clean_output = model(Tensor(images_np))
    clean_pred = np.argmax(clean_output.asnumpy(), axis=1)
    clean_acc = np.mean(clean_pred == labels_np)
    
    adversarial_output = model(Tensor(adversarial.astype(np.float32)))
    adv_pred = np.argmax(adversarial_output.asnumpy(), axis=1)
    adv_acc = np.mean(adv_pred == labels_np)
    
    return clean_acc, adv_acc


# =============================================================================
# 数据集
# =============================================================================

def create_test_dataset():
    """创建测试数据集"""
    import mindspore.dataset as mds
    
    print("加载测试数据集...")
    
    try:
        dataset = mds.Cifar10Dataset(
            dataset_dir="./data/cifar10",
            usage="test",
            num_samples=config.num_samples
        )
        
        # 数据预处理
        transform = [
            mds.Resize((224, 224)),
            mds.Rescale(1.0 / 255.0, 0.0),
            mds.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            mds.HWC2CHW()
        ]
        
        dataset = dataset.map(transform, input_columns=["image"])
        dataset = dataset.batch(config.batch_size)
        
        return dataset
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("将使用随机数据进行演示...")
        return None


# =============================================================================
# 主评估流程
# =============================================================================

def evaluate():
    """主评估函数"""
    print("=" * 60)
    print("ResNet50 鲁棒性评估")
    print("=" * 60)
    
    # 加载模型
    model = load_model()
    
    # 获取数据集
    dataset = create_test_dataset()
    
    if dataset is None:
        # 使用随机数据进行演示
        print("\n使用随机数据进行演示...")
        
        # 生成随机测试数据
        images = np.random.randn(config.batch_size, 3, 224, 224).astype(np.float32)
        labels = np.random.randint(0, config.num_classes, config.batch_size)
        
        # 转换为 Tensor
        images_tensor = Tensor(images)
        labels_tensor = Tensor(labels)
        
        # 测试 FGSM
        print("\n" + "-" * 40)
        print("测试 FGSM 攻击")
        print("-" * 40)
        clean_acc, fgsm_acc = test_fgsm_attack(model, images_tensor, labels_tensor)
        print(f"干净样本准确率: {clean_acc:.4f}")
        print(f"FGSM 攻击后准确率: {fgsm_acc:.4f}")
        print(f"准确率下降: {(clean_acc - fgsm_acc):.4f}")
        
        # 测试 PGD
        print("\n" + "-" * 40)
        print("测试 PGD 攻击")
        print("-" * 40)
        clean_acc, pgd_acc = test_pgd_attack(model, images_tensor, labels_tensor)
        print(f"干净样本准确率: {clean_acc:.4f}")
        print(f"PGD 攻击后准确率: {pgd_acc:.4f}")
        print(f"准确率下降: {(clean_acc - pgd_acc):.4f}")
        
    else:
        # 遍历数据集进行评估
        print("\n开始评估...")
        
        total_clean = 0
        total_fgsm = 0
        total_pgd = 0
        num_batches = 0
        
        for data in dataset.create_dict_iterator():
            images = data["image"]
            labels = data["label"].asnumpy()
            
            # 转换为需要的格式
            images_tensor = Tensor(images.asnumpy())
            labels_tensor = Tensor(labels)
            
            # 测试 FGSM
            try:
                clean_acc, fgsm_acc = test_fgsm_attack(model, images_tensor, labels_tensor)
                _, pgd_acc = test_pgd_attack(model, images_tensor, labels_tensor)
                
                total_clean += clean_acc * len(labels)
                total_fgsm += fgsm_acc * len(labels)
                total_pgd += pgd_acc * len(labels)
                num_batches += 1
                
                print(f"批次 {num_batches}: 干净={clean_acc:.2%}, FGSM={fgsm_acc:.2%}, PGD={pgd_acc:.2%}")
            except Exception as e:
                print(f"批次处理出错: {e}")
                continue
        
        # 计算平均结果
        if num_batches > 0:
            num_samples = num_batches * config.batch_size
            
            print("\n" + "=" * 60)
            print("评估结果汇总")
            print("=" * 60)
            print(f"评估样本数: {num_samples}")
            print(f"干净样本准确率: {total_clean / num_samples:.2%}")
            print(f"FGSM 攻击后准确率: {total_fgsm / num_samples:.2%}")
            print(f"PGD 攻击后准确率: {total_pgd / num_samples:.2%}")
            print("=" * 60)
    
    # 保存结果
    print("\n评估完成！")
    
    # 生成报告
    report_path = os.path.join(os.path.dirname(__file__), "..", "current_run", "robustness_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ResNet50 鲁棒性评估报告\n")
        f.write("=" * 40 + "\n\n")
        f.write("评估配置:\n")
        f.write(f"  - 攻击方法: FGSM, PGD\n")
        f.write(f"  - epsilon: {config.epsilon}\n")
        f.write(f"  - 样本数: {config.num_samples}\n\n")
        f.write("提示: 打开 MCP 使用 evaluate_model_robustness() 获取更详细的分析\n")
    
    print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    evaluate()
