"""MCP 工具封装层 - 将 msutils 功能暴露为 MCP 工具。

此模块封装 msutils 库的核心功能，使其可以通过 MCP 协议访问。
"""

from __future__ import annotations

from typing import Any, Optional


# =============================================================================
# AI 安全工具
# =============================================================================

def generate_adversarial_attack(
    attack_type: str = "fgsm",
    epsilon: float = 0.1,
    num_iterations: int = 10,
    target_class: Optional[int] = None,
) -> dict[str, Any]:
    """生成对抗攻击配置和示例代码。

    支持 FGSM、PGD、DeepFool、CW 等主流攻击方法。

    Args:
        attack_type: 攻击类型
            - "fgsm": Fast Gradient Sign Method (快速梯度符号攻击)
            - "pgd": Projected Gradient Descent (投影梯度下降)
            - "deepfool": DeepFool 算法
            - "cw": Carlini-Wagner 攻击
            - "jsma": Jacobian-based Saliency Map Attack
        epsilon: 扰动幅度 (默认 0.1)
        num_iterations: 迭代次数 (PGD/CW 使用，默认 10)
        target_class: 目标类别 (定向攻击时使用)

    Returns:
        {
            "attack_type": "攻击类型",
            "config": {...攻击配置...},
            "code_example": "MindSpore 使用示例代码",
            "description": "攻击方法说明",
            "reference": "论文引用"
        }

    Examples:
        >>> generate_adversarial_attack("fgsm", epsilon=0.1)
        >>> generate_adversarial_attack("pgd", epsilon=0.3, num_iterations=40)
    """
    attack_configs = {
        "fgsm": {
            "config": {"epsilon": epsilon},
            "description": "FGSM (Fast Gradient Sign Method) 是一种单步攻击方法，"
                          "通过在梯度方向添加扰动来生成对抗样本。",
            "reference": "Goodfellow et al., 'Explaining and Harnessing Adversarial Examples', ICLR 2015",
        },
        "pgd": {
            "config": {
                "epsilon": epsilon,
                "alpha": epsilon / num_iterations,
                "num_iterations": num_iterations,
                "random_start": True,
            },
            "description": "PGD (Projected Gradient Descent) 是一种多步迭代攻击，"
                          "被认为是强一阶攻击的基准方法。",
            "reference": "Madry et al., 'Towards Deep Learning Models Resistant to Adversarial Attacks', ICLR 2018",
        },
        "deepfool": {
            "config": {
                "max_iterations": num_iterations,
                "overshoot": 0.02,
            },
            "description": "DeepFool 是一种高效的对抗样本生成算法，"
                          "可以找到最小扰动使样本跨越决策边界。",
            "reference": "Moosavi-Dezfooli et al., 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks', CVPR 2016",
        },
        "cw": {
            "config": {
                "c": 1.0,
                "kappa": 0,
                "learning_rate": 0.01,
                "num_iterations": num_iterations,
            },
            "description": "Carlini-Wagner 攻击是一种优化-based 攻击，"
                          "可以生成高成功率的对抗样本。",
            "reference": "Carlini & Wagner, 'Towards Evaluating the Robustness of Neural Networks', IEEE S&P 2017",
        },
        "jsma": {
            "config": {
                "theta": 1.0,
                "gamma": 0.1,
            },
            "description": "JSMA (Jacobian-based Saliency Map Attack) 基于"
                          "显著图选择性修改像素，生成稀疏扰动。",
            "reference": "Papernot et al., 'The Limitations of Deep Learning in Adversarial Settings', IEEE EuroS&P 2016",
        },
    }

    if attack_type.lower() not in attack_configs:
        return {
            "error": f"不支持的攻击类型: {attack_type}",
            "supported_types": list(attack_configs.keys()),
        }

    attack_info = attack_configs[attack_type.lower()]
    
    # 生成代码示例
    code_example = _generate_attack_code(attack_type.lower(), attack_info["config"], target_class)

    return {
        "attack_type": attack_type.lower(),
        "config": attack_info["config"],
        "code_example": code_example,
        "description": attack_info["description"],
        "reference": attack_info["reference"],
        "targeted": target_class is not None,
    }


def _generate_attack_code(attack_type: str, config: dict, target_class: Optional[int]) -> str:
    """生成攻击代码示例。"""
    if attack_type == "fgsm":
        return f'''import mindspore as ms
from mindspore import nn, ops
import numpy as np

def fgsm_attack(model, data, label, epsilon={config["epsilon"]}):
    """FGSM 对抗攻击"""
    grad_fn = ops.value_and_grad(
        lambda x: model(x).cross_entropy(label),
        grad_position=0
    )
    loss, grad = grad_fn(data)
    adversarial = data + epsilon * ops.sign(grad)
    return ops.clip_by_value(adversarial, 0.0, 1.0)

# 使用示例
# adversarial_data = fgsm_attack(model, original_data, label)
'''

    elif attack_type == "pgd":
        return f'''import mindspore as ms
from mindspore import nn, ops
import numpy as np

def pgd_attack(model, data, label, 
               epsilon={config["epsilon"]},
               alpha={config["alpha"]},
               num_iterations={config["num_iterations"]}):
    """PGD 对抗攻击"""
    adversarial = data.copy()
    if {config["random_start"]}:
        adversarial = adversarial + ops.uniform(adversarial.shape, -epsilon, epsilon)
    
    for _ in range(num_iterations):
        grad_fn = ops.value_and_grad(
            lambda x: model(x).cross_entropy(label),
            grad_position=0
        )
        loss, grad = grad_fn(adversarial)
        adversarial = adversarial + alpha * ops.sign(grad)
        # 投影到 epsilon 球内
        delta = ops.clip_by_value(adversarial - data, -epsilon, epsilon)
        adversarial = data + delta
    
    return ops.clip_by_value(adversarial, 0.0, 1.0)

# 使用示例
# adversarial_data = pgd_attack(model, original_data, label)
'''

    elif attack_type == "deepfool":
        return f'''import mindspore as ms
from mindspore import ops

def deepfool_attack(model, data, 
                    max_iterations={config["max_iterations"]},
                    overshoot={config["overshoot"]}):
    """DeepFool 对抗攻击"""
    adversarial = data.copy()
    
    for _ in range(max_iterations):
        output = model(adversarial)
        pred_class = ops.argmax(output)
        
        # 计算到最近边界的最小扰动
        # ... 详细实现请参考 msutils.security.attacks
        
    return adversarial

# 使用示例
# adversarial_data = deepfool_attack(model, original_data)
'''

    return f"# {attack_type} 攻击代码示例请参考 msutils 库文档"


def evaluate_model_robustness(
    model_info: dict[str, Any],
    attack_configs: Optional[list[dict[str, Any]]] = None,
    metrics: list[str] = ["accuracy", "success_rate", "perturbation_norm"],
) -> dict[str, Any]:
    """评估模型鲁棒性配置。

    Args:
        model_info: 模型配置
            - "model_name": 模型名称
            - "input_shape": 输入形状
            - "num_classes": 类别数
        attack_configs: 攻击配置列表 (可选，默认使用标准攻击集)
        metrics: 评估指标列表

    Returns:
        {
            "evaluation_config": {...评估配置...},
            "default_attacks": [...默认攻击列表...],
            "metrics": [...评估指标...],
            "code_example": "评估代码示例"
        }
    """
    default_attacks = attack_configs or [
        {"type": "fgsm", "epsilon": 0.1},
        {"type": "pgd", "epsilon": 0.3, "num_iterations": 40},
        {"type": "deepfool", "max_iterations": 50},
    ]

    code_example = '''import mindspore as ms
from msutils.security.evaluation import RobustnessEvaluator

# 创建评估器
evaluator = RobustnessEvaluator(
    model=model,
    attacks=['fgsm', 'pgd', 'deepfool'],
    metrics=['accuracy', 'success_rate', 'perturbation_norm']
)

# 运行评估
results = evaluator.evaluate(test_dataset)

print(f"干净准确率: {results['clean_accuracy']:.2%}")
print(f"FGSM 攻击后准确率: {results['fgsm_accuracy']:.2%}")
print(f"PGD 攻击后准确率: {results['pgd_accuracy']:.2%}")
'''

    return {
        "model_info": model_info,
        "default_attacks": default_attacks,
        "metrics": metrics,
        "code_example": code_example,
        "description": "鲁棒性评估用于测试模型在对抗攻击下的性能表现",
    }


# =============================================================================
# 数据处理工具
# =============================================================================

def create_data_augmentation_pipeline(
    task_type: str = "image_classification",
    augmentations: Optional[list[str]] = None,
    custom_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """创建数据增强流水线。

    Args:
        task_type: 任务类型
            - "image_classification": 图像分类
            - "object_detection": 目标检测
            - "semantic_segmentation": 语义分割
            - "nlp": 自然语言处理
        augmentations: 增强方法列表 (可选)
        custom_config: 自定义配置 (可选)

    Returns:
        {
            "task_type": "任务类型",
            "augmentations": [...增强方法...],
            "code_example": "MindSpore 使用代码",
            "description": "增强方法说明"
        }
    """
    # 默认增强配置
    default_augmentations = {
        "image_classification": [
            "RandomHorizontalFlip",
            "RandomCrop",
            "ColorJitter",
            "RandomRotation",
            "Normalize",
        ],
        "object_detection": [
            "RandomHorizontalFlip",
            "RandomResize",
            "RandomCrop",
            "Normalize",
        ],
        "semantic_segmentation": [
            "RandomHorizontalFlip",
            "RandomResize",
            "RandomCrop",
            "Normalize",
        ],
        "nlp": [
            "RandomDelete",
            "RandomSwap",
            "SynonymReplace",
        ],
    }

    selected_augs = augmentations or default_augmentations.get(task_type, [])

    code_example = f'''import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from msutils.data.augmentations import AugmentationPipeline

# 创建增强流水线
pipeline = AugmentationPipeline(
    task_type="{task_type}",
    augmentations={selected_augs}
)

# 应用到数据集
dataset = ds.ImageFolderDataset(data_path)
dataset = dataset.map(operations=pipeline, input_columns="image")

# 或者使用 MindSpore 原生 API
transform = [
    vision.RandomHorizontalFlip(),
    vision.RandomCrop(224),
    vision.ColorJitter(brightness=0.4, contrast=0.4),
    vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
'''

    return {
        "task_type": task_type,
        "augmentations": selected_augs,
        "custom_config": custom_config or {},
        "code_example": code_example,
        "description": f"为 {task_type} 任务配置数据增强流水线",
    }


# =============================================================================
# 训练工具
# =============================================================================

def get_lr_scheduler(
    scheduler_type: str = "cosine_annealing",
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    base_lr: float = 0.001,
    min_lr: float = 1e-6,
) -> dict[str, Any]:
    """获取学习率调度器配置。

    Args:
        scheduler_type: 调度器类型
            - "cosine_annealing": 余弦退火
            - "step_lr": 阶梯式衰减
            - "polynomial": 多项式衰减
            - "one_cycle": One Cycle 策略
            - "warmup_cosine": 带预热的余弦
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        base_lr: 基础学习率
        min_lr: 最小学习率

    Returns:
        {
            "scheduler_type": "调度器类型",
            "config": {...配置参数...},
            "code_example": "使用代码",
            "lr_curve_points": [...学习率曲线关键点...]
        }
    """
    scheduler_configs = {
        "cosine_annealing": {
            "description": "余弦退火调度器，学习率按余弦曲线衰减",
            "params": {"T_max": total_epochs, "eta_min": min_lr},
        },
        "step_lr": {
            "description": "阶梯式衰减，每隔固定轮数降低学习率",
            "params": {"step_size": 30, "gamma": 0.1},
        },
        "polynomial": {
            "description": "多项式衰减，学习率按多项式曲线下降",
            "params": {"total_iters": total_epochs, "power": 1.0},
        },
        "one_cycle": {
            "description": "One Cycle 策略，先升后降，训练更快收敛",
            "params": {"max_lr": base_lr * 10, "total_steps": total_epochs},
        },
        "warmup_cosine": {
            "description": "带预热的余弦退火，适合大模型训练",
            "params": {"warmup_epochs": warmup_epochs, "T_max": total_epochs},
        },
    }

    if scheduler_type.lower() not in scheduler_configs:
        return {
            "error": f"不支持的调度器类型: {scheduler_type}",
            "supported_types": list(scheduler_configs.keys()),
        }

    config = scheduler_configs[scheduler_type.lower()]

    code_example = f'''import mindspore as ms
from mindspore import nn
from msutils.train.schedulers import create_scheduler

# 创建学习率调度器
lr_scheduler = create_scheduler(
    scheduler_type="{scheduler_type}",
    base_lr={base_lr},
    total_epochs={total_epochs},
    warmup_epochs={warmup_epochs},
    min_lr={min_lr}
)

# 应用到优化器
optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr_scheduler)

# 或者使用 MindSpore 原生 API
from msutils.train.schedulers import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    base_lr={base_lr},
    T_max={total_epochs},
    eta_min={min_lr},
    warmup_epochs={warmup_epochs}
)
'''

    # 计算学习率曲线关键点
    lr_points = _compute_lr_curve_points(scheduler_type, base_lr, min_lr, total_epochs, warmup_epochs)

    return {
        "scheduler_type": scheduler_type.lower(),
        "config": config["params"],
        "description": config["description"],
        "base_lr": base_lr,
        "total_epochs": total_epochs,
        "code_example": code_example,
        "lr_curve_points": lr_points,
    }


def _compute_lr_curve_points(
    scheduler_type: str,
    base_lr: float,
    min_lr: float,
    total_epochs: int,
    warmup_epochs: int,
) -> list[dict[str, Any]]:
    """计算学习率曲线关键点。"""
    import math

    points = []
    key_epochs = [0, warmup_epochs, total_epochs // 4, total_epochs // 2, 
                  3 * total_epochs // 4, total_epochs]

    for epoch in key_epochs:
        if scheduler_type == "cosine_annealing":
            if epoch < warmup_epochs:
                lr = base_lr * epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
        elif scheduler_type == "step_lr":
            lr = base_lr * (0.1 ** (epoch // 30))
        else:
            lr = base_lr

        points.append({"epoch": epoch, "lr": round(lr, 6)})

    return points


def get_training_callbacks(
    callback_types: list[str] = ["checkpoint", "early_stopping", "tensorboard"],
    checkpoint_config: Optional[dict[str, Any]] = None,
    early_stopping_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """获取训练回调函数配置。

    Args:
        callback_types: 回调类型列表
            - "checkpoint": 模型检查点保存
            - "early_stopping": 早停机制
            - "tensorboard": TensorBoard 日志
            - "lr_monitor": 学习率监控
            - "gradient_clip": 梯度裁剪
        checkpoint_config: 检查点配置
        early_stopping_config: 早停配置

    Returns:
        {
            "callbacks": [...回调配置...],
            "code_example": "使用代码"
        }
    """
    default_checkpoint = checkpoint_config or {
        "save_checkpoint_steps": 1000,
        "keep_checkpoint_max": 5,
        "prefix": "model",
    }

    default_early_stopping = early_stopping_config or {
        "monitor": "val_loss",
        "patience": 10,
        "min_delta": 0.001,
    }

    code_example = '''import mindspore as ms
from mindspore.train import ModelCheckpoint, CheckpointConfig
from msutils.train.callbacks import (
    EarlyStoppingCallback,
    TensorBoardCallback,
    LrMonitorCallback,
    GradientClipCallback
)

# 配置检查点保存
config = CheckpointConfig(
    save_checkpoint_steps=1000,
    keep_checkpoint_max=5
)
checkpoint_cb = ModelCheckpoint(
    prefix="model",
    config=config,
    directory="./checkpoints"
)

# 配置早停
early_stop_cb = EarlyStoppingCallback(
    monitor="val_loss",
    patience=10,
    min_delta=0.001
)

# 配置 TensorBoard
tensorboard_cb = TensorBoardCallback(log_dir="./logs")

# 组装回调
callbacks = [checkpoint_cb, early_stop_cb, tensorboard_cb]

# 训练时使用
model.train(epoch, dataset, callbacks=callbacks)
'''

    callback_configs = []
    for cb_type in callback_types:
        if cb_type == "checkpoint":
            callback_configs.append({"type": "checkpoint", "config": default_checkpoint})
        elif cb_type == "early_stopping":
            callback_configs.append({"type": "early_stopping", "config": default_early_stopping})
        elif cb_type == "tensorboard":
            callback_configs.append({"type": "tensorboard", "config": {"log_dir": "./logs"}})
        elif cb_type == "lr_monitor":
            callback_configs.append({"type": "lr_monitor", "config": {}})
        elif cb_type == "gradient_clip":
            callback_configs.append({"type": "gradient_clip", "config": {"max_norm": 1.0}})

    return {
        "callbacks": callback_configs,
        "code_example": code_example,
        "description": "训练回调函数用于监控和控制训练过程",
    }


# =============================================================================
# 分析工具
# =============================================================================

def compute_model_complexity(
    model_name: str = "",
    input_shape: tuple = (1, 3, 224, 224),
    include_memory: bool = True,
) -> dict[str, Any]:
    """计算模型复杂度（FLOPs、参数量）。

    Args:
        model_name: 模型名称 (如 "resnet50", "vit_base")
        input_shape: 输入形状 (batch, channels, height, width)
        include_memory: 是否计算内存占用

    Returns:
        {
            "model_name": "模型名称",
            "flops": "FLOPs 数量",
            "params": "参数量",
            "memory_mb": "内存占用 (MB)",
            "code_example": "分析代码"
        }
    """
    # 预定义模型的复杂度数据
    model_complexity_db = {
        "resnet18": {"flops": "1.82G", "params": "11.7M", "memory_mb": 45},
        "resnet50": {"flops": "4.12G", "params": "25.6M", "memory_mb": 98},
        "resnet101": {"flops": "7.84G", "params": "44.5M", "memory_mb": 170},
        "vit_base": {"flops": "17.6G", "params": "86.6M", "memory_mb": 330},
        "vit_large": {"flops": "61.6G", "params": "307M", "memory_mb": 1170},
        "bert_base": {"flops": "11.2G", "params": "110M", "memory_mb": 420},
        "llama_7b": {"flops": "14.0T", "params": "7B", "memory_mb": 26600},
    }

    if model_name.lower() in model_complexity_db:
        data = model_complexity_db[model_name.lower()]
    else:
        data = {
            "flops": "请使用代码计算",
            "params": "请使用代码计算",
            "memory_mb": "请使用代码计算",
        }

    code_example = '''import mindspore as ms
from msutils.analysis.complexity import compute_flops, count_parameters

# 计算模型复杂度
model = create_model()  # 你的模型

# 计算 FLOPs
flops = compute_flops(model, input_shape=(1, 3, 224, 224))
print(f"FLOPs: {flops / 1e9:.2f}G")

# 计算参数量
params = count_parameters(model)
print(f"Parameters: {params / 1e6:.2f}M")

# 计算内存占用
memory = params * 4 / (1024 * 1024)  # float32
print(f"Memory: {memory:.2f} MB")
'''

    return {
        "model_name": model_name,
        "input_shape": input_shape,
        "flops": data["flops"],
        "params": data["params"],
        "memory_mb": data["memory_mb"] if include_memory else None,
        "code_example": code_example,
        "description": "模型复杂度分析帮助选择合适的模型和硬件配置",
    }


# =============================================================================
# 分布式训练工具
# =============================================================================

def setup_distributed_training(
    num_gpus: int = 8,
    backend: str = "nccl",
    sync_bn: bool = True,
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    """配置分布式训练。

    Args:
        num_gpus: GPU 数量
        backend: 通信后端 ("nccl", "gloo", "hccl")
        sync_bn: 是否使用同步 BN
        gradient_accumulation_steps: 梯度累积步数

    Returns:
        {
            "config": {...分布式配置...},
            "code_example": "配置代码",
            "launch_command": "启动命令"
        }
    """
    code_example = '''import mindspore as ms
from mindspore import nn, context
from mindspore.communication import init
from msutils.distributed.ddp import setup_ddp, DistributedSampler

# 初始化分布式环境
context.set_context(mode=context.GRAPH_MODE)
init()
rank_id = get_rank()
rank_size = get_group_size()

# 配置 DDP
model = setup_ddp(model, sync_bn=True)

# 配置数据采样器
sampler = DistributedSampler(
    dataset=train_dataset,
    num_replicas=rank_size,
    rank=rank_id,
    shuffle=True
)

# 配置梯度累积
if gradient_accumulation_steps > 1:
    from msutils.distributed.ddp import GradientAccumulation
    model = GradientAccumulation(model, accumulation_steps=gradient_accumulation_steps)
'''

    launch_command = f'''# 单机多卡启动
mpirun -n {num_gpus} python train.py --distributed

# 多机多卡启动
mpirun -n {num_gpus} -hostfile hostfile python train.py --distributed

# 使用 msrun 启动 (MindSpore 推荐)
msrun --worker_num={num_gpus} --local_worker_num={num_gpus} python train.py
'''

    return {
        "config": {
            "num_gpus": num_gpus,
            "backend": backend,
            "sync_bn": sync_bn,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        "code_example": code_example,
        "launch_command": launch_command,
        "description": f"配置 {num_gpus} 卡分布式训练",
    }


# =============================================================================
# 部署工具
# =============================================================================

def quantize_model(
    quantization_type: str = "dynamic",
    precision: str = "int8",
    calibration_dataset_size: int = 100,
) -> dict[str, Any]:
    """模型量化配置。

    Args:
        quantization_type: 量化类型
            - "dynamic": 动态量化
            - "static": 静态量化
            - "qat": 量化感知训练
        precision: 目标精度 ("int8", "int4", "fp16")
        calibration_dataset_size: 校准数据集大小 (静态量化)

    Returns:
        {
            "quantization_type": "量化类型",
            "config": {...量化配置...},
            "code_example": "量化代码",
            "expected_speedup": "预期加速比"
        }
    """
    quant_configs = {
        "dynamic": {
            "description": "动态量化，运行时动态计算量化参数",
            "expected_speedup": "2-3x 推理加速",
        },
        "static": {
            "description": "静态量化，使用校准数据集预先计算量化参数",
            "expected_speedup": "3-4x 推理加速",
        },
        "qat": {
            "description": "量化感知训练，训练时模拟量化效果，精度损失最小",
            "expected_speedup": "3-4x 推理加速",
        },
    }

    if quantization_type.lower() not in quant_configs:
        return {
            "error": f"不支持的量化类型: {quantization_type}",
            "supported_types": list(quant_configs.keys()),
        }

    config = quant_configs[quantization_type.lower()]

    code_example = f'''import mindspore as ms
from msutils.deploy.quantization import quantize_model, Calibrator

# 动态量化
if "{quantization_type}" == "dynamic":
    quantized_model = quantize_model(
        model,
        quant_type="dynamic",
        precision="{precision}"
    )

# 静态量化
elif "{quantization_type}" == "static":
    calibrator = Calibrator(
        model,
        calibration_data=calib_dataset,
        num_samples={calibration_dataset_size}
    )
    quantized_model = quantize_model(
        model,
        quant_type="static",
        precision="{precision}",
        calibrator=calibrator
    )

# 量化感知训练
elif "{quantization_type}" == "qat":
    from msutils.deploy.quantization import QuantizationAwareTraining
    qat = QuantizationAwareTraining(model, precision="{precision}")
    quantized_model = qat.train(train_dataset, epochs=10)

# 导出量化模型
ms.export(quantized_model, inputs, file_name="quantized_model", file_format="MINDIR")
'''

    return {
        "quantization_type": quantization_type.lower(),
        "precision": precision,
        "config": {
            "calibration_dataset_size": calibration_dataset_size if quantization_type == "static" else None,
        },
        "description": config["description"],
        "expected_speedup": config["expected_speedup"],
        "code_example": code_example,
    }


def convert_model_format(
    source_format: str = "pytorch",
    target_format: str = "mindspore",
    model_type: str = "transformer",
) -> dict[str, Any]:
    """模型格式转换配置。

    Args:
        source_format: 源格式 ("pytorch", "tensorflow", "onnx")
        target_format: 目标格式 ("mindspore", "mindir", "onnx")
        model_type: 模型类型 ("transformer", "cnn", "mlp")

    Returns:
        {
            "source_format": "源格式",
            "target_format": "目标格式",
            "code_example": "转换代码",
            "supported_layers": [...支持的层...]
        }
    """
    code_example = f'''import mindspore as ms
from msutils.deploy.conversion import ModelConverter

# 创建转换器
converter = ModelConverter(
    source_framework="{source_format}",
    target_framework="{target_format}",
    model_type="{model_type}"
)

# 转换模型
mindspore_model = converter.convert(pytorch_model)

# 或者加载 PyTorch 权重到 MindSpore 模型
from msutils.deploy.conversion import load_pytorch_weights
load_pytorch_weights(mindspore_model, "pytorch_weights.pth")

# 导出为 MindIR
ms.export(
    mindspore_model,
    inputs,
    file_name="model",
    file_format="MINDIR"
)
'''

    return {
        "source_format": source_format,
        "target_format": target_format,
        "model_type": model_type,
        "code_example": code_example,
        "description": f"从 {source_format} 转换到 {target_format}",
        "note": "转换后请验证模型输出是否一致",
    }
