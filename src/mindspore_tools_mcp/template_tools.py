"""
MCP 工具 - 训练模板生成器
==========================

将训练模板生成器封装为 MCP 工具。
"""

from __future__ import annotations
from typing import Any, Optional

from mindspore_tools_mcp.templates.generator import (
    generate_training_script as _generate_script,
    list_available_models,
    list_available_datasets,
    get_template_preview,
)


def generate_training_template(
    task: str = "image_classification",
    model: str = "resnet50",
    dataset: str = "cifar10",
    hardware: str = "Ascend",
    num_epochs: int = 90,
    batch_size: int = 32,
    base_lr: float = 0.001,
    optimizer: str = "adam",
    lr_scheduler: str = "cosine",
    use_amp: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """生成 MindSpore 训练脚本模板

    根据任务类型、模型、数据集等参数，生成完整可运行的训练脚本。
    脚本包含数据加载、模型定义、训练配置、回调函数等完整代码。

    Args:
        task: 任务类型
            - "image_classification": 图像分类 (默认)
            - "object_detection": 目标检测
            - "nlp": NLP 任务
        model: 模型名称
            - "resnet18", "resnet34", "resnet50", "resnet101"
            - "lenet": LeNet
        dataset: 数据集名称
            - "cifar10", "cifar100", "imagenet"
        hardware: 硬件平台
            - "Ascend": 华为昇腾 NPU (默认)
            - "GPU": NVIDIA GPU
            - "CPU": CPU
        num_epochs: 训练轮数 (默认 90)
        batch_size: 批次大小 (默认 32)
        base_lr: 基础学习率 (默认 0.001)
        optimizer: 优化器类型
            - "adam": Adam 优化器 (默认)
            - "sgd": SGD 动量优化器
            - "adamw": AdamW 优化器
        lr_scheduler: 学习率调度器
            - "cosine": 余弦退火 (默认)
            - "step": 阶梯衰减
            - "none": 固定学习率
        use_amp: 是否使用混合精度训练 (默认 True，CPU 时自动关闭)
        seed: 随机种子 (默认 42)

    Returns:
        {
            "script": "完整训练脚本代码 (Python)",
            "filename": "train_image_classification_resnet50.py",
            "config": {
                "task": "image_classification",
                "model": "resnet50",
                "dataset": "cifar10",
                "hardware": "Ascend",
                "epochs": 90,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "lr_scheduler": "cosine",
                "use_amp": true
            },
            "description": "生成图像分类任务的完整训练脚本"
        }

    Examples:
        >>> # 生成 ResNet50 训练脚本
        >>> result = generate_training_template(
        ...     task="image_classification",
        ...     model="resnet50",
        ...     dataset="cifar10",
        ...     hardware="Ascend"
        ... )
        >>> print(f"Script saved to: {result['filename']}")
        >>> print(f"Total lines: {len(result['script'].split(chr(10)))}")

        >>> # 使用 SGD 优化器
        >>> result = generate_training_template(
        ...     model="resnet101",
        ...     optimizer="sgd",
        ...     num_epochs=200
        ... )
    """
    return _generate_script(
        task=task,
        model=model,
        dataset=dataset,
        hardware=hardware,
        num_epochs=num_epochs,
        batch_size=batch_size,
        base_lr=base_lr,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        use_amp=use_amp,
        seed=seed,
    )


def get_available_options() -> dict[str, Any]:
    """获取所有可用的模型、数据集和配置选项

    Returns:
        {
            "models": {
                "resnet": ["resnet18", "resnet34", "resnet50", "resnet101"],
                "custom": ["lenet", "simple_cnn"],
                "mobilenet": ["mobilenet_v2"],
            },
            "datasets": {
                "cifar10": {...},
                "cifar100": {...},
                "imagenet": {...}
            },
            "hardware": ["Ascend", "GPU", "CPU"],
            "optimizers": ["adam", "sgd", "adamw"],
            "lr_schedulers": ["cosine", "step", "none"]
        }

    Examples:
        >>> options = get_available_options()
        >>> print(f"Available models: {options['models']}")
    """
    return {
        "models": list_available_models(),
        "datasets": list_available_datasets(),
        "hardware": ["Ascend", "GPU", "CPU"],
        "optimizers": [
            {"name": "adam", "description": "Adam 优化器，适合大多数场景"},
            {"name": "sgd", "description": "SGD 动量优化器，适合大数据集"},
            {"name": "adamw", "description": "AdamW 优化器，带权重衰减正则化"},
        ],
        "lr_schedulers": [
            {"name": "cosine", "description": "余弦退火，平滑衰减"},
            {"name": "step", "description": "阶梯衰减，每隔固定轮数降低"},
            {"name": "none", "description": "固定学习率"},
        ],
        "tasks": [
            {"name": "image_classification", "description": "图像分类任务"},
            {"name": "object_detection", "description": "目标检测任务"},
            {"name": "nlp", "description": "NLP 任务"},
        ],
    }


def preview_template(
    task: str = "image_classification",
    model: str = "resnet50",
) -> dict[str, Any]:
    """预览训练模板的关键部分（不返回完整脚本）

    用于快速查看生成的脚本结构和关键代码片段。

    Args:
        task: 任务类型
        model: 模型名称

    Returns:
        {
            "header": "脚本头部信息",
            "key_parts": ["数据加载代码", "模型定义代码", "训练配置代码", "主函数代码"],
            "total_lines": 预计行数,
            "filename": "预期文件名"
        }

    Examples:
        >>> preview = preview_template("image_classification", "resnet50")
        >>> print(f"Preview: {preview['header']}")
    """
    return get_template_preview(task=task, model=model)


def generate_quick_start(
    level: str = "beginner",
) -> dict[str, Any]:
    """生成快速入门训练脚本

    Args:
        level: 入门级别
            - "beginner": 新手推荐（简单配置）
            - "intermediate": 中级（标准配置）
            - "advanced": 高级（完整配置）

    Returns:
        {
            "script": "训练脚本",
            "filename": "文件名",
            "config": "配置",
            "description": "说明"
        }

    Examples:
        >>> result = generate_quick_start("beginner")
        >>> print(f"Quick start script: {result['filename']}")
    """
    presets = {
        "beginner": {
            "task": "image_classification",
            "model": "lenet",
            "dataset": "cifar10",
            "hardware": "CPU",
            "num_epochs": 10,
            "batch_size": 64,
            "base_lr": 0.01,
            "optimizer": "sgd",
            "lr_scheduler": "none",
            "use_amp": False,
        },
        "intermediate": {
            "task": "image_classification",
            "model": "resnet50",
            "dataset": "cifar10",
            "hardware": "Ascend",
            "num_epochs": 90,
            "batch_size": 32,
            "base_lr": 0.001,
            "optimizer": "adam",
            "lr_scheduler": "cosine",
            "use_amp": True,
        },
        "advanced": {
            "task": "image_classification",
            "model": "resnet101",
            "dataset": "imagenet",
            "hardware": "Ascend",
            "num_epochs": 200,
            "batch_size": 256,
            "base_lr": 0.001,
            "optimizer": "adamw",
            "lr_scheduler": "cosine",
            "use_amp": True,
        },
    }

    config = presets.get(level, presets["beginner"])

    result = _generate_script(**config)

    level_descriptions = {
        "beginner": "适合新手入门，模型简单，训练快速",
        "intermediate": "标准配置，适合日常研究和实验",
        "advanced": "完整配置，适合大规模训练和竞赛",
    }

    result["description"] = level_descriptions.get(level, "")
    result["level"] = level

    return result
