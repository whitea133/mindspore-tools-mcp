#!/usr/bin/env python3
"""
示例 8：数据增强流水线配置
===========================

展示如何使用 MCP 工具配置数据增强流水线。

MCP 工具：
- create_data_augmentation_pipeline: 创建数据增强流水线
"""

from mindspore_tools_mcp import msutils_tools


def example_image_classification_augmentation():
    """图像分类数据增强"""
    print("=" * 60)
    print("示例 8.1: 图像分类数据增强")
    print("=" * 60)
    
    result = msutils_tools.create_data_augmentation_pipeline(
        task_type="image_classification",
        augmentations=None  # 使用默认
    )
    
    print(f"\n任务类型: {result['task_type']}")
    print(f"\n增强方法 ({len(result['augmentations'])} 种):")
    for aug in result['augmentations']:
        print(f"  ✓ {aug}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_object_detection_augmentation():
    """目标检测数据增强"""
    print("\n" + "=" * 60)
    print("示例 8.2: 目标检测数据增强")
    print("=" * 60)
    
    result = msutils_tools.create_data_augmentation_pipeline(
        task_type="object_detection"
    )
    
    print(f"\n任务类型: {result['task_type']}")
    print(f"\n增强方法:")
    for aug in result['augmentations']:
        print(f"  ✓ {aug}")


def example_nlp_augmentation():
    """NLP 数据增强"""
    print("\n" + "=" * 60)
    print("示例 8.3: NLP 数据增强")
    print("=" * 60)
    
    result = msutils_tools.create_data_augmentation_pipeline(
        task_type="nlp"
    )
    
    print(f"\n任务类型: {result['task_type']}")
    print(f"\n增强方法:")
    for aug in result['augmentations']:
        print(f"  ✓ {aug}")


def example_custom_augmentation():
    """自定义数据增强"""
    print("\n" + "=" * 60)
    print("示例 8.4: 自定义数据增强")
    print("=" * 60)
    
    custom_augs = [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "ColorJitter",
        "RandomRotation",
        "RandomAffine",
        "GaussianBlur"
    ]
    
    result = msutils_tools.create_data_augmentation_pipeline(
        task_type="image_classification",
        augmentations=custom_augs,
        custom_config={
            "flip_prob": 0.5,
            "rotation_range": 15,
            "color_jitter_strength": 0.3
        }
    )
    
    print(f"\n自定义增强方法 ({len(result['augmentations'])} 种):")
    for aug in result['augmentations']:
        print(f"  ✓ {aug}")
    
    print(f"\n自定义配置:")
    for k, v in result['custom_config'].items():
        print(f"  - {k}: {v}")


def example_augmentation_comparison():
    """数据增强方法对比"""
    print("\n" + "=" * 60)
    print("示例 8.5: 数据增强方法对比")
    print("=" * 60)
    
    task_types = [
        "image_classification",
        "object_detection",
        "semantic_segmentation",
        "nlp"
    ]
    
    print("\n不同任务的数据增强配置:")
    print("-" * 60)
    
    for task in task_types:
        result = msutils_tools.create_data_augmentation_pipeline(task_type=task)
        print(f"\n【{task}】")
        for aug in result['augmentations'][:4]:
            print(f"  • {aug}")


def main():
    """运行所有示例"""
    print("数据增强流水线配置示例")
    print("=" * 60)
    
    example_image_classification_augmentation()
    example_object_detection_augmentation()
    example_nlp_augmentation()
    example_custom_augmentation()
    example_augmentation_comparison()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
