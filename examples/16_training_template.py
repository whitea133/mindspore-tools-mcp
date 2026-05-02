#!/usr/bin/env python3
"""
示例 16：训练模板生成器
======================

展示如何使用 MCP 工具生成 MindSpore 训练脚本。

MCP 工具：
- generate_training_template: 生成完整训练脚本
- get_available_options: 获取可用选项
- generate_quick_start: 快速入门脚本
"""

from mindspore_tools_mcp import template_tools


def example_basic_template():
    """生成基础训练脚本"""
    print("=" * 60)
    print("示例 16.1: 生成 ResNet50 训练脚本")
    print("=" * 60)
    
    result = template_tools.generate_training_template(
        task="image_classification",
        model="resnet50",
        dataset="cifar10",
        hardware="Ascend"
    )
    
    print(f"\n文件名: {result['filename']}")
    print(f"脚本行数: {len(result['script'].split(chr(10)))}")
    print(f"\n配置:")
    for key, value in result['config'].items():
        print(f"  - {key}: {value}")


def example_custom_config():
    """自定义配置的训练脚本"""
    print("\n" + "=" * 60)
    print("示例 16.2: 自定义配置")
    print("=" * 60)
    
    result = template_tools.generate_training_template(
        task="image_classification",
        model="resnet101",
        dataset="imagenet",
        hardware="Ascend",
        num_epochs=200,
        batch_size=256,
        base_lr=0.001,
        optimizer="adamw",
        lr_scheduler="cosine",
        use_amp=True,
    )
    
    print(f"\n文件名: {result['filename']}")
    print(f"\n配置:")
    for key, value in result['config'].items():
        print(f"  - {key}: {value}")
    
    print(f"\n脚本前 20 行预览:")
    print("-" * 40)
    for i, line in enumerate(result['script'].split('\n')[:20]):
        print(f"{line}")


def example_cpu_template():
    """生成 CPU 训练脚本"""
    print("\n" + "=" * 60)
    print("示例 16.3: CPU 训练脚本 (快速测试)")
    print("=" * 60)
    
    result = template_tools.generate_training_template(
        task="image_classification",
        model="lenet",
        dataset="cifar10",
        hardware="CPU",
        num_epochs=10,
        batch_size=64,
        use_amp=False,  # CPU 不支持 AMP
    )
    
    print(f"\n文件名: {result['filename']}")
    print(f"配置:")
    print(f"  - 硬件: {result['config']['hardware']}")
    print(f"  - 模型: {result['config']['model']}")
    print(f"  - 混合精度: {result['config']['use_amp']}")


def example_available_options():
    """查看可用选项"""
    print("\n" + "=" * 60)
    print("示例 16.4: 查看可用选项")
    print("=" * 60)
    
    options = template_tools.get_available_options()
    
    print("\n📋 可用模型:")
    for category, models in options['models'].items():
        print(f"  {category}: {', '.join(models)}")
    
    print("\n📋 可用数据集:")
    for name, info in options['datasets'].items():
        print(f"  {name}: {info['description']}")
        print(f"    类别数: {info['classes']}, 训练样本: {info['train_samples']}")
    
    print("\n📋 硬件平台:")
    for hw in options['hardware']:
        print(f"  - {hw}")
    
    print("\n📋 优化器:")
    for opt in options['optimizers']:
        print(f"  - {opt['name']}: {opt['description']}")


def example_quick_start():
    """快速入门脚本"""
    print("\n" + "=" * 60)
    print("示例 16.5: 快速入门脚本")
    print("=" * 60)
    
    levels = ["beginner", "intermediate", "advanced"]
    
    for level in levels:
        result = template_tools.generate_quick_start(level)
        print(f"\n【{level.upper()}】级别:")
        print(f"  文件: {result['filename']}")
        print(f"  描述: {result['description']}")
        print(f"  模型: {result['config']['model']}")
        print(f"  轮数: {result['config']['epochs']}")
        print(f"  硬件: {result['config']['hardware']}")


def example_preview():
    """预览模板"""
    print("\n" + "=" * 60)
    print("示例 16.6: 模板预览")
    print("=" * 60)
    
    preview = template_tools.preview_template(
        task="image_classification",
        model="resnet50"
    )
    
    print(f"\n文件名: {preview['filename']}")
    print(f"总行数: {preview['total_lines']}")
    print(f"\n脚本头部:")
    print("-" * 40)
    print(preview['header'])


def main():
    """运行所有示例"""
    print("训练模板生成器示例")
    print("=" * 60)
    
    example_basic_template()
    example_custom_config()
    example_cpu_template()
    example_available_options()
    example_quick_start()
    example_preview()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
