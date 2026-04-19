#!/usr/bin/env python3
"""
示例 14：模型复杂度分析
========================

展示如何使用 MCP 工具分析模型复杂度。

MCP 工具：
- compute_model_complexity: 计算模型复杂度
"""

from mindspore_tools_mcp import msutils_tools


def example_resnet_complexity():
    """ResNet 模型复杂度"""
    print("=" * 60)
    print("示例 14.1: ResNet 模型复杂度分析")
    print("=" * 60)
    
    models = ["resnet18", "resnet50", "resnet101"]
    
    print("\n模型复杂度对比:")
    print("-" * 60)
    print(f"{'模型':<15} {'FLOPs':<15} {'参数量':<15} {'内存 (MB)':<15}")
    print("-" * 60)
    
    for model in models:
        result = msutils_tools.compute_model_complexity(model)
        print(f"{model:<15} {result['flops']:<15} {result['params']:<15} {result['memory_mb']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_vit_complexity():
    """Vision Transformer 复杂度"""
    print("\n" + "=" * 60)
    print("示例 14.2: Vision Transformer 复杂度分析")
    print("=" * 60)
    
    models = ["vit_base", "vit_large"]
    
    print("\nViT 模型复杂度:")
    print("-" * 60)
    
    for model in models:
        result = msutils_tools.compute_model_complexity(model)
        print(f"\n【{model}】")
        print(f"  FLOPs: {result['flops']}")
        print(f"  参数量: {result['params']}")
        print(f"  内存占用: {result['memory_mb']} MB")


def example_custom_model():
    """自定义模型复杂度计算"""
    print("\n" + "=" * 60)
    print("示例 14.3: 自定义模型复杂度计算")
    print("=" * 60)
    
    result = msutils_tools.compute_model_complexity(
        model_name="custom_model",
        input_shape=(1, 3, 224, 224),
        include_memory=True
    )
    
    print(f"\n输入形状: {result['input_shape']}")
    print(f"\n说明: {result['description']}")
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_complexity_analysis():
    """复杂度分析指南"""
    print("\n" + "=" * 60)
    print("示例 14.4: 模型复杂度分析指南")
    print("=" * 60)
    
    print("""
复杂度分析指标:
===============

1. FLOPs (浮点运算数)
   - 衡量计算复杂度
   - 决定推理速度
   - 推理速度 ≈ 1 / FLOPs

2. 参数量
   - 模型存储大小
   - 内存占用
   - 训练显存需求

3. 内存占用
   - 运行时内存
   - 推理时显存
   - 训练时显存 ≈ 参数量 × 4 × 3 (梯度+优化器+ activations)

选择建议:
=========
- 边缘部署: 选 FLOPs < 1G 的模型
- 移动端: 选 FLOPs < 10G 的模型
- 服务器: 可选更大模型
""")


def main():
    """运行所有示例"""
    print("模型复杂度分析示例")
    print("=" * 60)
    
    example_resnet_complexity()
    example_vit_complexity()
    example_custom_model()
    example_complexity_analysis()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
