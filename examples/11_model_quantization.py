#!/usr/bin/env python3
"""
示例 11：模型量化配置
=====================

展示如何使用 MCP 工具配置模型量化。

MCP 工具：
- quantize_model: 模型量化配置
"""

from mindspore_tools_mcp import msutils_tools


def example_dynamic_quantization():
    """动态量化"""
    print("=" * 60)
    print("示例 11.1: 动态量化")
    print("=" * 60)
    
    result = msutils_tools.quantize_model(
        quantization_type="dynamic",
        precision="int8"
    )
    
    print(f"\n量化类型: {result['quantization_type']}")
    print(f"目标精度: {result['precision']}")
    print(f"\n说明: {result['description']}")
    print(f"预期加速: {result['expected_speedup']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_static_quantization():
    """静态量化"""
    print("\n" + "=" * 60)
    print("示例 11.2: 静态量化")
    print("=" * 60)
    
    result = msutils_tools.quantize_model(
        quantization_type="static",
        precision="int8",
        calibration_dataset_size=100
    )
    
    print(f"\n量化类型: {result['quantization_type']}")
    print(f"目标精度: {result['precision']}")
    print(f"校准数据集大小: {result['config']['calibration_dataset_size']}")
    print(f"\n说明: {result['description']}")
    print(f"预期加速: {result['expected_speedup']}")


def example_qat():
    """量化感知训练"""
    print("\n" + "=" * 60)
    print("示例 11.3: 量化感知训练 (QAT)")
    print("=" * 60)
    
    result = msutils_tools.quantize_model(
        quantization_type="qat",
        precision="int8"
    )
    
    print(f"\n量化类型: {result['quantization_type']}")
    print(f"目标精度: {result['precision']}")
    print(f"\n说明: {result['description']}")
    print(f"预期加速: {result['expected_speedup']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_compare_quantization():
    """对比量化方法"""
    print("\n" + "=" * 60)
    print("示例 11.4: 对比量化方法")
    print("=" * 60)
    
    quant_types = ["dynamic", "static", "qat"]
    
    print("\n量化方法对比:")
    print("-" * 70)
    print(f"{'方法':<15} {'精度损失':<15} {'加速比':<15} {'使用难度'}")
    print("-" * 70)
    
    info = {
        "dynamic": ("低-中", "2-3x", "简单"),
        "static": ("中", "3-4x", "中等"),
        "qat": ("低", "3-4x", "较难"),
    }
    
    for qt in quant_types:
        result = msutils_tools.quantize_model(quantization_type=qt)
        i = info.get(qt, ("未知", "未知", "未知"))
        print(f"{qt:<15} {i[0]:<15} {result['expected_speedup']:<15} {i[2]}")


def main():
    """运行所有示例"""
    print("模型量化配置示例")
    print("=" * 60)
    
    example_dynamic_quantization()
    example_static_quantization()
    example_qat()
    example_compare_quantization()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
