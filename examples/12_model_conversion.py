#!/usr/bin/env python3
"""
示例 12：模型格式转换
=====================

展示如何使用 MCP 工具配置模型格式转换。

MCP 工具：
- convert_model_format: 模型格式转换
"""

from mindspore_tools_mcp import msutils_tools


def example_pytorch_to_mindspore():
    """PyTorch → MindSpore 转换"""
    print("=" * 60)
    print("示例 12.1: PyTorch → MindSpore 转换")
    print("=" * 60)
    
    result = msutils_tools.convert_model_format(
        source_format="pytorch",
        target_format="mindspore",
        model_type="transformer"
    )
    
    print(f"\n源格式: {result['source_format']}")
    print(f"目标格式: {result['target_format']}")
    print(f"模型类型: {result['model_type']}")
    print(f"\n说明: {result['description']}")
    print(f"\n注意: {result['note']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_tensorflow_to_mindspore():
    """TensorFlow → MindSpore 转换"""
    print("\n" + "=" * 60)
    print("示例 12.2: TensorFlow → MindSpore 转换")
    print("=" * 60)
    
    result = msutils_tools.convert_model_format(
        source_format="tensorflow",
        target_format="mindspore",
        model_type="cnn"
    )
    
    print(f"\n源格式: {result['source_format']}")
    print(f"目标格式: {result['target_format']}")
    print(f"模型类型: {result['model_type']}")
    print(f"\n说明: {result['description']}")


def example_onnx_conversion():
    """ONNX 格式转换"""
    print("\n" + "=" * 60)
    print("示例 12.3: ONNX 格式转换")
    print("=" * 60)
    
    result = msutils_tools.convert_model_format(
        source_format="pytorch",
        target_format="onnx"
    )
    
    print(f"\n源格式: {result['source_format']}")
    print(f"目标格式: {result['target_format']}")


def example_export_to_mindir():
    """导出为 MindIR 格式"""
    print("\n" + "=" * 60)
    print("示例 12.4: 导出为 MindIR 格式")
    print("=" * 60)
    
    print("""
MindIR 是 MindSpore 的模型存储格式，支持以下功能：
  ✓ 算子粒度可溯源
  ✓ 支持多后端部署
  ✓ 支持模型加密

导出步骤:
1. 加载模型
2. 准备输入张量
3. 调用 ms.export()
4. 部署到目标平台
""")


def main():
    """运行所有示例"""
    print("模型格式转换示例")
    print("=" * 60)
    
    example_pytorch_to_mindspore()
    example_tensorflow_to_mindspore()
    example_onnx_conversion()
    example_export_to_mindir()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
