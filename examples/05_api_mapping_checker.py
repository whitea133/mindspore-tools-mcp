#!/usr/bin/env python3
"""
示例 5：API 映射查询与验证
==========================

展示如何使用 query_op_mapping 工具查询和验证 API 映射。

MCP 工具：
- query_op_mapping: 查询 API 映射
"""

from mindspore_tools_mcp import tools


def example_query_by_section():
    """按 section 查询映射"""
    print("=" * 60)
    print("示例 5.1: 按 section 查询 API 映射")
    print("=" * 60)
    
    sections = ["torch", "torchvision", "torch.nn"]
    
    for section in sections:
        result = tools.query_op_mapping("conv", section=section)
        total = len(result['consistent']) + len(result['diff'])
        print(f"\n【{section}】找到 {total} 个相关映射")
        if result['consistent']:
            print(f"  一致映射: {len(result['consistent'])} 个")
        if result['diff']:
            print(f"  差异映射: {len(result['diff'])} 个")


def example_fuzzy_search():
    """模糊搜索映射"""
    print("\n" + "=" * 60)
    print("示例 5.2: 模糊搜索 API 映射")
    print("=" * 60)
    
    keywords = ["relu", "sigmoid", "softmax", "maxpool", "avgpool"]
    
    print("\n激活函数和池化映射:")
    print("-" * 60)
    
    for kw in keywords:
        result = tools.query_op_mapping(kw)
        if result['consistent']:
            for item in result['consistent'][:2]:
                print(f"  ✅ {item['pytorch']} → {item['mindspore']}")


def example_layer_mapping():
    """层结构映射"""
    print("\n" + "=" * 60)
    print("示例 5.3: 神经网络层 API 映射")
    print("=" * 60)
    
    layers = [
        "nn.Conv2d", "nn.Linear", "nn.BatchNorm2d",
        "nn.Dropout", "nn.LayerNorm", "nn.LSTM"
    ]
    
    print("\n层结构映射表:")
    print("-" * 60)
    
    for layer in layers:
        result = tools.query_op_mapping(layer)
        if result['consistent']:
            ms_api = result['consistent'][0]['mindspore']
            print(f"  PyTorch: {layer:<20} → MindSpore: {ms_api}")
        elif result['diff']:
            ms_api = result['diff'][0]['mindspore']
            print(f"  PyTorch: {layer:<20} → MindSpore: {ms_api} ⚠️")


def main():
    """运行所有示例"""
    print("API 映射查询与验证示例")
    print("=" * 60)
    
    example_query_by_section()
    example_fuzzy_search()
    example_layer_mapping()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
