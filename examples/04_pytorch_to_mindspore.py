#!/usr/bin/env python3
"""
示例 4：PyTorch → MindSpore 代码转换
=====================================

展示如何使用 MCP 工具进行 PyTorch 到 MindSpore 的代码转换。

MCP 工具：
- query_op_mapping: 查询 API 映射
- diagnose_translation: 诊断翻译完整性
"""

from mindspore_tools_mcp import tools


def example_query_single_api():
    """查询单个 API 映射"""
    print("=" * 60)
    print("示例 4.1: 查询单个 API 映射")
    print("=" * 60)
    
    # 查询 torch.add 的映射
    api = "torch.add"
    result = tools.query_op_mapping(api)
    
    print(f"\n查询 API: {api}")
    
    if result['consistent']:
        print(f"\n✅ 一致映射 ({len(result['consistent'])} 个):")
        for item in result['consistent'][:3]:
            print(f"  {item['pytorch']} → {item['mindspore']}")
            if item.get('description'):
                print(f"    说明: {item['description']}")
    
    if result['diff']:
        print(f"\n⚠️ 差异映射 ({len(result['diff'])} 个):")
        for item in result['diff'][:3]:
            print(f"  {item['pytorch']} → {item['mindspore']}")
            if item.get('description'):
                print(f"    说明: {item['description']}")


def example_query_common_apis():
    """查询常用 API 映射"""
    print("\n" + "=" * 60)
    print("示例 4.2: 查询常用 API 映射")
    print("=" * 60)
    
    common_apis = [
        "torch.nn.Linear",
        "torch.nn.Conv2d",
        "torch.relu",
        "torch.softmax",
        "torch.matmul",
        "torch.cat",
        "torch.mean",
        "torch.sum",
    ]
    
    print("\n常用 API 映射速查表:")
    print("-" * 50)
    
    for api in common_apis:
        result = tools.query_op_mapping(api)
        
        if result['consistent']:
            ms_api = result['consistent'][0]['mindspore']
            status = "✅"
        elif result['diff']:
            ms_api = result['diff'][0]['mindspore']
            status = "⚠️"
        else:
            ms_api = "未找到"
            status = "❌"
        
        print(f"{status} {api:25} → {ms_api}")


def example_diagnose_translation():
    """诊断翻译完整性"""
    print("\n" + "=" * 60)
    print("示例 4.3: 诊断翻译完整性")
    print("=" * 60)
    
    # PyTorch 原始代码
    py_code = '''
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = torch.softmax(x, dim=-1)
        return x
'''
    
    # 模拟的 MindSpore 翻译代码（有遗漏）
    ms_code = '''
import mindspore as ms
from mindspore import nn

class SimpleModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.linear = nn.Dense(128, 64)
        
    def construct(self, x):
        x = self.linear(x)
        x = ms.ops.relu(x)
        # 漏掉了 softmax
        return x
'''
    
    result = tools.diagnose_translation(py_code, ms_code)
    
    print("\n【诊断结果】")
    print("-" * 50)
    
    # 已应用的映射
    if result['applied_mappings']:
        print(f"\n✅ 已应用的映射 ({len(result['applied_mappings'])} 个):")
        for item in result['applied_mappings']:
            print(f"  {item['pytorch']} → {item['mindspore']}")
            if item.get('source_count') and item.get('translated_count'):
                if item['source_count'] > item['translated_count']:
                    print(f"    ⚠️ 源文件 {item['source_count']} 次，译文 {item['translated_count']} 次")
    
    # 缺失的映射
    if result['missing_mappings']:
        print(f"\n⚠️ 缺失的映射 ({len(result['missing_mappings'])} 个):")
        for item in result['missing_mappings']:
            print(f"  ❌ {item['pytorch']} → {item['mindspore']}")
    
    # 差异映射命中
    if result['diff_hits']:
        print(f"\n⚡ 差异映射需要人工检查 ({len(result['diff_hits'])} 个):")
        for item in result['diff_hits']:
            print(f"  ⚡ {item['pytorch']} → {item['mindspore']}")
            if item.get('shape_hint'):
                print(f"     提示: {item['shape_hint']}")
    
    # 标注后的代码
    if result['annotated']:
        print(f"\n📝 标注后的代码片段:")
        print("-" * 50)
        for line in result['annotated'].split('\n')[:20]:
            if 'TODO' in line:
                print(f"  🔴 {line.strip()}")


def example_translation_workflow():
    """完整翻译工作流"""
    print("\n" + "=" * 60)
    print("示例 4.4: 完整翻译工作流")
    print("=" * 60)
    
    # 1. 先查询需要用到的所有 API
    print("\n步骤 1: 查询所需 API 的映射")
    print("-" * 40)
    
    apis_to_check = ["torch.add", "torch.mul", "torch.cat", "torch.stack"]
    for api in apis_to_check:
        result = tools.query_op_mapping(api)
        if result['consistent']:
            print(f"  ✅ {api} → {result['consistent'][0]['mindspore']}")
        else:
            print(f"  ❌ {api} 未找到映射")
    
    # 2. 检查完整代码
    print("\n步骤 2: 使用 diagnose_translation 检查翻译")
    print("-" * 40)
    
    simple_py = "x = torch.add(a, b)\ny = torch.relu(x)"
    simple_ms = "x = a + b\ny = ms.ops.relu(x)"
    
    result = tools.diagnose_translation(simple_py, simple_ms)
    print(f"  缺失映射: {len(result['missing_mappings'])}")
    print(f"  已应用: {len(result['applied_mappings'])}")


def main():
    """运行所有示例"""
    print("PyTorch → MindSpore 代码转换示例")
    print("=" * 60)
    
    example_query_single_api()
    example_query_common_apis()
    example_diagnose_translation()
    example_translation_workflow()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
