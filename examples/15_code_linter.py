#!/usr/bin/env python3
"""
示例 15：MindSpore 代码评分器
=============================

展示如何使用代码评分器检查代码质量。

MCP 工具：
- lint_mindspore_code: 代码质量评分
- get_lint_rules: 获取检查规则
- compare_code_snippets: 对比代码质量
"""

from mindspore_tools_mcp import linter_tools
from mindspore_tools_mcp.linter import format_report


def example_bad_code():
    """检查有问题的代码"""
    print("=" * 60)
    print("示例 15.1: 检查有问题的代码")
    print("=" * 60)
    
    bad_code = """
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        
    def forward(self, x):
        for i in range(10):
            optimizer = nn.Adam(self.parameters())
        return x
"""
    
    result = linter_tools.lint_mindspore_code(bad_code)
    
    print(f"\n总分: {result['score']}/100 (Grade: {result['grade']})")
    print(f"\n各维度得分:")
    for dim, data in result['dimensions'].items():
        print(f"  {dim}: {data['score']}/100 ({data['issues_count']} issues)")
    
    print(f"\n发现问题:")
    for issue in result['issues']:
        icon = "🔴" if issue['severity'] == 'error' else "🟡" if issue['severity'] == 'warning' else "🔵"
        print(f"  {icon} [{issue['rule_id']}] 第{issue['line']}行: {issue['message']}")


def example_good_code():
    """检查良好代码"""
    print("\n" + "=" * 60)
    print("示例 15.2: 检查良好代码")
    print("=" * 60)
    
    good_code = """
import mindspore as ms
from mindspore import nn

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)
        
    def construct(self, x):
        return x

# 设置随机种子
ms.set_seed(42)
"""
    
    result = linter_tools.lint_mindspore_code(good_code)
    
    print(f"\n总分: {result['score']}/100 (Grade: {result['grade']})")
    print(f"\n各维度得分:")
    for dim, data in result['dimensions'].items():
        status = "✓" if data['issues_count'] == 0 else f"({data['issues_count']})"
        print(f"  {dim}: {data['score']}/100 {status}")


def example_pretty_report():
    """格式化报告输出"""
    print("\n" + "=" * 60)
    print("示例 15.3: 格式化报告")
    print("=" * 60)
    
    code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.linear(x)
"""
    
    result = linter_tools.lint_mindspore_code(code, style="pretty")
    print(result['formatted_report'])


def example_compare_code():
    """对比两个代码"""
    print("\n" + "=" * 60)
    print("示例 15.4: 对比代码质量")
    print("=" * 60)
    
    # 不好的写法
    bad = """
import torch
for i in range(10):
    model = torch.nn.Linear(10, 10)
"""
    
    # 好的写法
    good = """
import mindspore as ms
from mindspore import nn
model = nn.Dense(10, 10)
ms.set_seed(42)
"""
    
    result = linter_tools.compare_code_snippets(bad, good, labels=("bad", "good"))
    
    print(f"\n代码对比:")
    print(f"  bad 代码: {result['snippet_a']['score']}/100 (Grade: {result['snippet_a']['grade']})")
    print(f"  good 代码: {result['snippet_b']['score']}/100 (Grade: {result['snippet_b']['grade']})")
    print(f"\n结论: {result['comparison']}")


def example_get_rules():
    """获取检查规则"""
    print("\n" + "=" * 60)
    print("示例 15.5: 获取检查规则")
    print("=" * 60)
    
    # 获取所有规则
    all_rules = linter_tools.get_lint_rules()
    print(f"\n总规则数: {all_rules['total_rules']}")
    
    # 获取性能规则
    perf_rules = linter_tools.get_lint_rules(category="performance")
    print(f"\n性能规则 ({len(perf_rules['categories']['performance'])} 条):")
    for rule in perf_rules['categories']['performance']:
        print(f"  [{rule['id']}] {rule['name']}")
        print(f"    {rule['description']}")


def example_rules_reference():
    """规则速查表"""
    print("\n" + "=" * 60)
    print("示例 15.6: 规则速查表")
    print("=" * 60)
    
    # 获取所有规则
    result = linter_tools.get_lint_rules()
    
    print("\n📋 MindSpore 代码检查规则速查表")
    print("-" * 60)
    
    categories = {
        "performance": "⚡ 性能",
        "compatibility": "🔌 兼容性",
        "best_practices": "✨ 最佳实践",
        "maintainability": "🔧 可维护性",
    }
    
    for cat_key, cat_name in categories.items():
        rules = result['categories'].get(cat_key, [])
        print(f"\n{cat_name} ({len(rules)} 条)")
        print("-" * 40)
        for rule in rules:
            sev_icon = "🔴" if rule['severity'] == 'error' else "🟡"
            print(f"  {sev_icon} {rule['id']}: {rule['name']}")


def main():
    """运行所有示例"""
    print("MindSpore 代码评分器示例")
    print("=" * 60)
    
    example_bad_code()
    example_good_code()
    example_pretty_report()
    example_compare_code()
    example_get_rules()
    example_rules_reference()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
