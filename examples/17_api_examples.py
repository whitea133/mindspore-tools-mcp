"""
示例 17: MindSpore API 示例生成器
=================================

演示如何使用 API 示例生成器快速查找 MindSpore API 的使用方法。

功能：
- get_api_examples: 获取 API 完整示例
- search_apis: 搜索相关 API
- list_api_categories: 列出所有分类
- get_related_apis: 获取相关 API
- get_quick_reference: 快速参考
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, "src")

from mindspore_tools_mcp.api_tools import (
    get_api_examples,
    search_apis,
    list_api_categories,
    get_related_apis,
    get_quick_reference,
)


def main():
    print("=" * 60)
    print("MindSpore API 示例生成器演示")
    print("=" * 60)

    # 1. 获取 Conv2d 的完整示例
    print("\n【1】获取 nn.Conv2d 完整示例")
    print("-" * 40)
    result = get_api_examples("nn.Conv2d")
    # 只打印前 800 字符
    print(result[:800] + "..." if len(result) > 800 else result)

    # 2. 搜索优化器相关 API
    print("\n【2】搜索 '优化器' 相关 API")
    print("-" * 40)
    result = search_apis("优化器", max_results=5)
    print(result)

    # 3. 列出所有分类
    print("\n【3】列出所有 API 分类")
    print("-" * 40)
    result = list_api_categories()
    print(result[:600] + "..." if len(result) > 600 else result)

    # 4. 获取相关 API
    print("\n【4】获取 nn.Conv2d 的相关 API")
    print("-" * 40)
    result = get_related_apis("nn.Conv2d")
    print(result)

    # 5. 快速参考
    print("\n【5】获取 Adam 优化器快速参考")
    print("-" * 40)
    result = get_quick_reference("Adam")
    print(result)

    # 6. 搜索数据集相关 API
    print("\n【6】搜索 '数据' 相关 API")
    print("-" * 40)
    result = search_apis("数据", max_results=5)
    print(result)

    # 7. 获取交叉熵损失示例
    print("\n【7】获取 nn.CrossEntropyLoss 示例")
    print("-" * 40)
    result = get_api_examples("nn.CrossEntropyLoss")
    print(result[:1000] + "..." if len(result) > 1000 else result)

    # 8. 搜索不存在的 API（测试提示）
    print("\n【8】搜索不存在的 API")
    print("-" * 40)
    result = get_api_examples("nn.SomeNonExistentAPI")
    print(result)


if __name__ == "__main__":
    main()
