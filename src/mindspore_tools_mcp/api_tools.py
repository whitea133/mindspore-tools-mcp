"""
MindSpore API 示例生成器 - MCP 工具封装
=====================================

提供 MindSpore 常用 API 的示例代码和使用说明。

主要工具:
- get_api_examples: 获取指定 API 的完整示例
- search_apis: 搜索相关 API
- list_api_categories: 列出所有 API 分类
- get_quick_api_reference: 快速 API 参考
"""

import json
from typing import Optional
from pydantic import BaseModel, Field

from .api_examples import get_api_examples as _get_api_examples
from .api_examples import search_apis as _search_apis
from .api_examples import get_related_apis as _get_related_apis
from .api_examples import get_api_categories, list_all_apis


# =============================================================================
# MCP 工具数据模型
# =============================================================================

class APIExampleRequest(BaseModel):
    api_name: str = Field(description="API 名称，如 'nn.Conv2d', 'dataset', 'Adam', 'ms.load_checkpoint'")
    language: str = Field(default="python", description="代码语言，目前仅支持 python")


class SearchAPIRequest(BaseModel):
    query: str = Field(description="搜索关键词，如 '卷积', 'loss', '优化器'")
    max_results: int = Field(default=10, description="最大返回数量")


class ListCategoriesRequest(BaseModel):
    pass


class GetRelatedRequest(BaseModel):
    api_name: str = Field(description="API 名称")


class QuickRefRequest(BaseModel):
    api_name: str = Field(description="API 名称")


# =============================================================================
# MCP 工具函数
# =============================================================================

def get_api_examples(api_name: str, language: str = "python") -> str:
    """
    获取 MindSpore API 的完整示例代码

    Args:
        api_name: API 名称，如 'nn.Conv2d', 'Adam', 'dataset'
        language: 代码语言（仅支持 python）

    Returns:
        JSON 格式的 API 详细信息，包含描述、签名、示例代码等

    Example:
        get_api_examples_tool("nn.Conv2d")
        get_api_examples_tool("dataset")
        get_api_examples_tool("Adam")
    """
    try:
        result = _get_api_examples(api_name, language)

        if result.get("status") == "not_found":
            output = f"""❌ 未找到 API: {api_name}

💡 建议:
"""
            if result.get("suggestions"):
                output += "可能你想找的 API:\n"
                for api in result["suggestions"]:
                    output += f"  • {api}\n"
            output += f"\n提示: {result.get('tip', '使用 list_api_categories() 查看所有分类')}"
            return output

        # 格式化输出
        info = result
        output = f"""# 📚 {info['api_name']}

## 描述
{info['description']}

## 签名
```python
{info['signature']}
```

## 参数说明
"""
        for param in info.get("parameters", []):
            output += f"- **{param['name']}** (`{param['type']}`): {param['desc']}\n"

        if not info.get("parameters"):
            output += "_无参数_\n"

        output += "\n## 示例代码\n"
        for i, example in enumerate(info.get("examples", []), 1):
            output += f"\n### 示例 {i}: {example['title']}\n"
            output += f"```python\n{example['code']}\n```\n"

        if info.get("related_apis"):
            output += "\n## 相关 API\n"
            for api in info["related_apis"]:
                output += f"- `{api}`\n"

        if info.get("official_doc"):
            output += f"\n## 官方文档\n{info['official_doc']}\n"

        return output

    except Exception as e:
        return f"❌ 获取 API 示例时出错: {str(e)}"


def search_apis(query: str, max_results: int = 10) -> str:
    """
    搜索 MindSpore 相关 API

    Args:
        query: 搜索关键词
        max_results: 最大返回数量

    Returns:
        匹配的 API 列表

    Example:
        search_apis_tool("卷积")
        search_apis_tool("loss", max_results=5)
    """
    try:
        results = _search_apis(query, max_results)

        if not results:
            return f"❌ 未找到匹配 '{query}' 的 API\n\n💡 尝试搜索: 'nn', 'dataset', 'optimizer', 'loss', 'train'"

        output = f"# 🔍 搜索结果: '{query}'\n\n共找到 **{len(results)}** 个相关 API:\n\n"

        for i, api_name in enumerate(results, 1):
            info = _get_api_examples(api_name)
            if info.get("status") == "found":
                cat_cn = info.get("category_cn", info.get("category", ""))
                output += f"{i}. **{api_name}** — {info.get('description', '')}\n"
                output += f"   分类: `{cat_cn}`\n"
                output += f"   示例数: {len(info.get('examples', []))} 个\n\n"
            else:
                output += f"{i}. {api_name}\n"

        return output

    except Exception as e:
        return f"❌ 搜索 API 时出错: {str(e)}"


def list_api_categories() -> str:
    """
    列出所有 API 分类

    Returns:
        所有 API 的分类统计

    Example:
        list_api_categories_tool()
    """
    try:
        categories = get_api_categories()
        total = sum(cat["count"] for cat in categories.values())

        output = f"# 📂 API 分类目录\n\n共 **{total}** 个 API，分布在 {len(categories)} 个分类:\n\n"

        for cat_id, cat_info in sorted(categories.items()):
            output += f"## {cat_info['name_cn']} ({cat_info['count']} 个)\n"
            for api in sorted(cat_info["apis"]):
                output += f"- `{api}`\n"
            output += "\n"

        output += "---\n💡 使用 `get_api_examples_tool` 获取某个 API 的详细信息\n"
        output += "💡 使用 `search_apis_tool` 搜索特定功能的 API\n"

        return output

    except Exception as e:
        return f"❌ 获取分类列表时出错: {str(e)}"


def get_related_apis(api_name: str) -> str:
    """
    获取某个 API 的相关 API

    Args:
        api_name: API 名称

    Returns:
        相关 API 列表

    Example:
        get_related_apis_tool("nn.Conv2d")
    """
    try:
        related = _get_related_apis(api_name)

        if not related:
            return f"⚠️ 未找到 {api_name} 的相关信息\n\n💡 请检查 API 名称是否正确"

        output = f"# 🔗 {api_name} 相关 API\n\n"

        for api in related:
            info = _get_api_examples(api)
            if info["status"] == "found":
                output += f"## `{api}`\n{info['description']}\n\n"
            else:
                output += f"- `{api}`\n"

        return output

    except Exception as e:
        return f"❌ 获取相关 API 时出错: {str(e)}"


def get_quick_reference(api_name: str) -> str:
    """
    获取 API 快速参考（简洁版本）

    Args:
        api_name: API 名称

    Returns:
        简洁的 API 参考信息

    Example:
        get_quick_reference_tool("nn.Conv2d")
    """
    try:
        result = _get_api_examples(api_name)

        if result.get("status") == "not_found":
            return f"❌ 未找到 API: {api_name}"

        info = result
        output = f"**{info['api_name']}** — {info['description']}**\n\n"
        output += f"```python\n{info['signature']}\n```\n"

        if info["examples"]:
            output += f"\n```python\n{info['examples'][0]['code']}\n```\n"

        return output

    except Exception as e:
        return f"❌ 获取快速参考时出错: {str(e)}"


# =============================================================================
# MCP 工具定义
# =============================================================================

TOOLS = [
    {
        "name": "get_api_examples",
        "description": "获取 MindSpore API 的完整示例代码和使用说明。支持所有常用 API（nn.Conv2d, nn.BatchNorm2d, dataset, Adam, nn.CrossEntropyLoss 等）。返回 API 描述、参数说明、多个示例代码、相关 API 和官方文档链接。当你需要了解某个 MindSpore API 怎么使用时，调用此工具。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_name": {
                    "type": "string",
                    "description": "API 名称，如 'nn.Conv2d', 'dataset', 'Adam', 'ms.load_checkpoint', 'nn.CrossEntropyLoss'"
                },
                "language": {
                    "type": "string",
                    "description": "代码语言，默认为 'python'"
                }
            },
            "required": ["api_name"]
        }
    },
    {
        "name": "search_apis",
        "description": "搜索 MindSpore 相关 API。当你需要找某个功能的 API 但不知道具体名称时使用，如搜索 '卷积' 会返回所有卷积相关 API。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如 '卷积', '池化', 'loss', '优化器', '数据集'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "最大返回数量，默认 10"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_api_categories",
        "description": "列出所有 MindSpore API 的分类目录。返回所有 API 按类别（神经网络层、数据处理、优化器、损失函数等）组织的完整列表。",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_related_apis",
        "description": "获取某个 API 的相关 API 列表。当你已经知道一个 API，想了解其他相关的 API 时使用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_name": {
                    "type": "string",
                    "description": "API 名称"
                }
            },
            "required": ["api_name"]
        }
    },
    {
        "name": "get_quick_reference",
        "description": "获取 MindSpore API 的快速参考（简洁版本）。只返回 API 描述、签名和一个最基础的示例代码，适合快速查阅。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_name": {
                    "type": "string",
                    "description": "API 名称"
                }
            },
            "required": ["api_name"]
        }
    }
]


def get_tool_handlers():
    """返回所有工具处理器"""
    return {
        "get_api_examples": get_api_examples,
        "search_apis": search_apis,
        "list_api_categories": list_api_categories,
        "get_related_apis": get_related_apis,
        "get_quick_reference": get_quick_reference,
    }
