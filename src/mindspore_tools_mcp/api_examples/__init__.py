"""
MindSpore API 示例生成器
========================

提供常用 MindSpore API 的示例代码和使用说明。

主要功能:
- get_api_examples: 获取指定 API 的示例代码
- search_apis: 搜索相关 API
- list_api_categories: 列出所有 API 分类
- get_api_reference: 获取 API 参考文档
"""

from .registry import API_REGISTRY
from .searcher import search_apis, get_related_apis, get_api_categories

__version__ = "1.0.0"
__all__ = [
    "API_REGISTRY",
    "search_apis",
    "get_related_apis",
    "get_api_categories",
    "get_api_examples",
    "list_all_apis",
]


def get_api_examples(api_name: str, language: str = "python") -> dict:
    """
    获取指定 API 的示例代码

    Args:
        api_name: API 名称（如 "nn.Conv2d", "dataset", "Adam"）
        language: 代码语言（仅支持 python）

    Returns:
        dict: 包含 api_name, description, examples 等信息
    """
    api_name = api_name.strip()

    # 精确匹配
    if api_name in API_REGISTRY:
        return _format_api_info(api_name, API_REGISTRY[api_name])

    # 模糊匹配
    api_name_lower = api_name.lower()
    for name, info in API_REGISTRY.items():
        if api_name_lower in name.lower() or name.lower() in api_name_lower:
            return _format_api_info(name, info)

    # 返回搜索结果
    results = search_apis(api_name)
    if results:
        return {
            "status": "not_found",
            "api_name": api_name,
            "message": f"未找到精确匹配的 API: {api_name}",
            "suggestions": results[:5],
            "tip": "尝试搜索更通用的关键词，如 'nn.Conv2d' → 'Conv2d' 或 'Conv'"
        }

    return {
        "status": "not_found",
        "api_name": api_name,
        "message": f"未找到 API: {api_name}",
        "suggestions": [],
        "tip": "使用 list_all_apis() 查看所有可用 API"
    }


def list_all_apis() -> dict:
    """列出所有可用的 API"""
    categories = get_api_categories()
    return {
        "status": "success",
        "total": len(API_REGISTRY),
        "categories": categories,
        "all_apis": list(API_REGISTRY.keys())
    }


def _format_api_info(name: str, info: dict) -> dict:
    """格式化 API 信息"""
    return {
        "status": "found",
        "api_name": name,
        "category": info.get("category", "other"),
        "category_cn": info.get("category_cn", ""),
        "description": info.get("description", ""),
        "signature": info.get("signature", ""),
        "examples": info.get("examples", []),
        "parameters": info.get("parameters", []),
        "related_apis": info.get("related", []),
        "official_doc": info.get("official_doc", ""),
    }
