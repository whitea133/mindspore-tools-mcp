"""
API 搜索模块
"""

from .registry import API_REGISTRY


def search_apis(query: str, max_results: int = 10) -> list:
    """
    搜索相关的 API

    Args:
        query: 搜索关键词
        max_results: 最大返回数量

    Returns:
        匹配的 API 列表
    """
    query = query.lower().strip()
    if not query:
        return []

    results = []
    for api_name, info in API_REGISTRY.items():
        # 检查 API 名称
        if query in api_name.lower():
            results.append((api_name, 100))
            continue

        # 检查描述
        desc = info.get("description", "").lower()
        if query in desc:
            results.append((api_name, 80))
            continue

        # 检查分类
        category = info.get("category", "").lower()
        category_cn = info.get("category_cn", "").lower()
        if query in category or query in category_cn:
            results.append((api_name, 60))
            continue

        # 检查相关 API
        related = [r.lower() for r in info.get("related", [])]
        if query in related:
            results.append((api_name, 50))
            continue

        # 检查签名中的参数
        signature = info.get("signature", "").lower()
        if query in signature:
            results.append((api_name, 40))
            continue

    # 按分数排序
    results.sort(key=lambda x: x[1], reverse=True)
    return [name for name, score in results[:max_results]]


def get_related_apis(api_name: str) -> list:
    """获取相关 API"""
    if api_name not in API_REGISTRY:
        # 尝试模糊匹配
        for name, info in API_REGISTRY.items():
            if api_name.lower() in name.lower():
                return info.get("related", [])
        return []

    return API_REGISTRY[api_name].get("related", [])


def get_api_categories() -> dict:
    """获取所有 API 分类"""
    categories = {}
    for api_name, info in API_REGISTRY.items():
        cat = info.get("category", "other")
        cat_cn = info.get("category_cn", "其他")

        if cat not in categories:
            categories[cat] = {
                "name_cn": cat_cn,
                "count": 0,
                "apis": []
            }

        categories[cat]["count"] += 1
        categories[cat]["apis"].append(api_name)

    return categories
