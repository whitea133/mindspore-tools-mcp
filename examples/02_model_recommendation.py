#!/usr/bin/env python3
"""
示例 2：智能模型推荐
====================

展示如何使用 recommend_models 工具根据自然语言描述推荐合适的模型。

MCP 工具：
- recommend_models: 智能模型推荐
"""

from mindspore_tools_mcp import tools


def example_basic_recommendation():
    """基础推荐功能"""
    print("=" * 60)
    print("示例 2.1: 基础推荐 - 图像分类")
    print("=" * 60)
    
    result = tools.recommend_models("图像分类", limit=3)
    
    print(f"\n查询: {result['query']}")
    print(f"识别的任务: {result['interpreted']['tasks']}")
    print(f"识别的套件: {result['interpreted']['suite']}")
    print(f"\n找到 {result['total_found']} 个匹配模型，推荐 {len(result['recommendations'])} 个:")
    
    for i, rec in enumerate(result['recommendations'], 1):
        model = rec['model']
        print(f"\n  {i}. {model['name']} (分数: {rec['score']})")
        print(f"     套件: {model['suite']}")
        print(f"     任务: {', '.join(model['task'])}")
        for reason in rec['reasons'][:3]:
            print(f"     {reason}")


def example_recommendation_with_hardware():
    """带硬件约束的推荐"""
    print("\n" + "=" * 60)
    print("示例 2.2: 带硬件约束的推荐")
    print("=" * 60)
    
    hardware_options = ["ascend", "gpu", "cpu"]
    
    for hw in hardware_options:
        result = tools.recommend_models(
            "图像分类",
            hardware=hw,
            limit=2
        )
        print(f"\n【{hw.upper()}】约束下推荐:")
        for rec in result['recommendations']:
            print(f"  - {rec['model']['name']} (分数: {rec['score']})")


def example_text_generation():
    """文本生成模型推荐"""
    print("\n" + "=" * 60)
    print("示例 2.3: 文本生成模型推荐")
    print("=" * 60)
    
    queries = [
        "文本生成",
        "大语言模型",
        "对话系统",
        "机器翻译"
    ]
    
    for query in queries:
        result = tools.recommend_models(query, limit=2)
        print(f"\n【{query}】:")
        for rec in result['recommendations']:
            model = rec['model']
            print(f"  → {model['name']} ({model['suite']})")


def example_ocr_models():
    """OCR 模型推荐"""
    print("\n" + "=" * 60)
    print("示例 2.4: OCR 模型推荐")
    print("=" * 60)
    
    result = tools.recommend_models("文字识别 OCR", limit=5)
    
    print(f"\n查询: {result['query']}")
    print(f"识别的任务: {result['interpreted']['tasks']}")
    print(f"\n推荐模型:")
    
    for i, rec in enumerate(result['recommendations'], 1):
        model = rec['model']
        print(f"\n  {i}. {model['name']}")
        print(f"     分数: {rec['score']}")
        print(f"     任务: {', '.join(model['task'])}")


def example_suite_preference():
    """套件偏好推荐"""
    print("\n" + "=" * 60)
    print("示例 2.5: 指定套件的推荐")
    print("=" * 60)
    
    result = tools.recommend_models(
        "图像分类",
        suite="mindcv",
        limit=3
    )
    
    print(f"\n指定套件: mindcv")
    print(f"查询: {result['query']}")
    print(f"\n推荐模型:")
    for rec in result['recommendations']:
        print(f"  - {rec['model']['name']} (分数: {rec['score']})")


def main():
    """运行所有示例"""
    print("智能模型推荐示例")
    print("=" * 60)
    
    example_basic_recommendation()
    example_recommendation_with_hardware()
    example_text_generation()
    example_ocr_models()
    example_suite_preference()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
