#!/usr/bin/env python3
"""
示例 1：MindSpore 模型发现
=========================

展示如何使用 mindspore-tools-mcp 发现和筛选 MindSpore 官方模型。

MCP 工具：
- list_models: 列出模型
- get_model_info: 获取模型详情
"""

from mindspore_tools_mcp import tools


def example_list_all_models():
    """列出所有可用模型"""
    print("=" * 60)
    print("示例 1.1: 列出所有模型")
    print("=" * 60)
    
    models = tools.list_models()
    print(f"\n总计找到 {len(models)} 个模型")
    print("\n前 5 个模型:")
    for i, model in enumerate(models[:5], 1):
        print(f"  {i}. {model.get('name', 'N/A')} ({model.get('id', 'N/A')})")


def example_filter_by_task():
    """按任务类型筛选模型"""
    print("\n" + "=" * 60)
    print("示例 1.2: 按任务类型筛选")
    print("=" * 60)
    
    # 图像分类模型
    image_models = tools.list_models(task="image-classification")
    print(f"\n图像分类模型 ({len(image_models)} 个):")
    for model in image_models[:3]:
        print(f"  - {model.get('name')} ({model.get('suite')})")
    
    # 文本生成模型
    text_models = tools.list_models(task="text-generation")
    print(f"\n文本生成模型 ({len(text_models)} 个):")
    for model in text_models[:3]:
        print(f"  - {model.get('name')} ({model.get('suite')})")


def example_filter_by_suite():
    """按套件筛选模型"""
    print("\n" + "=" * 60)
    print("示例 1.3: 按套件筛选")
    print("=" * 60)
    
    suites = ["mindcv", "mindformers", "mindocr", "mindspore"]
    
    for suite in suites:
        models = tools.list_models(suite=suite)
        if models:
            print(f"\n{suite} ({len(models)} 个模型):")
            for model in models[:3]:
                print(f"  - {model.get('name')}")
            if len(models) > 3:
                print(f"  ... 还有 {len(models) - 3} 个")


def example_search_by_keyword():
    """按关键词搜索模型"""
    print("\n" + "=" * 60)
    print("示例 1.4: 按关键词搜索")
    print("=" * 60)
    
    keywords = ["resnet", "bert", "yolo", "llama"]
    
    for keyword in keywords:
        models = tools.list_models(q=keyword)
        print(f"\n搜索 '{keyword}' 找到 {len(models)} 个模型:")
        for model in models[:3]:
            print(f"  - {model.get('name')} ({model.get('id')})")


def example_get_model_details():
    """获取模型详细信息"""
    print("\n" + "=" * 60)
    print("示例 1.5: 获取模型详细信息")
    print("=" * 60)
    
    # 获取特定模型详情
    model_ids = ["resnet50", "bert_base", "llama2_7b"]
    
    for model_id in model_ids:
        try:
            info = tools.get_model_info(model_id)
            print(f"\n模型: {info.get('name', 'N/A')}")
            print(f"  ID: {info.get('id', 'N/A')}")
            print(f"  套件: {info.get('suite', 'N/A')}")
            print(f"  任务: {', '.join(info.get('task', []))}")
            print(f"  类别: {info.get('category', 'N/A')}")
            
            # 显示指标
            metrics = info.get('metrics', {})
            if metrics:
                print(f"  性能指标:")
                for k, v in metrics.items():
                    print(f"    - {k}: {v}")
            
            # 显示变体
            variants = info.get('variants', [])
            if variants:
                print(f"  变体 ({len(variants)} 个): {', '.join(variants[:5])}")
                
        except ValueError as e:
            print(f"\n模型 '{model_id}' 未找到: {e}")


def main():
    """运行所有示例"""
    print("MindSpore 模型发现示例")
    print("=" * 60)
    
    example_list_all_models()
    example_filter_by_task()
    example_filter_by_suite()
    example_search_by_keyword()
    example_get_model_details()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
