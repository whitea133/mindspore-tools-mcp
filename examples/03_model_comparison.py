#!/usr/bin/env python3
"""
示例 3：模型对比分析
====================

展示如何使用 compare_models 工具对比多个 MindSpore 模型。

MCP 工具：
- compare_models: 多模型对比
"""

from mindspore_tools_mcp import tools


def example_compare_image_models():
    """对比图像分类模型"""
    print("=" * 60)
    print("示例 3.1: 对比图像分类模型")
    print("=" * 60)
    
    models_to_compare = ["resnet50", "resnet101", "vit_base"]
    
    result = tools.compare_models(models_to_compare)
    
    print(f"\n对比模型: {', '.join(models_to_compare)}")
    print(f"找到 {len(result['models'])} 个模型\n")
    
    # 显示任务对比
    print("【任务对比】")
    for name, tasks in result['comparison']['tasks'].items():
        print(f"  {name}: {', '.join(tasks) if tasks else 'N/A'}")
    
    # 显示套件对比
    print("\n【套件对比】")
    for name, suite in result['comparison']['suites'].items():
        print(f"  {name}: {suite}")
    
    # 显示变体数量
    print("\n【变体数量】")
    for name, count in result['comparison']['variants_count'].items():
        print(f"  {name}: {count} 个变体")
    
    # 显示硬件支持
    print("\n【硬件支持】")
    for name, hw in result['comparison']['hardware'].items():
        if hw:
            hw_str = ", ".join([f"{k}: {v}" for k, v in hw.items() if v])
            print(f"  {name}: {hw_str}")
    
    # 显示推荐
    print(f"\n【选择建议】")
    print(f"  {result['recommendation']}")


def example_compare_llm_models():
    """对比大语言模型"""
    print("\n" + "=" * 60)
    print("示例 3.2: 对比大语言模型")
    print("=" * 60)
    
    llm_models = ["llama2_7b", "qwen1.5_7b", "glm3_6b"]
    
    try:
        result = tools.compare_models(llm_models)
        
        print(f"\n对比模型: {', '.join(llm_models)}")
        print(f"找到 {len(result['models'])} 个模型\n")
        
        for model in result['models']:
            print(f"\n【{model['name']}】")
            print(f"  套件: {model['suite']}")
            print(f"  任务: {', '.join(model['task'])}")
            metrics = model.get('metrics', {})
            if metrics:
                print(f"  指标: {metrics}")
                
    except Exception as e:
        print(f"对比出错: {e}")


def example_compare_cv_models():
    """对比计算机视觉模型"""
    print("\n" + "=" * 60)
    print("示例 3.3: 对比目标检测模型")
    print("=" * 60)
    
    # 先搜索目标检测模型
    detection_models = tools.list_models(task="object-detection")
    print(f"\n目标检测模型总数: {len(detection_models)}")
    
    # 取前3个对比
    model_ids = [m['id'] for m in detection_models[:3]]
    
    if model_ids:
        result = tools.compare_models(model_ids)
        
        print(f"\n对比模型: {', '.join(model_ids)}")
        print(f"找到 {len(result['models'])} 个模型\n")
        
        for model in result['models']:
            print(f"  • {model['name']} ({model['suite']})")


def example_interpret_result():
    """解读对比结果"""
    print("\n" + "=" * 60)
    print("示例 3.4: 解读对比结果")
    print("=" * 60)
    
    result = tools.compare_models(["resnet50", "vit_base"])
    
    print("\n对比结果解读:")
    print("-" * 40)
    
    # 解析推荐理由
    rec = result['recommendation']
    if "变体" in rec:
        print("📊 变体分析: 模型提供了多个变体选择")
    elif "性能基准" in rec:
        print("📈 性能分析: 有详细的性能数据支撑")
    elif "套件" in rec:
        print("🛠️ 套件分析: 同套件下有多种选择")
    else:
        print("💡 建议: 根据具体需求选择")
    
    print(f"\n推荐: {rec}")


def main():
    """运行所有示例"""
    print("模型对比分析示例")
    print("=" * 60)
    
    example_compare_image_models()
    example_compare_llm_models()
    example_compare_cv_models()
    example_interpret_result()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
