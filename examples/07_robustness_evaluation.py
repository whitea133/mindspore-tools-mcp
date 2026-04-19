#!/usr/bin/env python3
"""
示例 7：模型鲁棒性评估
=======================

展示如何使用 MCP 工具评估模型对抗鲁棒性。

MCP 工具：
- evaluate_model_robustness: 评估模型鲁棒性
"""

from mindspore_tools_mcp import msutils_tools


def example_basic_evaluation():
    """基础评估配置"""
    print("=" * 60)
    print("示例 7.1: 基础鲁棒性评估配置")
    print("=" * 60)
    
    model_config = {
        "model_name": "resnet50",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000
    }
    
    result = msutils_tools.evaluate_model_robustness(model_config)
    
    print(f"\n模型配置:")
    print(f"  - 名称: {result['model_info']['model_name']}")
    print(f"  - 输入形状: {result['model_info']['input_shape']}")
    print(f"  - 类别数: {result['model_info']['num_classes']}")
    
    print(f"\n默认攻击集:")
    for attack in result['default_attacks']:
        print(f"  - {attack}")
    
    print(f"\n评估指标:")
    for metric in result['metrics']:
        print(f"  - {metric}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_custom_attacks():
    """自定义攻击集的评估"""
    print("\n" + "=" * 60)
    print("示例 7.2: 自定义攻击集的评估")
    print("=" * 60)
    
    model_config = {
        "model_name": "vit_base",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000
    }
    
    # 自定义攻击配置
    custom_attacks = [
        {"type": "fgsm", "epsilon": 0.05},
        {"type": "fgsm", "epsilon": 0.1},
        {"type": "fgsm", "epsilon": 0.2},
        {"type": "pgd", "epsilon": 0.3, "num_iterations": 20},
    ]
    
    result = msutils_tools.evaluate_model_robustness(
        model_info=model_config,
        attack_configs=custom_attacks,
        metrics=["accuracy", "success_rate", "perturbation_norm", "attack_success_rate"]
    )
    
    print(f"\n自定义攻击配置:")
    for attack in result['default_attacks']:
        print(f"  - {attack}")
    
    print(f"\n评估指标:")
    for metric in result['metrics']:
        print(f"  - {metric}")


def example_evaluation_metrics():
    """评估指标说明"""
    print("\n" + "=" * 60)
    print("示例 7.3: 评估指标说明")
    print("=" * 60)
    
    metrics_info = {
        "accuracy": "干净样本和对抗样本的分类准确率",
        "success_rate": "对抗攻击成功率",
        "perturbation_norm": "对抗扰动的 L2/Linf 范数",
        "attack_success_rate": "攻击成功将模型输出改变的比例",
        "robust_accuracy": "对抗样本上的准确率",
    }
    
    print("\n鲁棒性评估指标:")
    print("-" * 50)
    for metric, desc in metrics_info.items():
        print(f"  📊 {metric}")
        print(f"     {desc}\n")


def example_workflow():
    """完整评估工作流"""
    print("\n" + "=" * 60)
    print("示例 7.4: 完整评估工作流")
    print("=" * 60)
    
    print("""
完整评估流程:
=============

1. 准备模型和数据
   model = load_mindspore_model("resnet50")
   test_data = load_test_dataset()

2. 配置评估
   evaluation_config = msutils_tools.evaluate_model_robustness(...)
   
3. 运行评估
   results = evaluator.evaluate(test_data)

4. 分析结果
   - clean_accuracy: 干净样本准确率
   - adversarial_accuracy: 对抗样本准确率
   - robustness_score: 鲁棒性得分 = adversarial_accuracy / clean_accuracy
   - attack_analysis: 各攻击方法的攻击效果

5. 生成报告
   - 可视化对抗样本
   - 对比不同攻击的扰动大小
   - 分析模型的薄弱环节
""")


def main():
    """运行所有示例"""
    print("模型鲁棒性评估示例")
    print("=" * 60)
    
    example_basic_evaluation()
    example_custom_attacks()
    example_evaluation_metrics()
    example_workflow()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
