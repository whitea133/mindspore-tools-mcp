#!/usr/bin/env python3
"""
示例 10：训练回调函数配置
===========================

展示如何使用 MCP 工具配置训练回调函数。

MCP 工具：
- get_training_callbacks: 获取训练回调配置
"""

from mindspore_tools_mcp import msutils_tools


def example_checkpoint_callback():
    """检查点保存回调"""
    print("=" * 60)
    print("示例 10.1: 检查点保存回调")
    print("=" * 60)
    
    result = msutils_tools.get_training_callbacks(
        callback_types=["checkpoint"]
    )
    
    print(f"\n配置回调类型:")
    for cb in result['callbacks']:
        print(f"  ✓ {cb['type']}")
        print(f"    配置: {cb['config']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_early_stopping():
    """早停回调"""
    print("\n" + "=" * 60)
    print("示例 10.2: 早停回调")
    print("=" * 60)
    
    result = msutils_tools.get_training_callbacks(
        callback_types=["early_stopping"],
        early_stopping_config={
            "monitor": "val_loss",
            "patience": 10,
            "min_delta": 0.001
        }
    )
    
    print(f"\n早停配置:")
    for cb in result['callbacks']:
        if cb['type'] == 'early_stopping':
            print(f"  监控指标: {cb['config']['monitor']}")
            print(f"  耐心轮数: {cb['config']['patience']}")
            print(f"  最小改善: {cb['config']['min_delta']}")


def example_tensorboard():
    """TensorBoard 日志回调"""
    print("\n" + "=" * 60)
    print("示例 10.3: TensorBoard 日志回调")
    print("=" * 60)
    
    result = msutils_tools.get_training_callbacks(
        callback_types=["tensorboard"]
    )
    
    print(f"\nTensorBoard 配置:")
    for cb in result['callbacks']:
        if cb['type'] == 'tensorboard':
            print(f"  日志目录: {cb['config']['log_dir']}")


def example_combined_callbacks():
    """组合多个回调"""
    print("\n" + "=" * 60)
    print("示例 10.4: 组合多个回调函数")
    print("=" * 60)
    
    result = msutils_tools.get_training_callbacks(
        callback_types=["checkpoint", "early_stopping", "tensorboard", "lr_monitor", "gradient_clip"]
    )
    
    print(f"\n配置回调类型 ({len(result['callbacks'])} 个):")
    for cb in result['callbacks']:
        print(f"  ✓ {cb['type']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_callbacks_explained():
    """回调函数说明"""
    print("\n" + "=" * 60)
    print("示例 10.5: 回调函数说明")
    print("=" * 60)
    
    callbacks_info = {
        "checkpoint": {
            "name": "模型检查点",
            "功能": "保存训练过程中的模型权重",
            "使用场景": "长时间训练，防止意外中断丢失进度"
        },
        "early_stopping": {
            "name": "早停",
            "功能": "监控验证集指标，连续无改善时停止训练",
            "使用场景": "防止过拟合，节省训练时间"
        },
        "tensorboard": {
            "name": "TensorBoard",
            "功能": "记录训练指标，可视化查看",
            "使用场景": "分析训练过程，调试模型"
        },
        "lr_monitor": {
            "name": "学习率监控",
            "功能": "记录学习率变化",
            "使用场景": "调试学习率调度器"
        },
        "gradient_clip": {
            "name": "梯度裁剪",
            "功能": "防止梯度爆炸",
            "使用场景": "RNN、Transformer 等容易梯度爆炸的模型"
        }
    }
    
    print("\n回调函数详解:")
    print("-" * 60)
    for name, info in callbacks_info.items():
        print(f"\n📌 {info['name']} ({name})")
        print(f"   功能: {info['功能']}")
        print(f"   场景: {info['使用场景']}")


def main():
    """运行所有示例"""
    print("训练回调函数配置示例")
    print("=" * 60)
    
    example_checkpoint_callback()
    example_early_stopping()
    example_tensorboard()
    example_combined_callbacks()
    example_callbacks_explained()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
