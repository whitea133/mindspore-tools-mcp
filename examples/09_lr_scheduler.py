#!/usr/bin/env python3
"""
示例 9：学习率调度器使用
========================

展示如何使用 MCP 工具配置学习率调度器。

MCP 工具：
- get_lr_scheduler: 获取学习率调度器配置
"""

from mindspore_tools_mcp import msutils_tools


def example_cosine_annealing():
    """余弦退火调度器"""
    print("=" * 60)
    print("示例 9.1: 余弦退火调度器")
    print("=" * 60)
    
    result = msutils_tools.get_lr_scheduler(
        scheduler_type="cosine_annealing",
        total_epochs=100,
        warmup_epochs=5,
        base_lr=0.001,
        min_lr=1e-6
    )
    
    print(f"\n调度器类型: {result['scheduler_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")
    print(f"基础学习率: {result['base_lr']}")
    print(f"总训练轮数: {result['total_epochs']}")
    
    print(f"\n学习率曲线关键点:")
    for point in result['lr_curve_points']:
        print(f"  Epoch {point['epoch']:3d}: LR = {point['lr']:.2e}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_step_lr():
    """阶梯衰减调度器"""
    print("\n" + "=" * 60)
    print("示例 9.2: 阶梯衰减调度器")
    print("=" * 60)
    
    result = msutils_tools.get_lr_scheduler(
        scheduler_type="step_lr",
        total_epochs=100,
        base_lr=0.01
    )
    
    print(f"\n调度器类型: {result['scheduler_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")
    
    print(f"\n学习率曲线关键点:")
    for point in result['lr_curve_points']:
        print(f"  Epoch {point['epoch']:3d}: LR = {point['lr']:.2e}")


def example_warmup_cosine():
    """带预热的余弦调度器"""
    print("\n" + "=" * 60)
    print("示例 9.3: 带预热的余弦调度器")
    print("=" * 60)
    
    result = msutils_tools.get_lr_scheduler(
        scheduler_type="warmup_cosine",
        total_epochs=100,
        warmup_epochs=10,
        base_lr=0.001
    )
    
    print(f"\n调度器类型: {result['scheduler_type']}")
    print(f"预热轮数: {result['config']['warmup_epochs']}")
    print(f"\n说明: {result['description']}")
    
    print(f"\n学习率曲线关键点:")
    for point in result['lr_curve_points']:
        print(f"  Epoch {point['epoch']:3d}: LR = {point['lr']:.2e}")


def example_polynomial_lr():
    """多项式衰减调度器"""
    print("\n" + "=" * 60)
    print("示例 9.4: 多项式衰减调度器")
    print("=" * 60)
    
    result = msutils_tools.get_lr_scheduler(
        scheduler_type="polynomial",
        total_epochs=100,
        base_lr=0.01
    )
    
    print(f"\n调度器类型: {result['scheduler_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")


def example_one_cycle():
    """One Cycle 策略"""
    print("\n" + "=" * 60)
    print("示例 9.5: One Cycle 学习策略")
    print("=" * 60)
    
    result = msutils_tools.get_lr_scheduler(
        scheduler_type="one_cycle",
        total_epochs=100,
        base_lr=0.001
    )
    
    print(f"\n调度器类型: {result['scheduler_type']}")
    print(f"最大学习率: {result['config']['max_lr']}")
    print(f"\n说明: {result['description']}")


def example_compare_schedulers():
    """对比不同调度器"""
    print("\n" + "=" * 60)
    print("示例 9.6: 对比不同学习率调度器")
    print("=" * 60)
    
    schedulers = ["cosine_annealing", "step_lr", "polynomial", "one_cycle", "warmup_cosine"]
    
    print("\n调度器对比表:")
    print("-" * 70)
    print(f"{'调度器':<20} {'特点':<25} {'适用场景'}")
    print("-" * 70)
    
    info = {
        "cosine_annealing": ("平滑衰减", "通用场景"),
        "step_lr": ("阶梯式", "需要明显阶段性调整"),
        "polynomial": ("多项式曲线", "需要快速衰减"),
        "one_cycle": ("先升后降", "快速收敛"),
        "warmup_cosine": ("预热+余弦", "大模型训练"),
    }
    
    for sched in schedulers:
        result = msutils_tools.get_lr_scheduler(sched, total_epochs=100, base_lr=0.001)
        i = info.get(sched, ("未知", "未知"))
        print(f"{sched:<20} {i[0]:<25} {i[1]}")


def main():
    """运行所有示例"""
    print("学习率调度器配置示例")
    print("=" * 60)
    
    example_cosine_annealing()
    example_step_lr()
    example_warmup_cosine()
    example_polynomial_lr()
    example_one_cycle()
    example_compare_schedulers()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
