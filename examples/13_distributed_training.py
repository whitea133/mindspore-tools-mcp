#!/usr/bin/env python3
"""
示例 13：分布式训练配置
========================

展示如何使用 MCP 工具配置分布式训练。

MCP 工具：
- setup_distributed_training: 配置分布式训练
"""

from mindspore_tools_mcp import msutils_tools


def example_single_machine_multi_gpu():
    """单机多卡配置"""
    print("=" * 60)
    print("示例 13.1: 单机多卡配置")
    print("=" * 60)
    
    result = msutils_tools.setup_distributed_training(
        num_gpus=8,
        backend="nccl",
        sync_bn=True,
        gradient_accumulation_steps=1
    )
    
    print(f"\n配置:")
    print(f"  - GPU 数量: {result['config']['num_gpus']}")
    print(f"  - 通信后端: {result['config']['backend']}")
    print(f"  - 同步 BN: {result['config']['sync_bn']}")
    print(f"  - 梯度累积步数: {result['config']['gradient_accumulation_steps']}")
    
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])
    
    print(f"\n启动命令:")
    print("-" * 40)
    print(result['launch_command'])


def example_multi_machine():
    """多机多卡配置"""
    print("\n" + "=" * 60)
    print("示例 13.2: 多机多卡配置")
    print("=" * 60)
    
    result = msutils_tools.setup_distributed_training(
        num_gpus=32,  # 4 机 x 8 卡
        backend="nccl",
        sync_bn=True,
        gradient_accumulation_steps=4
    )
    
    print(f"\n配置:")
    print(f"  - 总 GPU 数: {result['config']['num_gpus']}")
    print(f"  - 梯度累积: {result['config']['gradient_accumulation_steps']} 步")
    
    print(f"\n启动命令:")
    print("-" * 40)
    print(result['launch_command'])


def example_hccl_backend():
    """华为昇腾后端配置"""
    print("\n" + "=" * 60)
    print("示例 13.3: 华为昇腾 (Ascend) 分布式训练")
    print("=" * 60)
    
    result = msutils_tools.setup_distributed_training(
        num_gpus=8,
        backend="hccl",  # Ascend 专用
        sync_bn=True,
        gradient_accumulation_steps=1
    )
    
    print(f"\n后端: HCCL (华为集合通信库)")
    print(f"适用硬件: Ascend NPU")
    print(f"\n说明: 专为华为昇腾处理器优化的分布式训练配置")


def example_workflow():
    """分布式训练完整工作流"""
    print("\n" + "=" * 60)
    print("示例 13.4: 分布式训练完整工作流")
    print("=" * 60)
    
    print("""
分布式训练完整流程:
====================

1. 环境准备
   - 确保多卡/NPU 可用
   - 安装分布式通信库 (NCCL/HCCL)

2. 数据并行配置
   - 使用 DistributedSampler
   - 配置 batch_size = 单卡batch × 卡数

3. 混合精度配置
   - 启用 FP16 加速
   - 配置 loss_scale

4. 梯度同步
   - AllReduce 汇总梯度
   - 同步 BN (sync_bn=True)

5. 启动训练
   - mpirun / msrun 启动
   - 配置 rank 和 size

6. 验证结果
   - 检查 loss 下降
   - 对比单卡性能
""")


def main():
    """运行所有示例"""
    print("分布式训练配置示例")
    print("=" * 60)
    
    example_single_machine_multi_gpu()
    example_multi_machine()
    example_hccl_backend()
    example_workflow()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
