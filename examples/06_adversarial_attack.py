#!/usr/bin/env python3
"""
示例 6：对抗攻击配置与生成
===========================

展示如何使用 MCP 工具生成 MindSpore 对抗攻击配置。

MCP 工具：
- generate_adversarial_attack: 生成对抗攻击配置
"""

from mindspore_tools_mcp import msutils_tools


def example_fgsm_attack():
    """FGSM 快速梯度符号攻击"""
    print("=" * 60)
    print("示例 6.1: FGSM 快速梯度符号攻击")
    print("=" * 60)
    
    result = msutils_tools.generate_adversarial_attack(
        attack_type="fgsm",
        epsilon=0.1
    )
    
    print(f"\n攻击类型: {result['attack_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")
    print(f"论文: {result['reference']}")
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_pgd_attack():
    """PGD 投影梯度下降攻击"""
    print("\n" + "=" * 60)
    print("示例 6.2: PGD 投影梯度下降攻击")
    print("=" * 60)
    
    result = msutils_tools.generate_adversarial_attack(
        attack_type="pgd",
        epsilon=0.3,
        num_iterations=40
    )
    
    print(f"\n攻击类型: {result['attack_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_deepfool_attack():
    """DeepFool 攻击"""
    print("\n" + "=" * 60)
    print("示例 6.3: DeepFool 攻击")
    print("=" * 60)
    
    result = msutils_tools.generate_adversarial_attack(
        attack_type="deepfool",
        num_iterations=50
    )
    
    print(f"\n攻击类型: {result['attack_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")
    print(f"\n代码示例:")
    print("-" * 40)
    print(result['code_example'])


def example_cw_attack():
    """Carlini-Wagner 攻击"""
    print("\n" + "=" * 60)
    print("示例 6.4: Carlini-Wagner 攻击")
    print("=" * 60)
    
    result = msutils_tools.generate_adversarial_attack(
        attack_type="cw",
        num_iterations=100
    )
    
    print(f"\n攻击类型: {result['attack_type']}")
    print(f"配置: {result['config']}")
    print(f"\n说明: {result['description']}")
    print(f"论文: {result['reference']}")


def example_targeted_attack():
    """定向攻击配置"""
    print("\n" + "=" * 60)
    print("示例 6.5: 定向攻击配置")
    print("=" * 60)
    
    result = msutils_tools.generate_adversarial_attack(
        attack_type="fgsm",
        epsilon=0.1,
        target_class=5  # 定向攻击到类别 5
    )
    
    print(f"\n攻击类型: {result['attack_type']}")
    print(f"目标类别: {result['target_class']}")
    print(f"是否为定向攻击: {result['targeted']}")


def example_compare_attacks():
    """对比不同攻击方法"""
    print("\n" + "=" * 60)
    print("示例 6.6: 对比不同攻击方法")
    print("=" * 60)
    
    attacks = ["fgsm", "pgd", "deepfool", "cw", "jsma"]
    
    print("\n攻击方法对比表:")
    print("-" * 60)
    print(f"{'方法':<12} {'复杂度':<10} {'攻击强度':<10} {'适用场景'}")
    print("-" * 60)
    
    attack_info = {
        "fgsm": ("低", "中", "快速攻击"),
        "pgd": ("中", "高", "最强单步攻击"),
        "deepfool": ("中", "高", "最小扰动"),
        "cw": ("高", "极高", "高质量对抗样本"),
        "jsma": ("高", "中", "稀疏扰动"),
    }
    
    for attack in attacks:
        result = msutils_tools.generate_adversarial_attack(attack_type=attack)
        info = attack_info.get(attack, ("未知", "未知", "未知"))
        print(f"{attack:<12} {info[0]:<10} {info[1]:<10} {info[2]}")


def main():
    """运行所有示例"""
    print("对抗攻击配置与生成示例")
    print("=" * 60)
    
    example_fgsm_attack()
    example_pgd_attack()
    example_deepfool_attack()
    example_cw_attack()
    example_targeted_attack()
    example_compare_attacks()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
