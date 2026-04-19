"""
MindSpore 代码评分器 - 检查规则定义
=====================================

定义了四类检查规则：
1. 性能规则 (Performance)
2. 兼容性规则 (Compatibility)
3. 最佳实践规则 (Best Practices)
4. 可维护性规则 (Maintainability)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Rule:
    """检查规则"""
    id: str              # 规则 ID，如 "PERF001"
    name: str            # 规则名称
    description: str     # 规则描述
    severity: str        # "error", "warning", "info"
    category: str        # "performance", "compatibility", "best_practices", "maintainability"
    pattern: str | None  # 正则模式
    check_fn: Callable | None  # 自定义检查函数
    suggestion: str      # 改进建议
    examples: dict       # 错误示例和正确示例


# =============================================================================
# 性能规则
# =============================================================================

PERFORMANCE_RULES: list[Rule] = [
    Rule(
        id="PERF001",
        name="循环内创建优化器",
        description="优化器应在循环外创建，循环内重复创建会浪费内存",
        severity="error",
        category="performance",
        pattern=r"for\s+.*?:\s*.*?(?:torch\.optim\.\w+|mindspore\.nn\.\w*Optimizer)",
        check_fn=None,
        suggestion="将优化器创建移到训练循环外部",
        examples={
            "bad": "for epoch in range(10):\n    optimizer = nn.Adam(params)",
            "good": "optimizer = nn.Adam(params)\nfor epoch in range(10):\n    ..."
        }
    ),
    Rule(
        id="PERF002",
        name="循环内创建损失函数",
        description="损失函数应在循环外创建，重复创建没有意义",
        severity="warning",
        category="performance",
        pattern=r"for\s+.*?:\s*.*?(?:nn\.CrossEntropyLoss|nn\.MSELoss|nn\.BCELoss|Snn\.SmoothL1Loss)",
        check_fn=None,
        suggestion="将损失函数移到循环外部",
        examples={
            "bad": "for data in dataset:\n    loss_fn = nn.CrossEntropyLoss()",
            "good": "loss_fn = nn.CrossEntropyLoss()\nfor data in dataset:\n    ..."
        }
    ),
    Rule(
        id="PERF003",
        name="未使用数据增强缓存",
        description="数据增强在每个 epoch 都重新计算，建议使用缓存或生成器",
        severity="warning",
        category="performance",
        pattern=None,
        check_fn="check_data_augmentation",
        suggestion="考虑使用缓存的数据增强结果",
        examples={
            "bad": "for _ in range(epochs):\n    transforms = [...]",
            "good": "transforms = [...]\nfor _ in range(epochs):\n    ..."
        }
    ),
    Rule(
        id="PERF004",
        name="频繁 Tensor.cpu() 调用",
        description="频繁在 GPU 和 CPU 之间转移数据会影响性能",
        severity="warning",
        category="performance",
        pattern=r"\.cpu\(\)",
        check_fn=None,
        suggestion="尽量减少 .cpu() 调用，或使用 async=True",
        examples={
            "bad": "for x in data:\n    result = model(x).cpu()",
            "good": "results = model(data)\n# 批量转换"
        }
    ),
    Rule(
        id="PERF005",
        name="未启用混合精度训练",
        description="对于大模型，建议启用混合精度 (AMP) 以提高训练速度和节省显存",
        severity="info",
        category="performance",
        pattern=None,
        check_fn="check_amp_usage",
        suggestion="考虑使用 nn.build_train_network 或 context.set_auto_parallel_context",
        examples={
            "bad": "# 没有 AMP 配置",
            "good": "from mindspore.amp import DynamicLossScaleManager\nloss_scale_manager = DynamicLossScaleManager()"
        }
    ),
    Rule(
        id="PERF006",
        name="在计算图中执行 Python 控制流",
        description="使用 Python if/for while 等在 construct 中可能不支持动态控制流",
        severity="warning",
        category="performance",
        pattern=None,
        check_fn="check_control_flow",
        suggestion="使用 mindspore.nn.Cell 提供的控制流接口",
        examples={
            "bad": "def construct(self, x):\n    if x.shape[0] > 10: ...",
            "good": "# 使用 mindspore.ops 提供的算子处理条件逻辑"
        }
    ),
]

# =============================================================================
# 兼容性规则
# =============================================================================

COMPATIBILITY_RULES: list[Rule] = [
    Rule(
        id="COMP001",
        name="使用了 PyTorch 导入",
        description="检测到 'import torch'，这在纯 MindSpore 代码中不应该出现",
        severity="error",
        category="compatibility",
        pattern=r"^\s*import\s+torch\s*$|^\s*from\s+torch",
        check_fn=None,
        suggestion="使用 mindspore 替代 torch",
        examples={
            "bad": "import torch\nimport torch.nn as nn",
            "good": "import mindspore as ms\nfrom mindspore import nn"
        }
    ),
    Rule(
        id="COMP002",
        name="使用了 PyTorch 专属 API",
        description="代码中使用了 PyTorch 独有的 API",
        severity="error",
        category="compatibility",
        pattern=None,
        check_fn="check_torch_apis",
        suggestion="替换为 MindSpore 等价 API",
        examples={
            "bad": "torch.tensor([1, 2, 3])",
            "good": "ms.Tensor([1, 2, 3])"
        }
    ),
    Rule(
        id="COMP003",
        name="使用了 numpy 操作",
        description="在训练循环中使用 numpy 操作可能影响性能",
        severity="warning",
        category="compatibility",
        pattern=r"\.numpy\(\)",
        check_fn=None,
        suggestion="尽量使用 MindSpore Tensor 操作",
        examples={
            "bad": "x = x.numpy()\nx = x * 2\nx = ms.Tensor(x)",
            "good": "x = x * 2  # 直接用 Tensor 操作"
        }
    ),
    Rule(
        id="COMP004",
        name="使用了不推荐的 API",
        description="使用了已在新版 MindSpore 中废弃的 API",
        severity="warning",
        category="compatibility",
        pattern=None,
        check_fn="check_deprecated_apis",
        suggestion="请查阅 MindSpore 版本更新文档使用新版 API",
        examples={
            "bad": "# 使用了已废弃的回调函数",
            "good": "# 使用新版回调 API"
        }
    ),
    Rule(
        id="COMP005",
        name="Ascend 平台上性能较差的算子",
        description="某些算子在 Ascend NPU 上性能较差",
        severity="info",
        category="compatibility",
        pattern=None,
        check_fn="check_slow_npu_ops",
        suggestion="考虑使用 Ascend 优化过的替代算子",
        examples={
            "bad": "# 某些特殊操作在 NPU 上性能不佳",
            "good": "# 使用 mindspore.nn 提供的标准层"
        }
    ),
]

# =============================================================================
# 最佳实践规则
# =============================================================================

BEST_PRACTICE_RULES: list[Rule] = [
    Rule(
        id="BP001",
        name="未设置随机种子",
        description="未设置随机种子可能导致结果不可复现",
        severity="warning",
        category="best_practices",
        pattern=None,
        check_fn="check_seed_setting",
        suggestion="使用 mindspore.set_seed() 设置随机种子",
        examples={
            "bad": "# 没有设置种子",
            "good": "mindspore.set_seed(42)"
        }
    ),
    Rule(
        id="BP002",
        name="未设置学习率调度器",
        description="训练没有学习率调度，可能影响收敛",
        severity="info",
        category="best_practices",
        pattern=None,
        check_fn="check_lr_scheduler",
        suggestion="考虑使用学习率调度器如 CosineAnnealingLR",
        examples={
            "bad": "optimizer = nn.Adam(model.get_parameters())",
            "good": "scheduler = nn.cosine_decay_lr(...)\noptimizer = nn.Adam(model.get_parameters(), learning_rate=scheduler)"
        }
    ),
    Rule(
        id="BP003",
        name="未使用模型检查点",
        description="训练过程中没有保存检查点，意外中断会导致训练丢失",
        severity="warning",
        category="best_practices",
        pattern=None,
        check_fn="check_checkpoint",
        suggestion="使用 ModelCheckpoint 回调保存最佳模型",
        examples={
            "bad": "model.train(epoch, dataset)",
            "good": "ckpt_cb = ModelCheckpoint()\nmodel.train(epoch, dataset, callbacks=ckpt_cb)"
        }
    ),
    Rule(
        id="BP004",
        name="未设置梯度裁剪",
        description="大模型训练时建议使用梯度裁剪防止梯度爆炸",
        severity="info",
        category="best_practices",
        pattern=None,
        check_fn="check_gradient_clipping",
        suggestion="使用 ops.clip_by_norm 或 TrainOneStepCell with clip",
        examples={
            "bad": "# 没有梯度裁剪",
            "good": "grad_clip = ops.clip_by_norm(gradients, max_norm=1.0)"
        }
    ),
    Rule(
        id="BP005",
        name="未设置 mode",
        description="模型应在训练前设置为 train 模式，推理前设置为 eval 模式",
        severity="error",
        category="best_practices",
        pattern=None,
        check_fn="check_model_mode",
        suggestion="显式调用 model.set_train() 或 model.train()",
        examples={
            "bad": "output = model(input)",
            "good": "model.set_train(True)\noutput = model(input)"
        }
    ),
    Rule(
        id="BP006",
        name="未指定 pad_mode",
        description="Conv2d 等层未指定 pad_mode，在不同版本间可能行为不一致",
        severity="info",
        category="best_practices",
        pattern=r"nn\.Conv2d\([^)]*\)(?!\s*,\s*pad_mode)",
        check_fn=None,
        suggestion="显式指定 pad_mode='pad' 以确保行为一致",
        examples={
            "bad": "nn.Conv2d(3, 64, 3, padding=1)",
            "good": "nn.Conv2d(3, 64, 3, pad_mode='pad', padding=1)"
        }
    ),
    Rule(
        id="BP007",
        name="使用 view 而不是 reshape",
        description="view 要求数据是连续的，建议使用更灵活的 reshape",
        severity="info",
        category="best_practices",
        pattern=r"\.view\(",
        check_fn=None,
        suggestion="考虑使用 reshape() 替代 view()",
        examples={
            "bad": "x = x.view(batch, -1)",
            "good": "x = x.reshape(batch, -1)"
        }
    ),
]

# =============================================================================
# 可维护性规则
# =============================================================================

MAINTAINABILITY_RULES: list[Rule] = [
    Rule(
        id="MAINT001",
        name="函数过长",
        description=f"函数超过 200 行代码，难以维护",
        severity="warning",
        category="maintainability",
        pattern=None,
        check_fn="check_function_length",
        suggestion="将长函数拆分为多个小函数",
        examples={
            "bad": "def train():\n    # 500 行代码...",
            "good": "def train():\n    data = load_data()\n    model = build_model()\n    train_loop(data, model)"
        }
    ),
    Rule(
        id="MAINT002",
        name="重复代码",
        description="检测到相似的代码块，可能存在重复",
        severity="info",
        category="maintainability",
        pattern=None,
        check_fn="check_code_duplication",
        suggestion="考虑将重复代码提取为函数",
        examples={
            "bad": "x = x * 2\ny = y * 2\nz = z * 2",
            "good": "def multiply_by_two(x): return x * 2\nx, y, z = multiply_by_two(x), multiply_by_two(y), multiply_by_two(z)"
        }
    ),
    Rule(
        id="MAINT003",
        name="魔法数字",
        description="代码中使用了未命名的数字常量",
        severity="info",
        category="maintainability",
        pattern=r"(?<![A-Za-z_])(?:0x[a-fA-F0-9]+|\d+\.\d+|\d+)(?![A-Za-z0-9_])",
        check_fn=None,
        suggestion="将魔法数字定义为有意义的常量",
        examples={
            "bad": "for i in range(100):",
            "good": "EPOCHS = 100\nfor i in range(EPOCHS):"
        }
    ),
    Rule(
        id="MAINT004",
        name="缺少文档字符串",
        description="重要函数缺少文档字符串",
        severity="info",
        category="maintainability",
        pattern=None,
        check_fn="check_docstrings",
        suggestion="为函数和类添加 docstring",
        examples={
            "bad": "def train(): pass",
            "good": "def train():\n    \"\"\"训练模型的函数\"\"\"\n    pass"
        }
    ),
    Rule(
        id="MAINT005",
        name="变量命名不规范",
        description="变量命名不符合 Python 规范 (应使用 snake_case)",
        severity="info",
        category="maintainability",
        pattern=r"(?:^[A-Z][a-z]|[a-z][A-Z])",
        check_fn=None,
        suggestion="使用 snake_case 命名变量，如 user_name, learning_rate",
        examples={
            "bad": "userName = 1\nlearningRate = 0.001",
            "good": "user_name = 1\nlearning_rate = 0.001"
        }
    ),
]

# =============================================================================
# 所有规则汇总
# =============================================================================

ALL_RULES: list[Rule] = (
    PERFORMANCE_RULES +
    COMPATIBILITY_RULES +
    BEST_PRACTICE_RULES +
    MAINTAINABILITY_RULES
)


def get_rules_by_category(category: str) -> list[Rule]:
    """获取指定类别的规则"""
    return [r for r in ALL_RULES if r.category == category]


def get_rules_by_severity(severity: str) -> list[Rule]:
    """获取指定严重级别的规则"""
    return [r for r in ALL_RULES if r.severity == severity]


def get_rule_by_id(rule_id: str) -> Rule | None:
    """根据 ID 获取规则"""
    for rule in ALL_RULES:
        if rule.id == rule_id:
            return rule
    return None
