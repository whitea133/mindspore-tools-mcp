"""
MCP 工具 - MindSpore 代码评分器
===================================

将代码评分器封装为 MCP 工具。
"""

from __future__ import annotations
from typing import Any, Optional

from mindspore_tools_mcp.linter import (
    check_code as _check_code,
    format_report,
    CodeChecker,
    PERFORMANCE_RULES,
    COMPATIBILITY_RULES,
    BEST_PRACTICE_RULES,
    MAINTAINABILITY_RULES,
)


def lint_mindspore_code(
    code: str,
    level: str = "all",
    style: str = "pretty"
) -> dict[str, Any]:
    """MindSpore 代码评分器 - 检查代码质量

    Args:
        code: MindSpore 代码字符串
        level: 检查级别
            - "all": 检查所有规则 (默认)
            - "strict": 只检查 error 和 warning
            - "quick": 只检查关键规则
        style: 输出风格
            - "pretty": 彩色终端输出 (默认)
            - "simple": 简单文本
            - "json": JSON 格式
            - "markdown": Markdown 格式

    Returns:
        {
            "score": 85,           # 总分 0-100
            "grade": "B",          # 等级 A/B/C/D/F
            "dimensions": {           # 各维度得分
                "performance": {"score": 90, "issues_count": 0},
                "compatibility": {"score": 85, "issues_count": 1},
                "best_practices": {"score": 80, "issues_count": 2},
                "maintainability": {"score": 85, "issues_count": 0},
            },
            "formatted_report": "...",  # 格式化后的报告 (可选)
            "issues": [            # 问题列表
                {
                    "rule_id": "PERF001",
                    "rule_name": "...",
                    "severity": "warning",
                    "line": 15,
                    "message": "...",
                    "suggestion": "..."
                }
            ],
            "summary": "..."       # 总结
        }

    Examples:
        >>> code = '''
        ... import mindspore as ms
        ... from mindspore import nn
        ... model = nn.Conv2d(3, 64, 3)
        ... '''
        >>> result = lint_mindspore_code(code)
        >>> print(f"Score: {result['score']}/100")
    """
    result = _check_code(code, level=level)
    
    # 添加格式化报告
    if style:
        result["formatted_report"] = format_report(result, style=style)
    
    return result


def get_lint_rules(
    category: Optional[str] = None,
    severity: Optional[str] = None
) -> dict[str, Any]:
    """获取代码检查规则列表

    Args:
        category: 规则类别过滤
            - "performance": 性能规则
            - "compatibility": 兼容性规则
            - "best_practices": 最佳实践规则
            - "maintainability": 可维护性规则
        severity: 严重级别过滤
            - "error": 严重问题
            - "warning": 警告
            - "info": 提示

    Returns:
        {
            "categories": {
                "performance": [...规则列表...],
                "compatibility": [...],
                "best_practices": [...],
                "maintainability": [...]
            },
            "total_rules": 26
        }

    Examples:
        >>> rules = get_lint_rules(category="performance")
        >>> print(f"性能规则: {len(rules['categories']['performance'])} 条")
    """
    from mindspore_tools_mcp.linter.rules import ALL_RULES
    
    categories = {
        "performance": [],
        "compatibility": [],
        "best_practices": [],
        "maintainability": [],
    }
    
    for rule in ALL_RULES:
        categories[rule.category].append({
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "severity": rule.severity,
            "suggestion": rule.suggestion,
            "examples": rule.examples,
        })
    
    # 应用过滤
    if category:
        categories = {category: categories.get(category, [])}
    
    if severity:
        for cat in categories:
            categories[cat] = [r for r in categories[cat] if r["severity"] == severity]
    
    return {
        "categories": categories,
        "total_rules": len(ALL_RULES),
    }


def lint_code_snippet(
    snippet_type: str,
    code: str,
    level: str = "quick"
) -> dict[str, Any]:
    """快速检查代码片段

    针对特定类型的代码片段进行快速检查。

    Args:
        snippet_type: 代码片段类型
            - "model": 模型定义 (nn.Cell)
            - "train": 训练循环
            - "data": 数据处理
            - "inference": 推理代码
        code: 代码字符串
        level: 检查级别

    Returns:
        快速检查结果
    """
    # 包装成完整代码
    templates = {
        "model": f'''import mindspore as ms
from mindspore import nn

class Model(nn.Cell):
    def __init__(self):
        super().__init__()
{chr(10).join("    " + line for line in code.split(chr(10)))}
''',
        "train": f'''import mindspore as ms
from mindspore import nn

model = nn.ResNet50()
{chr(10).join("    " + line for line in code.split(chr(10)))}
''',
        "data": f'''import mindspore.dataset as ds

{chr(10).join("    " + line for line in code.split(chr(10)))}
''',
        "inference": f'''import mindspore as ms
from mindspore import nn

model = nn.ResNet50()
model.set_eval()

{chr(10).join("    " + line for line in code.split(chr(10)))}
''',
    }
    
    wrapped_code = templates.get(snippet_type, code)
    return _check_code(wrapped_code, level=level)


def compare_code_snippets(
    code_a: str,
    code_b: str,
    labels: tuple[str, str] = ("A", "B")
) -> dict[str, Any]:
    """对比两个代码片段的质量

    Args:
        code_a: 第一个代码
        code_b: 第二个代码
        labels: 两个代码的标签

    Returns:
        {
            "snippet_a": {"score": 75, "issues_count": 5},
            "snippet_b": {"score": 88, "issues_count": 2},
            "winner": "B",
            "difference": {"score": 13, "issues": 3},
            "comparison": "代码 B 比代码 A 高 13 分，少 3 个问题"
        }
    """
    result_a = _check_code(code_a, level="all")
    result_b = _check_code(code_b, level="all")
    
    score_a = result_a["score"]
    score_b = result_b["score"]
    
    issues_a = len(result_a["issues"])
    issues_b = len(result_b["issues"])
    
    if score_b > score_a:
        winner = labels[1]
        diff_score = score_b - score_a
    elif score_a > score_b:
        winner = labels[0]
        diff_score = score_a - score_b
    else:
        winner = "tie"
        diff_score = 0
    
    diff_issues = abs(issues_a - issues_b)
    
    comparison = f"代码 {winner} 评分更高"
    if winner != "tie":
        comparison = f"代码 {winner} 比代码 {'B' if winner == 'A' else 'A'} 高 {diff_score} 分"
        if diff_issues > 0:
            comparison += f"，少 {diff_issues} 个问题"
    else:
        comparison = "两个代码评分相同"
    
    return {
        "snippet_a": {
            "label": labels[0],
            "score": score_a,
            "grade": result_a["grade"],
            "issues_count": issues_a,
            "issues": result_a["issues"],
        },
        "snippet_b": {
            "label": labels[1],
            "score": score_b,
            "grade": result_b["grade"],
            "issues_count": issues_b,
            "issues": result_b["issues"],
        },
        "winner": winner,
        "difference": {
            "score": diff_score,
            "issues": diff_issues,
        },
        "comparison": comparison,
    }
