"""
MindSpore 代码评分器
====================

自动分析 MindSpore 代码，检测性能问题、兼容性问题、最佳实践违规，
并输出质量报告和改进建议。

评分维度：
- 性能 (Performance): 30%
- 兼容性 (Compatibility): 25%
- 最佳实践 (Best Practices): 25%
- 可维护性 (Maintainability): 20%
"""

from .checker import CodeChecker
from .rules import (
    PERFORMANCE_RULES, COMPATIBILITY_RULES, 
    BEST_PRACTICE_RULES, MAINTAINABILITY_RULES, ALL_RULES
)
from .formatter import format_report

__all__ = [
    "CodeChecker",
    "check_code",
    "format_report",
    "PERFORMANCE_RULES",
    "COMPATIBILITY_RULES",
    "BEST_PRACTICE_RULES",
    "MAINTAINABILITY_RULES",
    "ALL_RULES",
]


def check_code(code: str, level: str = "all") -> dict:
    """
    检查代码并返回评分报告。
    
    Args:
        code: MindSpore 代码字符串
        level: 检查级别 ("all", "strict", "quick")
    
    Returns:
        {
            "score": 85,           # 总分 0-100
            "grade": "A",          # 等级 A/B/C/D/F
            "dimensions": {         # 各维度得分
                "performance": {"score": 90, "issues": []},
                "compatibility": {"score": 80, "issues": []},
                "best_practices": {"score": 85, "issues": []},
                "maintainability": {"score": 80, "issues": []},
            },
            "issues": [            # 所有问题
                {
                    "type": "performance",
                    "severity": "warning",
                    "line": 15,
                    "code": "...",
                    "message": "...",
                    "suggestion": "..."
                }
            ],
            "summary": "..."       # 总结
        }
    """
    checker = CodeChecker(level=level)
    return checker.check(code)
