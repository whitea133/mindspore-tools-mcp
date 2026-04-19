"""
MindSpore 代码评分器 - 报告格式化
===================================

将检查结果格式化为易读的文本或 JSON 报告。
"""

from __future__ import annotations
from typing import Any


# 颜色代码 (ANSI)
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"


def _color(text: str, color: str) -> str:
    """给文本添加颜色"""
    return f"{color}{text}{Colors.RESET}"


def _severity_icon(severity: str) -> str:
    """获取严重级别图标"""
    icons = {
        "error": "🔴",
        "warning": "🟡",
        "info": "🔵",
    }
    return icons.get(severity, "⚪")


def _severity_color(text: str, severity: str) -> str:
    """根据严重级别着色"""
    colors = {
        "error": Colors.RED,
        "warning": Colors.YELLOW,
        "info": Colors.BLUE,
    }
    return _color(text, colors.get(severity, ""))


def format_report(result: dict, style: str = "pretty") -> str:
    """
    格式化检查报告。
    
    Args:
        result: check_code() 返回的结果
        style: 报告风格
            - "pretty": 彩色终端输出
            - "simple": 简单文本输出
            - "json": JSON 格式
            - "markdown": Markdown 格式
    
    Returns:
        格式化后的报告字符串
    """
    if style == "json":
        return _format_json(result)
    elif style == "markdown":
        return _format_markdown(result)
    elif style == "simple":
        return _format_simple(result)
    else:
        return _format_pretty(result)


def _format_pretty(result: dict) -> str:
    """格式化彩色终端报告"""
    lines = []
    
    # 标题
    lines.append("")
    lines.append("=" * 60)
    lines.append(_color("  MindSpore 代码评分报告", Colors.BOLD))
    lines.append("=" * 60)
    
    # 总分
    score = result["score"]
    grade = result["grade"]
    
    # 根据分数选择颜色
    if score >= 80:
        score_color = Colors.GREEN
    elif score >= 60:
        score_color = Colors.YELLOW
    else:
        score_color = Colors.RED
    
    lines.append("")
    lines.append(f"  {_color('总分:', Colors.BOLD)} {score_color}{score}/100{Colors.RESET}  {_color('等级:', Colors.BOLD)} {score_color}{grade}{Colors.RESET}")
    
    # 各维度得分
    lines.append("")
    lines.append(f"  {_color('各维度得分:', Colors.BOLD)}")
    lines.append("-" * 40)
    
    dims = result["dimensions"]
    for dim_name, dim_data in dims.items():
        dim_score = dim_data["score"]
        issue_count = dim_data["issues_count"]
        
        # 维度名称映射
        dim_names = {
            "performance": "⚡ 性能",
            "compatibility": "🔌 兼容性",
            "best_practices": "✨ 最佳实践",
            "maintainability": "🔧 可维护性",
        }
        
        # 得分颜色
        if dim_score >= 80:
            dim_color = Colors.GREEN
        elif dim_score >= 60:
            dim_color = Colors.YELLOW
        else:
            dim_color = Colors.RED
        
        issue_str = f"({issue_count} 问题)" if issue_count > 0 else "(✓)"
        lines.append(f"  {dim_names.get(dim_name, dim_name)}")
        lines.append(f"    {dim_color}{dim_score}/100{Colors.RESET} {Colors.GRAY}{issue_str}{Colors.RESET}")
    
    # 问题列表
    issues = result["issues"]
    if issues:
        lines.append("")
        lines.append(f"  {_color('发现问题:', Colors.BOLD)}")
        lines.append("-" * 40)
        
        # 按严重程度分组
        for severity in ["error", "warning", "info"]:
            severity_issues = [i for i in issues if i["severity"] == severity]
            if not severity_issues:
                continue
            
            severity_names = {"error": "严重", "warning": "警告", "info": "提示"}
            lines.append("")
            lines.append(f"  {_severity_color(f'{_severity_icon(severity)} {severity_names[severity]} ({len(severity_issues)})', severity)}")
            
            for issue in severity_issues[:10]:  # 最多显示10个
                lines.append(f"    第 {issue['line']} 行: {issue['message']}")
                if issue["suggestion"]:
                    lines.append(f"    {_color('→ 建议:', Colors.GRAY)} {issue['suggestion']}")
    
    # 总结
    lines.append("")
    lines.append("-" * 40)
    lines.append(f"  {_color('总结:', Colors.BOLD)} {result['summary']}")
    lines.append("=" * 60)
    lines.append("")
    
    return "\n".join(lines)


def _format_simple(result: dict) -> str:
    """格式化简单文本报告"""
    lines = []
    
    lines.append("MindSpore 代码评分报告")
    lines.append(f"总分: {result['score']}/100 (Grade: {result['grade']})")
    lines.append("")
    
    for dim_name, dim_data in result["dimensions"].items():
        lines.append(f"{dim_name}: {dim_data['score']}/100 ({dim_data['issues_count']} issues)")
    
    lines.append("")
    lines.append("问题列表:")
    
    for issue in result["issues"]:
        lines.append(f"  [{issue['severity'].upper()}] {issue['rule_id']} 第{issue['line']}行: {issue['message']}")
    
    return "\n".join(lines)


def _format_json(result: dict) -> str:
    """格式化为 JSON"""
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


def _format_markdown(result: dict) -> str:
    """格式化为 Markdown"""
    lines = []
    
    lines.append("# MindSpore 代码评分报告")
    lines.append("")
    
    # 总分
    score = result["score"]
    grade = result["grade"]
    
    if grade in ["A"]:
        badge = "🟢 优秀"
    elif grade in ["B"]:
        badge = "🟡 良好"
    elif grade in ["C"]:
        badge = "🟠 一般"
    else:
        badge = "🔴 较差"
    
    lines.append(f"**总分:** {score}/100 {badge}")
    lines.append("")
    lines.append(f"> {result['summary']}")
    lines.append("")
    
    # 各维度
    lines.append("## 各维度得分")
    lines.append("")
    lines.append("| 维度 | 得分 | 问题数 |")
    lines.append("|------|------|--------|")
    
    for dim_name, dim_data in result["dimensions"].items():
        dim_score = dim_data["score"]
        issue_count = dim_data["issues_count"]
        
        # 进度条
        bar_len = 20
        filled = int(dim_score / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        lines.append(f"| {dim_name} | {bar} {dim_score} | {issue_count} |")
    
    lines.append("")
    
    # 问题列表
    if result["issues"]:
        lines.append("## 问题列表")
        lines.append("")
        
        for severity in ["error", "warning", "info"]:
            severity_issues = [i for i in result["issues"] if i["severity"] == severity]
            if not severity_issues:
                continue
            
            severity_names = {"error": "🔴 严重", "warning": "🟡 警告", "info": "🔵 提示"}
            lines.append(f"### {severity_names[severity]} ({len(severity_issues)})")
            lines.append("")
            
            for issue in severity_issues:
                lines.append(f"**{issue['rule_id']}** 第 {issue['line']} 行")
                lines.append(f"> {issue['message']}")
                if issue["suggestion"]:
                    lines.append(f"> 💡 建议: {issue['suggestion']}")
                lines.append("")
    
    return "\n".join(lines)


def print_report(result: dict, style: str = "pretty") -> None:
    """打印报告到终端"""
    print(format_report(result, style))


def save_report(result: dict, filepath: str, style: str = "markdown") -> None:
    """保存报告到文件"""
    content = format_report(result, style)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
