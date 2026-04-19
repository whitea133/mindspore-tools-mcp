"""
MindSpore 代码评分器 - 检查器核心
===================================

实现代码检查逻辑，包括正则匹配和自定义函数检查。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .rules import (
    Rule, ALL_RULES, PERFORMANCE_RULES, COMPATIBILITY_RULES,
    BEST_PRACTICE_RULES, MAINTAINABILITY_RULES
)


@dataclass
class Issue:
    """代码问题"""
    rule_id: str
    rule_name: str
    severity: str
    category: str
    line: int
    column: int
    message: str
    suggestion: str
    code_snippet: str = ""


@dataclass
class CheckResult:
    """检查结果"""
    score: int
    grade: str
    dimensions: dict
    issues: list[Issue]
    summary: str


class CodeChecker:
    """代码检查器"""
    
    # 维度权重
    DIMENSION_WEIGHTS = {
        "performance": 0.30,
        "compatibility": 0.25,
        "best_practices": 0.25,
        "maintainability": 0.20,
    }
    
    # PyTorch 专属 API 列表
    TORCH_APIS = [
        "torch.tensor", "torch.zeros", "torch.ones", "torch.arange",
        "torch.rand", "torch.randn", "torch.from_numpy",
        "torch.nn.Module", "torch.optim", "torch.cuda",
        ".to('cuda'", ".cuda()", "nn.Module", "nn.Sequential",
        "torch.save", "torch.load", "torch.no_grad",
    ]
    
    # 废弃的 API 列表
    DEPRECATED_APIS = [
        "mindspore.nn.SummaryCell",
        "mindspore.train.callback.InternTrainSession",
    ]
    
    def __init__(self, level: str = "all"):
        """
        初始化检查器。
        
        Args:
            level: 检查级别
                - "all": 检查所有规则
                - "strict": 只检查 error 和 warning
                - "quick": 只检查关键规则
        """
        self.level = level
        self.rules = self._get_rules_by_level()
        
    def _get_rules_by_level(self) -> list[Rule]:
        """根据级别获取规则"""
        if self.level == "quick":
            # 只检查关键规则
            key_rules = ["PERF001", "PERF002", "COMP001", "COMP002", "BP001", "BP003"]
            return [r for r in ALL_RULES if r.id in key_rules]
        elif self.level == "strict":
            return [r for r in ALL_RULES if r.severity in ["error", "warning"]]
        return ALL_RULES
    
    def check(self, code: str) -> dict:
        """
        检查代码并返回报告。
        
        Args:
            code: MindSpore 代码字符串
            
        Returns:
            检查报告字典
        """
        lines = code.split('\n')
        issues: list[Issue] = []
        
        # 1. 正则规则检查
        for rule in self.rules:
            if rule.pattern:
                rule_issues = self._check_pattern(code, lines, rule)
                issues.extend(rule_issues)
        
        # 2. 自定义函数检查
        custom_checkers = {
            "check_data_augmentation": self._check_data_augmentation,
            "check_amp_usage": self._check_amp_usage,
            "check_control_flow": self._check_control_flow,
            "check_torch_apis": self._check_torch_apis,
            "check_deprecated_apis": self._check_deprecated_apis,
            "check_slow_npu_ops": self._check_slow_npu_ops,
            "check_seed_setting": self._check_seed_setting,
            "check_lr_scheduler": self._check_lr_scheduler,
            "check_checkpoint": self._check_checkpoint,
            "check_gradient_clipping": self._check_gradient_clipping,
            "check_model_mode": self._check_model_mode,
            "check_function_length": self._check_function_length,
            "check_code_duplication": self._check_code_duplication,
            "check_docstrings": self._check_docstrings,
        }
        
        for rule in self.rules:
            if rule.check_fn and rule.check_fn in custom_checkers:
                rule_issues = custom_checkers[rule.check_fn](code, lines, rule)
                issues.extend(rule_issues)
        
        # 3. 计算各维度得分
        dimensions = self._compute_dimension_scores(issues)
        
        # 4. 计算总分
        score = self._compute_total_score(dimensions)
        grade = self._score_to_grade(score)
        
        # 5. 生成总结
        summary = self._generate_summary(score, grade, issues)
        
        return {
            "score": score,
            "grade": grade,
            "dimensions": dimensions,
            "issues": [
                {
                    "rule_id": i.rule_id,
                    "rule_name": i.rule_name,
                    "severity": i.severity,
                    "category": i.category,
                    "line": i.line,
                    "column": i.column,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "code_snippet": i.code_snippet,
                }
                for i in issues
            ],
            "summary": summary,
        }
    
    def _check_pattern(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查正则模式"""
        issues = []
        
        for line_no, line in enumerate(lines, 1):
            if re.search(rule.pattern, line, re.MULTILINE):
                # 检查多行情况
                if 'for' in rule.pattern:
                    # 对于循环检查，需要看上下文
                    context_start = max(0, line_no - 2)
                    context_end = min(len(lines), line_no + 3)
                    context = '\n'.join(lines[context_start:context_end])
                    
                    # 简单检查：optimizer/loss 在 for 循环内
                    if 'for' in line and ('Optimizer' in line or 'Adam' in line or 'loss' in line.lower()):
                        pass  # 已经在 pattern 里匹配了
                
                issue = Issue(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    category=rule.category,
                    line=line_no,
                    column=1,
                    message=f"{rule.description}",
                    suggestion=rule.suggestion,
                    code_snippet=line.strip(),
                )
                issues.append(issue)
        
        return issues
    
    def _check_torch_apis(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查 PyTorch API 使用"""
        issues = []
        
        for line_no, line in enumerate(lines, 1):
            for api in self.TORCH_APIS:
                if api in line and not line.strip().startswith('#'):
                    issue = Issue(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        line=line_no,
                        column=1,
                        message=f"使用了 PyTorch 专属 API: {api}",
                        suggestion=rule.suggestion,
                        code_snippet=line.strip(),
                    )
                    issues.append(issue)
                    break
        
        return issues
    
    def _check_data_augmentation(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查数据增强配置"""
        issues = []
        
        # 检查是否在循环内重复创建 transforms
        in_loop = False
        loop_indent = 0
        
        for line_no, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # 检测 for 循环
            if re.match(r'for\s+', stripped):
                in_loop = True
                loop_indent = indent
                continue
            
            # 离开 for 循环
            if in_loop and stripped and indent <= loop_indent:
                in_loop = False
            
            # 在循环内创建 transforms
            if in_loop and ('transform' in stripped.lower() or 'augment' in stripped.lower()):
                if 'nn.transforms' in stripped or 'mindspore.dataset' in stripped:
                    issue = Issue(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        line=line_no,
                        column=1,
                        message=rule.description,
                        suggestion=rule.suggestion,
                        code_snippet=stripped,
                    )
                    issues.append(issue)
                    break
        
        return issues
    
    def _check_amp_usage(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查是否使用混合精度"""
        issues = []
        
        has_amp = any([
            'amp' in code.lower(),
            'loss_scale' in code.lower(),
            'DynamicLossScaleManager' in code,
            'FixedLossScaleManager' in code,
        ])
        
        # 如果没有设置种子（需要训练上下文），且没有 AMP
        has_train_loop = 'model.train' in code or 'train_network' in code
        
        if has_train_loop and not has_amp:
            issue = Issue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                line=1,
                column=1,
                message=rule.description,
                suggestion=rule.suggestion,
                code_snippet="",
            )
            issues.append(issue)
        
        return issues
    
    def _check_control_flow(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查控制流"""
        issues = []
        
        # 检测 construct 方法中的 Python if/for while
        in_construct = False
        
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if 'def construct' in stripped:
                in_construct = True
                continue
            
            # 离开函数
            if in_construct and stripped and not stripped.startswith('#'):
                if stripped.startswith('def ') or (stripped.startswith('class ') and not stripped.startswith('class Cell')):
                    in_construct = False
                    continue
                
                # 在 construct 中使用 Python 控制流
                if in_construct and any([
                    re.match(r'if\s+.*:', stripped),
                    re.match(r'for\s+.*:', stripped),
                    re.match(r'while\s+.*:', stripped),
                ]):
                    # 排除 return 语句
                    if not stripped.startswith('return'):
                        issue = Issue(
                            rule_id=rule.id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            category=rule.category,
                            line=line_no,
                            column=1,
                            message=f"在 construct 中使用了 Python 控制流: {stripped[:30]}...",
                            suggestion=rule.suggestion,
                            code_snippet=stripped,
                        )
                        issues.append(issue)
        
        return issues
    
    def _check_deprecated_apis(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查废弃 API"""
        issues = []
        
        for line_no, line in enumerate(lines, 1):
            for api in self.DEPRECATED_APIS:
                if api in line:
                    issue = Issue(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        line=line_no,
                        column=1,
                        message=f"使用了已废弃的 API: {api}",
                        suggestion=rule.suggestion,
                        code_snippet=line.strip(),
                    )
                    issues.append(issue)
                    break
        
        return issues
    
    def _check_slow_npu_ops(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查可能在 NPU 上慢的算子"""
        issues = []
        
        # 这里简化处理，实际应该有更复杂的检测
        slow_patterns = [
            (r'\.expand\(', 'expand 可能不如 broadcast 高效'),
            (r'\.repeat\(', 'repeat 可能导致内存问题'),
        ]
        
        for line_no, line in enumerate(lines, 1):
            for pattern, msg in slow_patterns:
                if re.search(pattern, line):
                    issue = Issue(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        line=line_no,
                        column=1,
                        message=msg,
                        suggestion=rule.suggestion,
                        code_snippet=line.strip(),
                    )
                    issues.append(issue)
                    break
        
        return issues
    
    def _check_seed_setting(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查是否设置随机种子"""
        issues = []
        
        has_seed = 'set_seed' in code
        has_train = any(x in code for x in ['model.train', 'train_network', 'model.compile'])
        
        if has_train and not has_seed:
            issue = Issue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                line=1,
                column=1,
                message=rule.description,
                suggestion=rule.suggestion,
                code_snippet="",
            )
            issues.append(issue)
        
        return issues
    
    def _check_lr_scheduler(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查是否使用学习率调度器"""
        issues = []
        
        has_train = 'model.train' in code or 'train_network' in code
        has_scheduler = any(x in code for x in [
            'CosineAnnealingLR', 'StepLR', 'PolynomialLR',
            'ExponentialLR', 'MStepLROptimizer', 'lr_scheduler'
        ])
        
        if has_train and not has_scheduler:
            issue = Issue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                line=1,
                column=1,
                message=rule.description,
                suggestion=rule.suggestion,
                code_snippet="",
            )
            issues.append(issue)
        
        return issues
    
    def _check_checkpoint(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查是否使用检查点"""
        issues = []
        
        has_train = 'model.train' in code or 'train_network' in code
        has_checkpoint = 'Checkpoint' in code or 'save_checkpoint' in code
        
        if has_train and not has_checkpoint:
            issue = Issue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                line=1,
                column=1,
                message=rule.description,
                suggestion=rule.suggestion,
                code_snippet="",
            )
            issues.append(issue)
        
        return issues
    
    def _check_gradient_clipping(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查是否使用梯度裁剪"""
        issues = []
        
        has_train = 'model.train' in code or 'train_network' in code
        has_clip = any(x in code for x in ['clip_by_norm', 'clip_by_global_norm', 'grad_clip'])
        
        if has_train and not has_clip:
            issue = Issue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                line=1,
                column=1,
                message=rule.description,
                suggestion=rule.suggestion,
                code_snippet="",
            )
            issues.append(issue)
        
        return issues
    
    def _check_model_mode(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查模型模式设置"""
        issues = []
        
        # 检查是否有 forward 或 construct 调用但没有模式设置
        has_forward = 'model(' in code or 'self.forward' in code or 'self.construct' in code
        has_mode_set = any(x in code for x in ['.set_train()', '.train()', '.set_train(True)', '.set_eval()', '.eval()'])
        
        if has_forward and not has_mode_set:
            issue = Issue(
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                category=rule.category,
                line=1,
                column=1,
                message=rule.description,
                suggestion=rule.suggestion,
                code_snippet="",
            )
            issues.append(issue)
        
        return issues
    
    def _check_function_length(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查函数长度"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 1
                    
                    if func_lines > 200:
                        # 找到函数定义的行
                        line_content = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        issue = Issue(
                            rule_id=rule.id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            category=rule.category,
                            line=node.lineno,
                            column=1,
                            message=f"函数 '{node.name}' 有 {func_lines} 行，超过 200 行限制",
                            suggestion=rule.suggestion,
                            code_snippet=line_content.strip(),
                        )
                        issues.append(issue)
        except SyntaxError:
            pass
        
        return issues
    
    def _check_code_duplication(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查代码重复 - 简化版"""
        issues = []
        
        # 提取非空非注释行
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # 移除字符串和数字，保留结构
                normalized = re.sub(r'["\'].*?["\']', 'STR', line)
                normalized = re.sub(r'\d+', 'N', normalized)
                normalized = re.sub(r'\s+', ' ', normalized)
                code_lines.append(normalized)
        
        # 简单检查：连续3行以上相同结构
        seen = {}
        for i, line in enumerate(code_lines):
            if line in seen:
                prev_idx = seen[line]
                if i - prev_idx < 10:  # 10行内重复
                    issue = Issue(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        category=rule.category,
                        line=i + 1,
                        column=1,
                        message="检测到可能的代码重复",
                        suggestion=rule.suggestion,
                        code_snippet=lines[i].strip()[:50],
                    )
                    issues.append(issue)
                    break
            else:
                seen[line] = i
        
        return issues
    
    def _check_docstrings(self, code: str, lines: list[str], rule: Rule) -> list[Issue]:
        """检查文档字符串"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            important_funcs = ['train', 'eval', 'forward', 'construct', 'predict', 'inference']
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.lower() in important_funcs:
                        # 检查是否有 docstring
                        has_docstring = (
                            ast.get_docstring(node) is not None or
                            (node.body and isinstance(node.body[0], ast.Expr) and
                             isinstance(node.body[0].value, (ast.Str, ast.Constant)))
                        )
                        
                        if not has_docstring:
                            line_content = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                            issue = Issue(
                                rule_id=rule.id,
                                rule_name=rule.name,
                                severity=rule.severity,
                                category=rule.category,
                                line=node.lineno,
                                column=1,
                                message=f"重要函数 '{node.name}' 缺少文档字符串",
                                suggestion=rule.suggestion,
                                code_snippet=line_content.strip(),
                            )
                            issues.append(issue)
        except SyntaxError:
            pass
        
        return issues
    
    def _compute_dimension_scores(self, issues: list[Issue]) -> dict:
        """计算各维度得分"""
        dimensions = {}
        
        for dim_name in ["performance", "compatibility", "best_practices", "maintainability"]:
            dim_issues = [i for i in issues if i.category == dim_name]
            
            # 基础分 100
            score = 100
            
            # 根据问题扣分
            for issue in dim_issues:
                if issue.severity == "error":
                    score -= 15
                elif issue.severity == "warning":
                    score -= 8
                else:  # info
                    score -= 3
            
            score = max(0, score)
            dimensions[dim_name] = {
                "score": score,
                "issues_count": len(dim_issues),
                "issues": dim_issues,
            }
        
        return dimensions
    
    def _compute_total_score(self, dimensions: dict) -> int:
        """计算总分"""
        total = 0
        for dim_name, weight in self.DIMENSION_WEIGHTS.items():
            total += dimensions[dim_name]["score"] * weight
        return round(total)
    
    def _score_to_grade(self, score: int) -> str:
        """分数转等级"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_summary(self, score: int, grade: str, issues: list[Issue]) -> str:
        """生成总结"""
        error_count = len([i for i in issues if i.severity == "error"])
        warning_count = len([i for i in issues if i.severity == "warning"])
        info_count = len([i for i in issues if i.severity == "info"])
        
        summary_parts = []
        
        if error_count > 0:
            summary_parts.append(f"发现 {error_count} 个严重问题需要修复")
        if warning_count > 0:
            summary_parts.append(f"{warning_count} 个警告建议优化")
        if info_count > 0:
            summary_parts.append(f"{info_count} 个提示供参考")
        
        if not summary_parts:
            summary_parts.append("代码质量良好，没有发现问题")
        
        grade_desc = {
            "A": "优秀",
            "B": "良好",
            "C": "一般",
            "D": "较差",
            "F": "不合格",
        }
        
        return f"整体评分 {score}/100 (Grade: {grade} {grade_desc.get(grade, '')})。{'，'.join(summary_parts)}。"
