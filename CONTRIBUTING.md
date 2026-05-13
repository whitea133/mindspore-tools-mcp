# 贡献指南

感谢你有意为 `mindspore-tools-mcp` 做出贡献！本项目欢迎各种形式的参与，包括但不限于代码、功能、文档、测试、数据更新等。

---

## 📋 目录

- [快速开始](#快速开始)
- [开发环境](#开发环境)
- [项目结构说明](#项目结构说明)
- [如何贡献](#如何贡献)
- [工具模块说明](#工具模块说明)
- [提交规范](#提交规范)
- [测试指南](#测试指南)
- [文档指南](#文档指南)
- [数据更新](#数据更新)
- [问题反馈](#问题反馈)

---

## 快速开始

```bash
# Fork 并克隆仓库
git clone https://github.com/whitea133/mindspore-tools-mcp.git
cd mindspore-tools-mcp

# 安装依赖
uv sync

# 启动 MCP 服务（测试）
uv run python -m mindspore_tools_mcp.server
```

---

## 开发环境

### 环境要求

- **Python**: 3.8+
- **包管理器**: uv（推荐）或 pip
- **MCP 客户端**: Cline / Cursor / Claude Code 等

### 安装开发依赖

```bash
# 安装所有依赖（包括开发依赖）
uv sync

# 运行测试
uv run pytest tests/

# 代码格式检查
uv run ruff check src/
```

---

## 项目结构说明

```
mindspore-tools-mcp/
├── src/mindspore_tools_mcp/     # 核心代码
│   ├── tools.py                  # 模型检索工具
│   ├── msutils_tools.py          # msutils MCP 封装
│   ├── linter_tools.py          # 代码评分器封装
│   ├── template_tools.py         # 训练模板生成器封装
│   ├── api_tools.py             # API 示例生成器封装
│   ├── server.py                 # MCP 服务入口
│   ├── msutils/                  # MindSpore 开发工具库
│   ├── linter/                   # 代码评分器核心
│   ├── templates/                # 训练模板生成器
│   └── api_examples/             # API 示例生成器
│
├── examples/                      # 使用示例
├── tests/                        # 测试文件
├── scripts/                      # 数据更新脚本
└── data/                        # 模型与映射数据
```

---

## 如何贡献

### 方式一：代码贡献

#### 1. 认领 Issue

在 [GitHub Issues](https://github.com/whitea133/mindspore-tools-mcp/issues) 中选择你想要处理的问题，或创建新 Issue 说明你想要实现的功能。

#### 2. 创建分支

```bash
# 创建功能分支
git checkout -b feature/your-feature-name
# 或修复分支
git checkout -b fix/issue-description
```

#### 3. 开发与测试

```bash
# 在本地开发
# 修改代码后运行测试
uv run pytest tests/

# 确保所有测试通过
uv run pytest -v
```

#### 4. 提交代码

```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

#### 5. 创建 Pull Request

在 GitHub 上创建 PR，描述你的改动内容和解决的问题。

---

### 方式二：文档贡献

- 改善现有文档的清晰度和完整性
- 补充缺失的使用示例
- 翻译文档为其他语言
- 修正文档中的错误

---

### 方式三：数据贡献

#### 更新模型清单

如果你发现 MindSpore 官方发布了新模型，可以通过以下方式贡献：

1. 运行 `scripts/update_model_list.py` 自动更新
2. 或手动编辑 `data/mindspore_official_models.json`

#### 补充 API 映射

如果发现 PyTorch API 有缺失的 MindSpore 映射：

1. 编辑 `data/pytorch_ms_api_mapping_diff.json`
2. 或运行 `scripts/fetch_api_mapping.py` 重新抓取

#### 扩展 API 示例

在 `src/mindspore_tools_mcp/api_examples/registry.py` 中添加新的 API 示例：

```python
# 添加新的 API 示例
API_EXAMPLES = {
    # ... 现有内容
    "nn.NewAPILayer": {
        "description": "新 API 层描述",
        "signature": "nn.NewAPILayer(args, kwargs)",
        "params": [...],
        "examples": [
            {
                "title": "基础用法",
                "code": "import mindspore.nn as nn\n...",
                "description": "示例说明"
            }
        ],
        "related_apis": ["nn.Conv2d", "nn.Dense"],
        "official_doc": "https://www.mindspore.cn/docs/..."
    }
}
```

---

## 工具模块说明

### msutils 工具库

如需扩展 `msutils` 模块，请遵循以下目录结构：

```
src/mindspore_tools_mcp/msutils/
├── data/           # 数据处理模块
│   ├── augmentations.py   # 数据增强
│   ├── loaders.py         # 数据加载
│   └── transforms.py      # 数据变换
├── train/          # 训练工具模块
│   ├── callbacks.py       # 回调函数
│   └── schedulers.py      # 学习率调度
├── security/       # AI 安全模块
│   ├── attacks.py         # 攻击方法
│   ├── defenses.py        # 防御方法
│   └── evaluation.py       # 评估工具
└── ...
```

**编写规范：**
- 每个模块需包含清晰的文档字符串
- 函数需标注参数类型和返回值类型
- 添加使用示例在 docstring 中

### Linter 代码评分器

如需扩展检查规则，在 `src/mindspore_tools_mcp/linter/rules.py` 中添加：

```python
# 新增检查规则
RULES = [
    # ... 现有规则
    Rule(
        id="MS001",
        category="performance",
        name="avoid_tensor_copy",
        description="避免在循环内进行大 tensor 拷贝",
        severity="warning",
        check_fn=check_avoid_tensor_copy
    )
]
```

### 训练模板生成器

如需扩展支持的模型或数据集，在 `src/mindspore_tools_mcp/templates/generator.py` 中添加选项。

---

## 提交规范

### Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型：**

| 类型 | 说明 |
|------|------|
| `Add` | 新增功能 |
| `Fix` | 修复 bug |
| `Update` | 更新现有功能 |
| `Refactor` | 代码重构 |
| `Docs` | 文档更新 |
| `Test` | 添加或更新测试 |
| `Chore` | 构建或辅助工具更新 |
| `Data` | 数据文件更新 |

**示例：**

```
Add: 新增 ResNet101 模型复杂度计算支持

- 在 msutils/analysis/complexity.py 中添加 resnet101 计算逻辑
- 添加对应测试用例
- 更新文档说明

Closes #12
```

### 代码格式

- 使用 **Black** 进行代码格式化
- 使用 **Ruff** 进行 lint 检查
- 遵循 **PEP 8** 规范

```bash
# 格式化代码
uv run black src/

# 检查代码
uv run ruff check src/
```

---

## 测试指南

### 运行测试

```bash
# 运行所有测试
uv run pytest tests/

# 运行指定测试
uv run pytest tests/test_recommend.py

# 带详细输出
uv run pytest -v tests/
```

### 编写测试

```python
# tests/test_example.py
import pytest
from mindspore_tools_mcp import tools

def test_list_models():
    """测试模型列表查询"""
    result = tools.list_models(task="image-classification")
    assert isinstance(result, list)
    assert len(result) > 0

def test_recommend_models():
    """测试模型推荐"""
    result = tools.recommend_models("图像分类")
    assert "models" in result or "recommendations" in result
```

### 测试覆盖率

我们期望核心功能的测试覆盖率达到 **80%+**。

---

## 文档指南

### API 文档

所有公共函数和类需包含以下文档：

```python
def function_name(param1: str, param2: int = 10) -> dict:
    """简要描述函数功能。

    详细说明函数的用途、工作原理和使用场景。

    Args:
        param1: 参数1的说明
        param2: 参数2的说明（默认: 10）

    Returns:
        返回值的说明

    Examples:
        >>> result = function_name("test", 20)
        >>> print(result)
        {'status': 'ok'}

    Note:
        重要注意事项或限制。
    """
```

### README 文档

新增功能需要在 README 中补充：
- 功能说明
- 使用示例
- 相关工具链接

---

## 数据更新

### 更新官方模型清单

```bash
# 自动从 MindSpore 官方获取最新模型列表
uv run python scripts/update_model_list.py
```

### 更新 API 映射

```bash
# 从 MindSpore 官方文档抓取最新 API 映射
uv run python scripts/fetch_api_mapping.py
```

### 数据格式要求

- **JSON 文件**需符合 `data/` 目录下的格式规范
- 新增数据需包含完整的字段说明
- 提交前验证 JSON 格式正确性

```python
import json

# 验证 JSON 格式
with open("data/your_file.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    # 验证结构完整性
```

---

## 问题反馈

### 创建 Issue

- 选择合适的 Issue 模板（Bug Report / Feature Request / Documentation）
- 清晰描述问题和期望
- 附上相关代码、日志或截图
- 提供复现步骤

### 讨论

欢迎在 GitHub Discussions 中发起讨论：
- 询问使用方法
- 提出功能建议
- 分享使用经验

---

## 行为准则

- 保持友好和专业的交流态度
- 尊重不同的观点和建议
- 欢迎新手提问，帮助他人成长
- 代码审查时注重建设性反馈

---

## 联系方式

- **GitHub**: https://github.com/whitea133/mindspore-tools-mcp
- **Gitee**: https://gitee.com/whitea133/mindspore-tools-mcp
- **邮箱**: 1309848726@qq.com

感谢你的贡献！🎉