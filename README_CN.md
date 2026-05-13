# MindSpore Tools MCP

<div align="center">

![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Ready-purple?style=flat-square)

**基于 MCP（Model Context Protocol）的 MindSpore 开发工具套件**

提供官方模型清单查询、智能模型推荐、PyTorch→MindSpore API 映射，以及 AI 安全、数据处理、训练工具等开发必备功能。

[English](./README.md) | 中文

</div>

---

## 🎯 这是什么？

**mindspore-tools-mcp** 是一套面向 MindSpore 开发者的 MCP 工具集。它让你的 AI 编程助手（如 Cline、Cursor、Claude Code）能够：

- 🔍 **检索 MindSpore 官方模型**（300+ 模型，分类筛选）
- 🤖 **智能推荐模型**（用自然语言描述需求，AI 帮你选）
- 🗺️ **PyTorch→MindSpore API 映射**（快速找到对应 API）
- 🛡️ **AI 安全工具**（对抗攻击、鲁棒性评估）
- 📊 **数据处理与训练工具**（数据增强、学习率调度、回调函数）
- 📈 **模型分析**（FLOPs、参数量、内存占用）
- 📝 **代码质量评分**（MindSpore 专属代码检查，26 条规则）
- 🚂 **训练模板生成**（一键生成 ResNet/LeNet 等完整训练脚本）

所有工具均通过 MCP 协议暴露，AI 助手可以直接调用，无需手动查文档。

---

## ✨ 功能特性一览

### 🔍 官方模型检索

| 工具 | 说明 |
|------|------|
| `list_models` | 按 group/category/task/suite 或关键词过滤模型 |
| `get_model_info` | 获取单模型的完整信息（参数、精度、数据集） |
| `recommend_models` | 自然语言智能推荐模型（支持硬件约束：Ascend/GPU/CPU） |
| `compare_models` | 对比多个模型参数和性能指标 |

### 🗺️ API 映射工具

| 工具 | 说明 |
|------|------|
| `query_op_mapping` | 查询 PyTorch API 对应的 MindSpore 实现（支持模糊匹配） |
| `diagnose_translation` | 检查 PyTorch→MindSpore 代码翻译是否完整 |

### 🛡️ AI 安全工具

| 工具 | 说明 |
|------|------|
| `generate_adversarial_attack` | 生成对抗攻击配置（FGSM/PGD/DeepFool/CW/JSMA） |
| `evaluate_model_robustness` | 评估模型鲁棒性（生成评估配置和示例代码） |

### 📊 数据处理工具

| 工具 | 说明 |
|------|------|
| `create_data_augmentation_pipeline` | 创建数据增强流水线（图像分类/目标检测/语义分割/NLP） |

### 🔧 训练工具

| 工具 | 说明 |
|------|------|
| `get_lr_scheduler` | 获取学习率调度器（余弦退火/阶梯衰减/多项式/One Cycle） |
| `get_training_callbacks` | 获取训练回调函数（检查点/早停/TensorBoard/梯度裁剪） |

### 📈 模型分析工具

| 工具 | 说明 |
|------|------|
| `compute_model_complexity` | 计算 FLOPs、参数量、内存占用 |

### 🌐 分布式训练

| 工具 | 说明 |
|------|------|
| `setup_distributed_training` | 配置分布式训练（DDP 多卡训练） |

### 🚀 部署工具

| 工具 | 说明 |
|------|------|
| `quantize_model` | 模型量化（动态量化/静态量化/量化感知训练） |
| `convert_model_format` | 模型格式转换（PyTorch ↔ MindSpore ↔ ONNX） |

### 🔍 代码评分器（Linter）

| 工具 | 说明 |
|------|------|
| `lint_mindspore_code` | MindSpore 代码质量评分（4 维度、26 条规则） |
| `get_lint_rules` | 获取所有检查规则详情 |
| `compare_code_snippets` | 对比两个代码片段的质量差异 |

### 🚂 训练模板生成器

| 工具 | 说明 |
|------|------|
| `generate_training_template` | 一键生成完整训练脚本（模型+数据集+配置+回调） |
| `get_available_options` | 获取所有可用模型/数据集/硬件选项 |
| `generate_quick_start` | 生成适合新手的入门级训练脚本 |

### 📖 API 示例生成器

| 工具 | 说明 |
|------|------|
| `get_api_examples` | 获取 MindSpore API 完整示例（描述/签名/参数/多示例代码） |
| `search_apis` | 模糊搜索相关 API（输入"卷积"自动匹配所有卷积相关 API） |
| `list_api_categories` | 列出所有 API 分类和数量 |
| `get_related_apis` | 获取相关 API 列表（如 Conv2d → BatchNorm2d、Dense） |
| `get_quick_reference` | 快速 API 参考（简洁版，适合快速查阅） |

---

## 📁 项目结构

```
mindspore-tools-mcp/
├── data/                              # 官方模型与 API 映射数据
│   ├── mindspore_official_models.json  # 300+ 官方模型清单
│   ├── pytorch_ms_api_mapping_*.json  # PyTorch→MindSpore API 映射
│   └── convert/                       # 按 section 分片的映射数据
│
├── scripts/                           # 数据更新脚本
│   ├── update_model_list.py           # 更新官方模型 JSON
│   └── fetch_api_mapping.py           # 抓取 API 映射
│
├── src/mindspore_tools_mcp/            # MCP 服务核心
│   ├── server.py                      # MCP 服务入口
│   ├── tools.py                       # 模型检索工具
│   ├── msutils_tools.py               # msutils MCP 封装
│   ├── linter_tools.py                # 代码评分器封装
│   ├── template_tools.py              # 训练模板生成器封装
│   ├── api_tools.py                   # API 示例生成器封装
│   ├── resource.py                    # MCP 资源定义
│   ├── prompt.py                     # MCP Prompt 定义
│   │
│   ├── msutils/                      # MindSpore 开发工具库
│   │   ├── data/                     # 数据增强、加载、变换
│   │   ├── train/                    # 回调函数、学习率调度
│   │   ├── security/                 # 对抗攻击、防御、评估
│   │   ├── eval/                     # 评估指标
│   │   ├── nlp/                      # 文本增强、分词器
│   │   ├── distributed/              # DDP 分布式训练
│   │   ├── deploy/                   # 模型转换、量化
│   │   └── analysis/                 # 复杂度分析、可视化
│   │
│   ├── linter/                       # 代码评分器核心
│   │   ├── rules.py                  # 26 条检查规则定义
│   │   ├── checker.py                # 检查逻辑
│   │   └── formatter.py              # 报告格式化
│   │
│   ├── templates/                    # 训练模板生成器
│   │   └── generator.py              # 模板生成逻辑
│   │
│   └── api_examples/                 # API 示例生成器
│       ├── registry.py               # API 注册表（34 个 API）
│       └── searcher.py               # 搜索逻辑
│
├── examples/                          # 使用示例（17 个）
├── tests/                             # 测试文件
├── pyproject.toml                     # 项目配置
└── uv.lock                           # 依赖锁文件
```

---

## 🚀 快速开始

### 方式一：通过 MCP 客户端调用（推荐）

#### 步骤 1：安装依赖

```bash
cd E:\CodeProject\mindspore-tools-mcp
uv sync
```

#### 步骤 2：配置 MCP 客户端

以 **Cline** 为例，在 `cline_mcp_settings.json` 中添加：

```jsonc
{
  "mcpServers": {
    "mindspore_tools_mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "E:/CodeProject/mindspore-tools-mcp",
        "run",
        "python",
        "-m",
        "mindspore_tools_mcp.server"
      ],
      "autoApprove": []
    }
  }
}
```

#### 步骤 3：开始使用

在 Cline 的 MCP Tools 面板中，你可以直接调用：

```
recommend_models("图像分类任务，需要在 Ascend 芯片上运行")
lint_mindspore_code("你的 MindSpore 训练代码")
generate_training_template(model="resnet50", dataset="cifar10", hardware="Ascend")
```

### 方式二：直接导入 Python 模块

```python
from mindspore_tools_mcp import tools
from mindspore_tools_mcp import msutils_tools
from mindspore_tools_mcp import linter_tools
from mindspore_tools_mcp import template_tools
from mindspore_tools_mcp import api_tools

# 模型查询
models = tools.list_models(task="image-classification")
print(models)

# AI 安全攻击配置
attack = msutils_tools.generate_adversarial_attack("fgsm", epsilon=0.1)
print(attack['code_example'])

# 代码评分
result = linter_tools.lint_mindspore_code(code)
print(result['score'])

# 生成训练脚本
template = template_tools.generate_training_template(
    task="image_classification",
    model="resnet50",
    dataset="cifar10",
    hardware="Ascend"
)
print(template['script'][:200])

# API 示例
api_info = api_tools.get_api_examples("nn.Conv2d")
print(api_info['description'])
```

---

## 💡 常见工作流

### 工作流 1：选择模型 → 训练 → 部署（完整流程）

```
1. recommend_models("图像分类")               → 智能推荐模型
2. compute_model_complexity("resnet50")       → 分析模型复杂度
3. create_data_augmentation_pipeline("image_classification")  → 配置数据增强
4. get_lr_scheduler("cosine_annealing")      → 配置学习率调度
5. get_training_callbacks(["checkpoint", "early_stopping"])   → 配置回调
6. generate_training_template(...)           → 一键生成训练脚本
7. quantize_model("dynamic", precision="int8") → 模型量化
8. convert_model_format("mindspore", "onnx")  → 导出部署
```

### 工作流 2：PyTorch 项目迁移

```
1. query_op_mapping("torch.xxx")              → 查找 API 映射
2. diagnose_translation(py_code, ms_code)     → 检查翻译完整性
3. compute_model_complexity("resnet50")       → 对比模型大小
4. lint_mindspore_code(ms_code)               → 检查代码质量
```

### 工作流 3：AI 安全研究

```
1. recommend_models("图像分类")               → 选择目标模型
2. compute_model_complexity(model_name)       → 分析目标模型
3. generate_adversarial_attack("pgd", epsilon=0.3)  → 生成攻击配置
4. evaluate_model_robustness(model_config)   → 评估鲁棒性
5. quantize_model("dynamic")                  → 部署防御后模型
```

### 工作流 4：学习 MindSpore API

```
1. list_api_categories()                      → 查看所有 API 分类
2. search_apis("卷积")                        → 搜索相关 API
3. get_api_examples("nn.Conv2d")              → 获取 Conv2d 完整示例
4. get_related_apis("nn.Conv2d")             → 获取相关 API
5. get_quick_reference("nn.Conv2d")          → 快速查阅
```

---

## 📊 msutils 工具库统计

`msutils` 是集成在项目中的 MindSpore 开发工具库：

| 模块 | 功能 | 文件数 | 代码行数 |
|------|------|--------|---------|
| `data/` | 数据增强、加载器、变换 | 4 | 1,173 |
| `train/` | 回调函数、学习率调度器 | 3 | 976 |
| `security/` | 对抗攻击、防御、鲁棒性评估 | 4 | 855 |
| `eval/` | 评估指标 | 2 | 516 |
| `nlp/` | 文本增强、分词器 | 3 | 849 |
| `distributed/` | DDP 工具 | 2 | 398 |
| `deploy/` | 模型转换、量化 | 3 | 315 |
| `analysis/` | 复杂度分析、可视化 | 3 | 454 |
| **总计** | | **25** | **5,664** |

---

## 🔍 代码评分器（Linter）详解

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| ⚡ 性能 | 30% | 循环内创建对象、未使用混合精度等 |
| 🔌 兼容性 | 25% | PyTorch API 使用、废弃 API、Ascend 优化 |
| ✨ 最佳实践 | 25% | 随机种子、学习率调度、检查点、梯度裁剪 |
| 🔧 可维护性 | 20% | 函数长度、代码重复、文档字符串 |

### 评分等级

| 等级 | 分数范围 | 说明 |
|------|---------|------|
| A | 90-100 | 优秀，代码质量很高 |
| B | 80-89 | 良好，有少量可改进之处 |
| C | 70-79 | 一般，建议优化 |
| D | 60-69 | 较差，需要改进 |
| F | 0-59 | 不合格，存在严重问题 |

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 🐛 报告问题

- 在 GitHub Issues 中描述清晰的问题或建议
- 附上复现步骤和运行环境信息

### 💻 代码贡献

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add: xxx feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 📖 文档贡献

- 改善文档清晰度或补充缺失内容
- 翻译文档为其他语言
- 添加更多示例代码

### 🔧 维护数据

- 更新 MindSpore 官方模型清单
- 补充 PyTorch→MindSpore API 映射
- 扩展 API 示例注册表

---

## 📝 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](./LICENSE) 文件。

---

## 👤 作者信息

- **GitHub**: [@whitea133](https://github.com/whitea133)
- **Gitee**: [@whitea133](https://gitee.com/whitea133)
- **邮箱**: 1309848726@qq.com
- **学校**: 暨南大学 · 网络空间安全学院
- **指导教师**: 冯丙文（副研究员）

---

## 🙏 致谢

- [MindSpore](https://www.mindspore.cn/) — 华为全场景深度学习框架
- [MCP](https://modelcontextprotocol.io/) — Model Context Protocol
- 感谢所有为本项目贡献代码和文档的朋友！

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

[![Star](https://img.shields.io/github/stars/whitea133/mindspore-tools-mcp?style=social)](https://github.com/whitea133/mindspore-tools-mcp)

</div>