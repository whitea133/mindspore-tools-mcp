# mindspore-tools-mcp

<div align="center">

![MindSpore](https://img.shields.io/badge/MindSpore-2.0+-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-Ready-purple?style=flat-square)

**基于 MCP 的 MindSpore 开发工具套件**

提供官方模型清单查询、智能模型推荐、PyTorch→MindSpore API 映射、以及 **AI 安全、数据处理、训练工具** 等开发必备功能。

[English](./README.md) | [中文](./README_CN.md)

</div>

---

## ✨ 功能特性

### 🔍 官方模型检索

- `list_models` - 支持按 group/category/task/suite 或关键词过滤
- `get_model_info` - 返回单模型详情
- `mindspore://models/official` - 资源端点提供完整模型清单

### 🤖 智能模型推荐

- `recommend_models` - 根据自然语言描述智能推荐合适的模型
  - 支持任务描述: "图像分类"、"文本生成"、"OCR"、"推荐系统" 等
  - 支持硬件约束: ascend、gpu、cpu
  - 提供推荐理由和匹配分数
- `compare_models` - 对比多个模型，帮助选择最适合的模型

### 🗺️ API 映射工具

- `query_op_mapping` - 支持 section 过滤与模糊匹配
- `diagnose_translation` - 检查 PyTorch→MindSpore 代码翻译是否完整
- `mindspore://opmap/...` - 资源暴露 PyTorch→MindSpore API 映射

### 🛡️ AI 安全工具 (NEW!)

- `generate_adversarial_attack` - 生成对抗攻击配置和示例代码
  - 支持: FGSM、PGD、DeepFool、CW、JSMA
- `evaluate_model_robustness` - 评估模型鲁棒性配置

### 📊 数据处理工具 (NEW!)

- `create_data_augmentation_pipeline` - 创建数据增强流水线
  - 支持: 图像分类、目标检测、语义分割、NLP

### 🔧 训练工具 (NEW!)

- `get_lr_scheduler` - 获取学习率调度器配置
  - 支持: 余弦退火、阶梯衰减、多项式衰减、One Cycle
- `get_training_callbacks` - 获取训练回调函数配置
  - 支持: 检查点保存、早停、TensorBoard、梯度裁剪

### 📈 分析工具 (NEW!)

- `compute_model_complexity` - 计算模型 FLOPs、参数量、内存占用

### 🌐 分布式训练 (NEW!)

- `setup_distributed_training` - 配置分布式训练

### 🚀 部署工具 (NEW!)

- `quantize_model` - 模型量化配置
  - 支持: 动态量化、静态量化、量化感知训练
- `convert_model_format` - 模型格式转换配置

### 🔍 代码评分器 (NEW!)

- `lint_mindspore_code` - MindSpore 代码质量评分
  - 4 个维度评分: 性能、兼容性、最佳实践、可维护性
  - 26 条检查规则，自动发现问题并给出建议
- `get_lint_rules` - 获取所有检查规则列表
- `compare_code_snippets` - 对比两个代码片段的质量

### 📊 数据脚本

- `scripts/update_model_list.py` - 更新官方模型 JSON
- `scripts/fetch_api_mapping.py` - 抓取并刷新 API 映射

---

## 📦 目录结构

```
mindspore-tools-mcp/
├── data/                              # 官方模型与 API 映射数据
│   ├── convert/                       # 分 section 的映射分片
│   ├── mindspore_official_models.json
│   ├── pytorch_ms_api_mapping_consistent.json
│   └── pytorch_ms_api_mapping_diff.json
│
├── scripts/                           # 数据/映射更新脚本
│   ├── update_model_list.py
│   └── fetch_api_mapping.py
│
├── src/
│   └── mindspore_tools_mcp/           # MCP 服务
│       ├── server.py                   # MCP 入口
│       ├── tools.py                    # 模型检索工具
│       ├── msutils_tools.py            # 🆕 msutils MCP 工具封装
│       ├── linter_tools.py             # 🆕 代码评分器 MCP 封装
│       ├── msutils/                    # 🆕 MindSpore 开发工具库
│       │   ├── data/                   # 数据处理
│       │   ├── train/                  # 训练工具
│       │   ├── security/               # AI 安全
│       │   ├── eval/                   # 评估指标
│       │   ├── nlp/                    # NLP 工具
│       │   ├── distributed/            # 分布式训练
│       │   ├── deploy/                 # 部署工具
│       │   └── analysis/               # 分析可视化
│       └── linter/                     # 🆕 代码评分器核心
│           ├── __init__.py
│           ├── rules.py                 # 检查规则定义
│           ├── checker.py               # 检查逻辑
│           └── formatter.py             # 报告格式化
│       ├── resource.py                 # 资源定义
│       └── prompt.py                   # Prompt 注册
│
├── examples/                          # 🆕 使用示例
├── tests/                             # 测试文件
├── pyproject.toml                     # 项目配置
└── uv.lock                            # 依赖锁文件
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
uv sync
```

### 2️⃣ 启动 MCP 服务

```bash
uv run python -m mindspore_tools_mcp.server
```

### 3️⃣ 客户端配置

```jsonc
// cline_mcp_settings.json
{
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
```

---

## 📖 API 参考

### 模型检索工具

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `list_models` | 列出模型 | `list_models(task="text-generation")` |
| `get_model_info` | 获取模型详情 | `get_model_info("llama2")` |
| `recommend_models` | 智能模型推荐 | `recommend_models("图像分类")` |
| `compare_models` | 对比模型 | `compare_models(["resnet50", "vit"])` |
| `query_op_mapping` | 查询 API 映射 | `query_op_mapping("torch.add")` |
| `diagnose_translation` | 诊断代码翻译 | `diagnose_translation(py_code, ms_code)` |

### AI 安全工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `generate_adversarial_attack` | 生成对抗攻击配置 | `generate_adversarial_attack("fgsm", epsilon=0.1)` |
| `evaluate_model_robustness` | 评估模型鲁棒性 | `evaluate_model_robustness(model_config)` |

### 数据处理工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `create_data_augmentation_pipeline` | 创建数据增强流水线 | `create_data_augmentation_pipeline("image_classification")` |

### 训练工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `get_lr_scheduler` | 获取学习率调度器 | `get_lr_scheduler("cosine_annealing", total_epochs=100)` |
| `get_training_callbacks` | 获取训练回调配置 | `get_training_callbacks(["checkpoint", "early_stopping"])` |

### 分析工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `compute_model_complexity` | 计算模型复杂度 | `compute_model_complexity("resnet50")` |

### 分布式训练 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `setup_distributed_training` | 配置分布式训练 | `setup_distributed_training(num_gpus=8)` |

### 部署工具 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `quantize_model` | 模型量化配置 | `quantize_model("dynamic", precision="int8")` |
| `convert_model_format` | 模型格式转换 | `convert_model_format("pytorch", "mindspore")` |

### 代码评分器 🆕

| 工具名 | 说明 | 示例 |
|--------|------|------|
| `lint_mindspore_code` | 代码质量评分 | `lint_mindspore_code(code, level="all")` |
| `get_lint_rules` | 获取检查规则 | `get_lint_rules(category="performance")` |
| `compare_code_snippets` | 对比代码质量 | `compare_code_snippets(code_a, code_b)` |

### 资源列表

| 资源名 | 说明 |
|--------|------|
| `mindspore://models/official` | 官方模型完整清单 |
| `mindspore://opmap/pytorch/consistent` | 一致 API 映射 |
| `mindspore://opmap/pytorch/diff` | 差异 API 映射 |

### Prompt 列表

| Prompt 名 | 说明 |
|-----------|------|
| `model_lookup` | 按任务查找模型 |
| `model_recommend` | 智能模型推荐 |
| `model_compare` | 模型对比 |
| `migration_guide` | 迁移指南 |
| `performance_optimize` | 性能优化建议 |

---

## 💡 使用示例

### 智能模型推荐

```python
# 推荐"图像分类"模型
recommend_models("图像分类", limit=3)
# 返回: resnet18, resnet34, resnet50 (带推荐理由和性能指标)

# 推荐支持 Ascend 的文本生成模型
recommend_models("文本生成大模型", hardware="ascend", limit=3)
# 返回: llama2, qwen, baichuan2

# OCR 模型推荐
recommend_models("OCR文字识别", limit=5)
# 返回: dbnet_resnet18, dbnet_resnet50, ...
```

### AI 安全对抗攻击 🆕

```python
# 生成 FGSM 攻击配置
generate_adversarial_attack("fgsm", epsilon=0.1)
# 返回: 攻击配置 + MindSpore 示例代码

# 生成 PGD 攻击配置
generate_adversarial_attack("pgd", epsilon=0.3, num_iterations=40)
# 返回: 迭代攻击配置 + 代码示例
```

### 数据增强流水线 🆕

```python
# 创建图像分类增强流水线
create_data_augmentation_pipeline("image_classification")
# 返回: 增强方法列表 + MindSpore Dataset 代码

# 创建 NLP 数据增强
create_data_augmentation_pipeline("nlp", augmentations=["RandomDelete", "SynonymReplace"])
```

### 学习率调度 🆕

```python
# 余弦退火调度
get_lr_scheduler("cosine_annealing", total_epochs=100, warmup_epochs=5)
# 返回: 调度器配置 + 学习率曲线 + MindSpore 代码
```

### 模型复杂度分析 🆕

```python
# 分析 ResNet50 复杂度
compute_model_complexity("resnet50")
# 返回: FLOPs=4.12G, Params=25.6M, Memory=98MB
```

### 模型量化 🆕

```python
# 动态量化配置
quantize_model("dynamic", precision="int8")
# 返回: 量化配置 + MindSpore 代码 + 预期加速比

# 静态量化配置
quantize_model("static", precision="int8", calibration_dataset_size=100)
```

### 代码评分器 🆕

```python
# 检查代码质量
code = '''
import mindspore as ms
from mindspore import nn

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
'''

result = lint_mindspore_code(code)
# 返回: score=85, grade="B", dimensions={...}, issues=[...]

# 对比两个代码
compare_code_snippets(good_code, bad_code)
# 返回: winner, score差异, 问题差异
```

---

## 📊 msutils 模块说明

`msutils` 是集成在项目中的 MindSpore 开发工具库，包含以下模块：

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

## 🔍 代码评分器 (Linter)

`linter` 模块提供 MindSpore 代码质量评分功能，帮助开发者发现代码问题并给出改进建议。

### 评分维度

| 维度 | 权重 | 说明 |
|------|------|------|
| ⚡ 性能 | 30% | 循环内创建对象、未使用混合精度等 |
| 🔌 兼容性 | 25% | PyTorch API 使用、废弃 API、Ascend 优化 |
| ✨ 最佳实践 | 25% | 随机种子、学习率调度、检查点、梯度裁剪 |
| 🔧 可维护性 | 20% | 函数长度、代码重复、文档字符串 |

### 检查规则

| 类别 | 规则数 | 示例 |
|------|--------|------|
| 性能规则 | 6 | 循环内创建优化器、未启用 AMP |
| 兼容性规则 | 5 | 使用 torch API、废弃 API |
| 最佳实践 | 7 | 未设置种子、未使用检查点 |
| 可维护性 | 5 | 函数过长、缺少文档 |
| **总计** | **26** | |

### 评分等级

| 等级 | 分数范围 | 说明 |
|------|---------|------|
| A | 90-100 | 优秀，代码质量很高 |
| B | 80-89 | 良好，有少量可改进之处 |
| C | 70-79 | 一般，建议优化 |
| D | 60-69 | 较差，需要改进 |
| F | 0-59 | 不合格，存在严重问题 |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📝 许可

MIT License - 详见 LICENSE 文件

---

## 👤 作者

- **GitHub**: [@whitea133](https://github.com/whitea133)
- **邮箱**: 1309848726@qq.com

---

## 🙏 致谢

- [MindSpore](https://www.mindspore.cn/) - 华为全场景深度学习框架
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐**

</div>