# ResNet50 示例项目

通过此项目学习如何使用 msutils 进行模型训练和鲁棒性评估。

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练模型](#训练模型)
- [评估鲁棒性](#评估鲁棒性)
- [使用 MCP](#使用-mcp)

---

## 快速开始

### Step 1: 训练模型

```bash
cd examples/resnet50_example
python train.py
```

训练完成后模型会保存到 `current_run/resnet50_best.ckpt`

### Step 2: 评估鲁棒性

```bash
python evaluate_robustness.py
```

---

## 项目结构

```
resnet50_example/
├── train.py                  # 训练脚本
├── evaluate_robustness.py    # 鲁棒性评估脚本
└── README.md                # 本文件
```

---

## 训练模型

### 运行训练

```bash
python train.py
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 50 | 训练轮数 |
| batch_size | 32 | 批次大小 |
| learning_rate | 0.01 | 学习率 |
| num_samples | 5000 | 训练样本数 |

### 训练输出

训练完成后会生成：
- `current_run/resnet50_best.ckpt` - 模型权重文件
- `current_run/model_info.txt` - 模型信息

---

## 评估鲁棒性

### 运行评估

```bash
python evaluate_robustness.py
```

### 评估内容

| 攻击方法 | epsilon | 说明 |
|----------|---------|------|
| FGSM | 0.03 | 快速梯度符号攻击 |
| PGD | 0.03 | 投影梯度下降攻击 |

### 评估输出

| 指标 | 说明 |
|------|------|
| 干净样本准确率 | 正常样本的分类准确率 |
| FGSM 攻击后准确率 | FGSM 对抗样本的准确率 |
| PGD 攻击后准确率 | PGD 对抗样本的准确率 |

---

## 使用 MCP

### 获取评估代码

在支持 MCP 的 AI 助手中输入：

```
帮我生成一个评估 ResNet50 鲁棒性的代码
```

### 获取攻击配置

```
生成 FGSM 和 PGD 攻击配置，epsilon=0.03
```

### 获取优化建议

```
如何提高模型对对抗攻击的抵抗力？
```

---

## 预期结果

### 干净样本

| 指标 | 预期值 |
|------|--------|
| 准确率 | 60-80% |

### 对抗攻击后

| 攻击 | 预期准确率下降 |
|------|---------------|
| FGSM (ε=0.03) | 20-40% |
| PGD (ε=0.03) | 30-50% |

---

## 下一步

1. 尝试不同的 epsilon 值
2. 使用对抗训练提高模型鲁棒性
3. 测试其他攻击方法

---

## 常见问题

### Q: 训练太慢怎么办？
A: 可以减少 `num_samples` 参数，或使用 GPU

### Q: 模型文件不存在？
A: 确保先运行 `train.py`

### Q: 如何提高模型鲁棒性？
A: 使用对抗训练 (Adversarial Training)
