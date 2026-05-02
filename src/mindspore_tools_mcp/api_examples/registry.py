"""
MindSpore API 示例注册表
========================

包含常用 MindSpore API 的示例代码，按分类组织。

分类:
- nn: 神经网络层
- dataset: 数据处理
- ops: 算子
- optim: 优化器
- loss: 损失函数
- train: 训练相关
- cell: 网络单元
- common: 常用工具
"""

# =============================================================================
# API 注册表
# =============================================================================

API_REGISTRY = {

    # ==================== 神经网络层 (nn) ====================
    "nn.Conv2d": {
        "category": "nn",
        "category_cn": "神经网络层",
        "description": "二维卷积层，用于图像等2D信号处理",
        "signature": "nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False)",
        "parameters": [
            {"name": "in_channels", "type": "int", "desc": "输入通道数"},
            {"name": "out_channels", "type": "int", "desc": "输出通道数"},
            {"name": "kernel_size", "type": "int|tuple", "desc": "卷积核大小"},
            {"name": "stride", "type": "int|tuple", "desc": "步长，默认1"},
            {"name": "pad_mode", "type": "str", "desc": "'same'或'valid'或'pad'"},
            {"name": "padding", "type": "int", "desc": "边缘填充大小"},
            {"name": "dilation", "type": "int|tuple", "desc": "空洞卷积膨胀率"},
            {"name": "group", "type": "int", "desc": "分组卷积组数"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 创建卷积层
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, pad_mode='same')

# 输入: (batch_size, channels, height, width)
x = ms.Tensor.randn(1, 3, 224, 224)
output = conv(x)
print(output.shape)  # (1, 64, 224, 224)""",
            },
            {
                "title": "自定义卷积配置",
                "code": """import mindspore as ms
from mindspore import nn

# 3x3 卷积，步长为2
conv = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=2,
    pad_mode='valid',
    has_bias=True
)

x = ms.Tensor.randn(1, 64, 112, 112)
output = conv(x)
print(output.shape)  # (1, 128, 55, 55)""",
            },
            {
                "title": "深度可分离卷积",
                "code": """import mindspore as ms
from mindspore import nn

# 深度可分离卷积 = Depthwise + Pointwise
# Depthwise: 每个通道单独卷积
depthwise = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    group=64,  # group=in_channels 实现深度卷积
    pad_mode='same'
)
# Pointwise: 1x1 卷积融合通道
pointwise = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

x = ms.Tensor.randn(1, 64, 56, 56)
out = depthwise(x)
out = pointwise(out)
print(out.shape)  # (1, 128, 56, 56)""",
            },
        ],
        "related": ["nn.Conv1d", "nn.Conv3d", "nn.Dense", "nn.BatchNorm2d"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Conv2d.html",
    },

    "nn.BatchNorm2d": {
        "category": "nn",
        "category_cn": "神经网络层",
        "description": "批归一化层，对小批量数据进行归一化",
        "signature": "nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.99, affine=True, moving_mean_init='zeros', moving_var_init='ones')",
        "parameters": [
            {"name": "num_features", "type": "int", "desc": "特征通道数"},
            {"name": "eps", "type": "float", "desc": "防止除零的小常数"},
            {"name": "momentum", "type": "float", "desc": "移动平均动量"},
            {"name": "affine", "type": "bool", "desc": "是否使用可学习的γ和β"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 批归一化层
bn = nn.BatchNorm2d(num_features=64)

# 输入: (N, C, H, W)
x = ms.Tensor.randn(8, 64, 32, 32)
output = bn(x)
print(output.shape)  # (8, 64, 32, 32)
print(f"均值: {output.mean(axis=(0,2,3)).asnumpy()[:5]}")  # 接近0
print(f"方差: {output.var(axis=(0,2,3)).asnumpy()[:5]}")  # 接近1""",
            },
            {
                "title": "ResNet 中的应用",
                "code": """import mindspore as ms
from mindspore import nn

# ResNet 中的 BasicBlock
class ResidualBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, pad_mode='same', has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, pad_mode='same', has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 残差连接
        self.down_sample = None
        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.SequentialCell([
                nn.Conv2d(in_channels, out_channels, 1, stride, pad_mode='same', has_bias=False),
                nn.BatchNorm2d(out_channels)
            ])

    def construct(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down_sample:
            identity = self.down_sample(x)
        return self.relu(out + identity)""",
            },
        ],
        "related": ["nn.BatchNorm1d", "nn.BatchNorm3d", "nn.LayerNorm"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.BatchNorm2d.html",
    },

    "nn.Dense": {
        "category": "nn",
        "category_cn": "神经网络层",
        "description": "全连接层（线性变换）",
        "signature": "nn.Dense(in_channels, out_channels, has_bias=True, activation=None)",
        "parameters": [
            {"name": "in_channels", "type": "int", "desc": "输入特征维度"},
            {"name": "out_channels", "type": "int", "desc": "输出特征维度"},
            {"name": "has_bias", "type": "bool", "desc": "是否使用偏置"},
            {"name": "activation", "type": "str|nn.Cell", "desc": "激活函数"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 全连接层
fc = nn.Dense(in_channels=512, out_channels=10)

# 输入: (batch_size, in_channels)
x = ms.Tensor.randn(32, 512)
output = fc(x)
print(output.shape)  # (32, 10)""",
            },
            {
                "title": "带激活函数",
                "code": """import mindspore as ms
from mindspore import nn

# 全连接 + ReLU
fc_relu = nn.Dense(in_channels=512, out_channels=256, activation='relu')

x = ms.Tensor.randn(32, 512)
output = fc_relu(x)
print(output.shape)  # (32, 256)""",
            },
        ],
        "related": ["nn.Conv2d", "nn.Conv1d", "nn.BatchNorm2d"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Dense.html",
    },

    "nn.ReLU": {
        "category": "nn",
        "category_cn": "激活函数",
        "description": "ReLU 激活函数",
        "signature": "nn.ReLU()",
        "parameters": [],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

relu = nn.ReLU()
x = ms.Tensor([-1, 0, 1, 2], dtype=ms.float32)
output = relu(x)
print(output)  # [0, 0, 1, 2]""",
            },
        ],
        "related": ["nn.ReLU6", "nn.LeakyReLU", "nn.PReLU", "nn.Sigmoid", "nn.Tanh"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.ReLU.html",
    },

    "nn.Softmax": {
        "category": "nn",
        "category_cn": "激活函数",
        "description": "Softmax 激活函数",
        "signature": "nn.Softmax(axis=-1)",
        "parameters": [
            {"name": "axis", "type": "int", "desc": "进行 Softmax 的轴"},
        ],
        "examples": [
            {
                "title": "多分类任务",
                "code": """import mindspore as ms
from mindspore import nn

softmax = nn.Softmax(axis=-1)
logits = ms.Tensor([[2.0, 1.0, 0.1]], dtype=ms.float32)
probs = softmax(logits)
print(probs.asnumpy())
# 输出: [[0.6590012  0.24243298 0.09856589]]
print(probs.sum(axis=-1))  # 1.0""",
            },
        ],
        "related": ["nn.LogSoftmax", "nn.CrossEntropyLoss"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Softmax.html",
    },

    "nn.MaxPool2d": {
        "category": "nn",
        "category_cn": "池化层",
        "description": "二维最大池化",
        "signature": "nn.MaxPool2d(kernel_size, stride=None, pad_mode='valid', padding=0, dilation=1)",
        "parameters": [
            {"name": "kernel_size", "type": "int|tuple", "desc": "池化核大小"},
            {"name": "stride", "type": "int|tuple", "desc": "步长，默认等于kernel_size"},
            {"name": "pad_mode", "type": "str", "desc": "'same'或'valid'"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 输入: (N, C, H, W)
x = ms.Tensor.randn(1, 64, 224, 224)
output = pool(x)
print(output.shape)  # (1, 64, 112, 112)""",
            },
        ],
        "related": ["nn.AvgPool2d", "nn.AdaptiveMaxPool2d", "nn.AdaptiveAvgPool2d"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.MaxPool2d.html",
    },

    "nn.AvgPool2d": {
        "category": "nn",
        "category_cn": "池化层",
        "description": "二维平均池化",
        "signature": "nn.AvgPool2d(kernel_size, stride=None, pad_mode='valid', padding=0)",
        "parameters": [
            {"name": "kernel_size", "type": "int|tuple", "desc": "池化核大小"},
            {"name": "stride", "type": "int|tuple", "desc": "步长"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

pool = nn.AvgPool2d(kernel_size=2, stride=2)

x = ms.Tensor.randn(1, 64, 224, 224)
output = pool(x)
print(output.shape)  # (1, 64, 112, 112)""",
            },
        ],
        "related": ["nn.MaxPool2d", "nn.AdaptiveAvgPool2d"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.AvgPool2d.html",
    },

    "nn.Dropout": {
        "category": "nn",
        "category_cn": "正则化",
        "description": "Dropout 层，防止过拟合",
        "signature": "nn.Dropout(keep_prob=0.5)",
        "parameters": [
            {"name": "keep_prob", "type": "float", "desc": "保留概率"},
        ],
        "examples": [
            {
                "title": "训练和推理模式",
                "code": """import mindspore as ms
from mindspore import nn

dropout = nn.Dropout(keep_prob=0.8)

# 训练模式: 随机丢弃
dropout.set_train(True)
x = ms.Tensor.ones(4, 10, dtype=ms.float32)
output_train = dropout(x)
print(f"训练模式均值: {output_train.mean().asnumpy():.2f}")  # ~0.8

# 推理模式: 不丢弃，输出原值
dropout.set_train(False)
output_eval = dropout(x)
print(f"推理模式均值: {output_eval.mean().asnumpy():.2f}")  # 1.0""",
            },
        ],
        "related": ["nn.Dropout2d", "nn.Dropout1d"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Dropout.html",
    },

    "nn.LSTM": {
        "category": "nn",
        "category_cn": "循环神经网络",
        "description": "长短期记忆网络",
        "signature": "nn.LSTM(input_size, hidden_size, num_layers=1, has_bias=True, dropout=0, bidirectional=False)",
        "parameters": [
            {"name": "input_size", "type": "int", "desc": "输入特征维度"},
            {"name": "hidden_size", "type": "int", "desc": "隐藏层维度"},
            {"name": "num_layers", "type": "int", "desc": "LSTM 层数"},
            {"name": "bidirectional", "type": "bool", "desc": "是否双向"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True)

# 输入: (seq_len, batch_size, input_size)
x = ms.Tensor.randn(10, 32, 128)
output, (hn, cn) = lstm(x)
print(f"输出: {output.shape}")   # (10, 32, 512)  双向=hidden*2
print(f"隐藏: {hn.shape}")       # (4, 32, 256)   num_layers*2
print(f"细胞: {cn.shape}")       # (4, 32, 256)""",
            },
        ],
        "related": ["nn.RNN", "nn.GRU", "nn.LSTMCell"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.LSTM.html",
    },

    "nn.Embedding": {
        "category": "nn",
        "category_cn": "嵌入层",
        "description": "词嵌入层",
        "signature": "nn.Embedding(vocab_size, embedding_dim, padding_idx=None, embedding_lookup=None)",
        "parameters": [
            {"name": "vocab_size", "type": "int", "desc": "词汇表大小"},
            {"name": "embedding_dim", "type": "int", "desc": "嵌入维度"},
        ],
        "examples": [
            {
                "title": "词嵌入",
                "code": """import mindspore as ms
from mindspore import nn

# 词汇表大小10000，嵌入维度256
embedding = nn.Embedding(vocab_size=10000, embedding_dim=256)

# 输入: (batch_size, seq_len) 整数索引
x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.int32)
output = embedding(x)
print(output.shape)  # (2, 3, 256)""",
            },
        ],
        "related": ["nn.EmbeddingLookup"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Embedding.html",
    },

    # ==================== 损失函数 (loss) ====================
    "nn.CrossEntropyLoss": {
        "category": "loss",
        "category_cn": "损失函数",
        "description": "交叉熵损失，多分类分类任务常用",
        "signature": "nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)",
        "parameters": [
            {"name": "weight", "type": "Tensor", "desc": "各类别权重"},
            {"name": "ignore_index", "type": "int", "desc": "忽略的标签值"},
            {"name": "reduction", "type": "str", "desc": "'mean'/'sum'/'none'"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

loss_fn = nn.CrossEntropyLoss()

# 模拟10类分类
logits = ms.Tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.5]], dtype=ms.float32)
labels = ms.Tensor([0, 1], dtype=ms.int32)

loss = loss_fn(logits, labels)
print(f"损失值: {loss.asnumpy():.4f}")""",
            },
        ],
        "related": ["nn.SoftmaxCrossEntropyWithLogits", "nn.BCEWithLogitsLoss", "nn.MSELoss"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.CrossEntropyLoss.html",
    },

    "nn.BCEWithLogitsLoss": {
        "category": "loss",
        "category_cn": "损失函数",
        "description": "二分类交叉熵损失（ logits 输入）",
        "signature": "nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)",
        "parameters": [
            {"name": "weight", "type": "Tensor", "desc": "样本权重"},
            {"name": "pos_weight", "type": "Tensor", "desc": "正样本权重（用于不平衡数据）"},
        ],
        "examples": [
            {
                "title": "不平衡二分类",
                "code": """import mindspore as ms
from mindspore import nn

# 正样本权重为3，表示正样本更少
loss_fn = nn.BCEWithLogitsLoss(pos_weight=ms.Tensor([3.0]))

logits = ms.Tensor([[2.0], [-1.0], [0.5]], dtype=ms.float32)
labels = ms.Tensor([[1], [0], [1]], dtype=ms.float32)

loss = loss_fn(logits, labels)
print(f"损失值: {loss.asnumpy():.4f}")""",
            },
        ],
        "related": ["nn.CrossEntropyLoss", "nn.BCELoss"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.BCEWithLogitsLoss.html",
    },

    "nn.MSELoss": {
        "category": "loss",
        "category_cn": "损失函数",
        "description": "均方误差损失，回归任务常用",
        "signature": "nn.MSELoss(reduction='mean')",
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

loss_fn = nn.MSELoss()

predictions = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
targets = ms.Tensor([[1.1, 2.1], [2.9, 3.9]], dtype=ms.float32)

loss = loss_fn(predictions, targets)
print(f"损失值: {loss.asnumpy():.4f}")""",
            },
        ],
        "related": ["nn.L1Loss", "nn.SmoothL1Loss"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.MSELoss.html",
    },

    # ==================== 优化器 (optim) ====================
    "Adam": {
        "category": "optim",
        "category_cn": "优化器",
        "description": "Adam 优化器，最常用的优化器之一",
        "signature": "mindspore.nn.Adam(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)",
        "parameters": [
            {"name": "params", "type": "list", "desc": "待优化参数"},
            {"name": "learning_rate", "type": "float", "desc": "学习率"},
            {"name": "beta1", "type": "float", "desc": "一阶矩估计衰减率"},
            {"name": "beta2", "type": "float", "desc": "二阶矩估计衰减率"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 定义网络
net = nn.Dense(10, 2)
optimizer = nn.Adam(params=net.trainable_params(), learning_rate=1e-3)

# 在训练循环中使用
# optimizer(grads)  # grads 是梯度""",
            },
        ],
        "related": ["nn.AdamW", "nn.SGD", "nn.Momentum", "nn.RMSProp"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Adam.html",
    },

    "nn.AdamW": {
        "category": "optim",
        "category_cn": "优化器",
        "description": "AdamW 优化器，带权重衰减的 Adam（解耦权重衰减）",
        "signature": "mindspore.nn.AdamW(params, learning_rate=1e-3, weight_decay=1e-2)",
        "parameters": [
            {"name": "params", "type": "list", "desc": "待优化参数"},
            {"name": "learning_rate", "type": "float", "desc": "学习率"},
            {"name": "weight_decay", "type": "float", "desc": "权重衰减系数"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

net = nn.Dense(10, 2)
optimizer = nn.AdamW(params=net.trainable_params(), learning_rate=1e-3, weight_decay=1e-2)""",
            },
        ],
        "related": ["Adam", "nn.AdamWeightDecay"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.AdamW.html",
    },

    "nn.SGD": {
        "category": "optim",
        "category_cn": "优化器",
        "description": "随机梯度下降优化器",
        "signature": "mindspore.nn.SGD(params, learning_rate=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)",
        "parameters": [
            {"name": "params", "type": "list", "desc": "待优化参数"},
            {"name": "learning_rate", "type": "float", "desc": "学习率"},
            {"name": "momentum", "type": "float", "desc": "动量"},
        ],
        "examples": [
            {
                "title": "带动量 SGD",
                "code": """import mindspore as ms
from mindspore import nn

net = nn.Dense(10, 2)
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)""",
            },
        ],
        "related": ["nn.Momentum", "Adam", "nn.RMSProp"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.SGD.html",
    },

    # ==================== 数据集处理 (dataset) ====================
    "dataset": {
        "category": "dataset",
        "category_cn": "数据处理",
        "description": "MindSpore 数据集处理模块",
        "signature": "mindspore.dataset",
        "parameters": [],
        "examples": [
            {
                "title": "CIFAR-10 数据集",
                "code": """import mindspore.dataset as ds
from mindspore.dataset import vision, transforms

# 加载 CIFAR-10
data_dir = "./cifar10"
train_dataset = ds.Cifar10Dataset(data_dir=data_dir, usage="train")

# 数据预处理
train_dataset = train_dataset.map(
    operations=transforms.Compose([
        vision.Resize((224, 224)),
        vision.RandomHorizontalFlip(),
        vision.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201]),
        vision.HWC2CHW()
    ]),
    input_columns=["image"]
)

# 批量加载
train_dataset = train_dataset.batch(batch_size=32)

# 迭代
for image, label in train_dataset.create_tuple_iterator():
    print(image.shape, label.shape)""",
            },
            {
                "title": "ImageFolder 数据集",
                "code": """import mindspore.dataset as ds
from mindspore.dataset import vision, transforms

# 从文件夹自动加载图像分类数据集
# 目录结构: root/class1/img1.jpg, root/class2/img2.jpg
dataset = ds.ImageFolderDataset(dataset_dir="./data/train")

# 预处理
dataset = dataset.map(
    operations=transforms.Compose([
        vision.Decode(),
        vision.Resize(256),
        vision.CenterCrop(224),
        vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        vision.HWC2CHW()
    ]),
    input_columns=["image"]
)

dataset = dataset.batch(32)""",
            },
        ],
        "related": ["dataset.transforms", "dataset.vision", "MnistDataset", "Cifar10Dataset"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/dataset/mindspore.dataset.html",
    },

    "MnistDataset": {
        "category": "dataset",
        "category_cn": "数据处理",
        "description": "MNIST 手写数字数据集",
        "signature": "mindspore.dataset.MnistDataset(dataset_dir, usage=None, num_samples=None)",
        "parameters": [
            {"name": "dataset_dir", "type": "str", "desc": "数据集路径"},
            {"name": "usage", "type": "str", "desc": "'train'/'test'/'all'"},
        ],
        "examples": [
            {
                "title": "加载 MNIST",
                "code": """import mindspore.dataset as ds
from mindspore.dataset import vision, transforms

# 加载 MNIST
train_dataset = ds.MnistDataset(dataset_dir="./mnist", usage="train")

# 预处理: 归一化 + reshape
train_dataset = train_dataset.map(
    operations=transforms.Compose([
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW()
    ]),
    input_columns=["image"]
)
train_dataset = train_dataset.batch(32)

for image, label in train_dataset.create_tuple_iterator():
    print(image.shape)  # (32, 1, 28, 28)""",
            },
        ],
        "related": ["Cifar10Dataset", "dataset"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/dataset/mindspore.dataset.MnistDataset.html",
    },

    # ==================== 训练 (train) ====================
    "nn.TrainOneStepCell": {
        "category": "train",
        "category_cn": "训练",
        "description": "单步训练网络单元",
        "signature": "mindspore.nn.TrainOneStepCell(network, optimizer, sens=None)",
        "parameters": [
            {"name": "network", "type": "Cell", "desc": "网络模型"},
            {"name": "optimizer", "type": "Cell", "desc": "优化器"},
        ],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 定义网络
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(10, 1)

    def construct(self, x):
        return self.fc(x)

net = Net()
optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-3)
loss_fn = nn.MSELoss()

# 单步训练单元
train_net = nn.TrainOneStepCell(
    nn.WithLossCell(net, loss_fn),
    optimizer
)

# 训练
x = ms.Tensor.randn(32, 10)
y = ms.Tensor.randn(32, 1)
loss = train_net(x, y)
print(f"Loss: {loss.asnumpy():.4f}")""",
            },
        ],
        "related": ["nn.WithLossCell", "nn.Model", "Model.train"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.TrainOneStepCell.html",
    },

    "nn.Model": {
        "category": "train",
        "category_cn": "训练",
        "description": "MindSpore 高层训练/推理封装",
        "signature": "mindspore.nn.Model(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None)",
        "parameters": [
            {"name": "network", "type": "Cell", "desc": "网络模型"},
            {"name": "loss_fn", "type": "Cell", "desc": "损失函数"},
            {"name": "optimizer", "type": "Cell", "desc": "优化器"},
        ],
        "examples": [
            {
                "title": "训练模型",
                "code": """import mindspore as ms
from mindspore import nn

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(10, 1)

    def construct(self, x):
        return self.fc(x)

net = Net()

# 创建模型
model = nn.ModelNet(
    network=net,
    loss_fn=nn.MSELoss(),
    optimizer=nn.Adam(net.trainable_params(), learning_rate=1e-3),
    metrics={'mae': nn.MAE()}
)

# 训练
# model.train(epoch=10, train_dataset=train_ds, callbacks=[...])""",
            },
        ],
        "related": ["nn.TrainOneStepCell", "Model.train", "Model.eval"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Model.html",
    },

    "nn.DynamicLossScaleUpdateCell": {
        "category": "train",
        "category_cn": "训练",
        "description": "动态损失缩放，用于混合精度训练",
        "signature": "mindspore.nn.DynamicLossScaleUpdateCell(loss_scale_value, scale_factor=2, scale_window=1000)",
        "parameters": [
            {"name": "loss_scale_value", "type": "float", "desc": "初始损失缩放值"},
            {"name": "scale_factor", "type": "int", "desc": "溢出后缩放倍数"},
            {"name": "scale_window", "type": "int", "desc": "正常更新窗口"},
        ],
        "examples": [
            {
                "title": "混合精度训练",
                "code": """import mindspore as ms
from mindspore import nn

# 定义网络和优化器
net = MyNet()
optimizer = nn.Adam(net.trainable_params(), learning_rate=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 动态损失缩放
loss_scale = nn.DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)

# 训练网络
train_net = nn.TrainOneStepCell(
    nn.WithLossCell(net, loss_fn),
    optimizer,
    loss_scale=loss_scale
)""",
            },
        ],
        "related": ["nn.TrainOneStepCell", "nn.FixedLossScaleUpdateCell"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.DynamicLossScaleUpdateCell.html",
    },

    # ==================== 学习率调度 (lr) ====================
    "nn.cosine_decay_lr": {
        "category": "lr",
        "category_cn": "学习率调度",
        "description": "余弦退火学习率衰减",
        "signature": "mindspore.nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch=-1)",
        "parameters": [
            {"name": "min_lr", "type": "float", "desc": "最小学习率"},
            {"name": "max_lr", "type": "float", "desc": "最大学习率"},
            {"name": "total_step", "type": "int", "desc": "总步数"},
            {"name": "step_per_epoch", "type": "int", "desc": "每epoch步数"},
        ],
        "examples": [
            {
                "title": "余弦退火调度",
                "code": """import mindspore as ms
from mindspore import nn

# 参数设置
min_lr = 1e-6
max_lr = 1e-3
total_step = 10000
step_per_epoch = 100

# 计算学习率
lr_list = nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch=80)

print(f"初始学习率: {lr_list[0]:.6f}")  # 接近 0.001
print(f"最终学习率: {lr_list[-1]:.6f}")  # 接近 0.000001
print(f"总长度: {len(lr_list)}")""",
            },
        ],
        "related": ["nn.step_lr", "nn.exponential_decay_lr", "nn.polynomial_decay_lr"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.cosine_decay_lr.html",
    },

    "nn.step_lr": {
        "category": "lr",
        "category_cn": "学习率调度",
        "description": "阶梯式学习率衰减",
        "signature": "mindspore.nn.step_lr(learning_rate, epoch_size, gamma=0.1)",
        "parameters": [
            {"name": "learning_rate", "type": "float", "desc": "初始学习率"},
            {"name": "epoch_size", "type": "int", "desc": "衰减周期"},
            {"name": "gamma", "type": "float", "desc": "衰减倍数"},
        ],
        "examples": [
            {
                "title": "阶梯衰减",
                "code": """import mindspore as ms
from mindspore import nn

# 每30个epoch，学习率乘以0.1
lr_scheduler = nn.step_lr(learning_rate=0.1, epoch_size=30, gamma=0.1)""",
            },
        ],
        "related": ["nn.cosine_decay_lr", "nn.exponential_decay_lr"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.step_lr.html",
    },

    # ==================== 回调函数 (callbacks) ====================
    "nn.Callback": {
        "category": "callbacks",
        "category_cn": "回调函数",
        "description": "训练回调基类",
        "signature": "mindspore.nn.Callback",
        "parameters": [],
        "examples": [
            {
                "title": "自定义回调",
                "code": """import mindspore as ms
from mindspore import nn

class MyCallback(nn.Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, run_context):
        print("训练开始!")

    def on_train_epoch_begin(self, run_context):
        print(f"Epoch {self.cur_epoch_num} 开始")

    def on_train_epoch_end(self, run_context):
        print(f"Epoch {self.cur_epoch_num} 结束")

    def on_train_end(self, run_context):
        print("训练结束!")

# 使用
callbacks = [MyCallback()]
# model.train(epoch=10, train_dataset=ds, callbacks=callbacks)""",
            },
        ],
        "related": ["nn.TimeMonitor", "nn.LossMonitor", "nn.ModelCheckpoint"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Callback.html",
    },

    "nn.TimeMonitor": {
        "category": "callbacks",
        "category_cn": "回调函数",
        "description": "监控训练时间",
        "signature": "mindspore.nn.TimeMonitor(data_size=None, tensor_size=None)",
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 记录每个epoch和step的训练时间
time_cb = nn.TimeMonitor(data_size=train_dataset.get_dataset_size())
# model.train(epoch=10, train_dataset=train_dataset, callbacks=[time_cb])""",
            },
        ],
        "related": ["nn.LossMonitor", "nn.ModelCheckpoint"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.TimeMonitor.html",
    },

    "nn.ModelCheckpoint": {
        "category": "callbacks",
        "category_cn": "回调函数",
        "description": "模型检查点保存",
        "signature": "mindspore.nn.ModelCheckpoint(directory, prefix='CKP', checkpoint_format='ckpt', save_checkpoint_steps=1, keep_checkpoint_max=5, metric_names=['loss'])",
        "parameters": [
            {"name": "directory", "type": "str", "desc": "保存目录"},
            {"name": "prefix", "type": "str", "desc": "文件名前缀"},
            {"name": "save_checkpoint_steps", "type": "int", "desc": "保存间隔"},
            {"name": "keep_checkpoint_max", "type": "int", "desc": "最大保存数量"},
        ],
        "examples": [
            {
                "title": "保存检查点",
                "code": """import mindspore as ms
from mindspore import nn

# 每5个epoch保存一次，最多保留10个
ckpt_cb = nn.ModelCheckpoint(
    directory="./checkpoints",
    prefix="resnet50",
    save_checkpoint_steps=5,
    keep_checkpoint_max=10
)
# model.train(epoch=90, train_dataset=ds, callbacks=[ckpt_cb])""",
            },
        ],
        "related": ["nn.LossMonitor", "nn.TimeMonitor"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-lores/mindspore.nn.ModelCheckpoint.html",
    },

    # ==================== 网络单元 (cell) ====================
    "nn.Cell": {
        "category": "cell",
        "category_cn": "网络单元",
        "description": "所有神经网络模块的基类",
        "signature": "mindspore.nn.Cell(auto_prefix=True)",
        "parameters": [
            {"name": "auto_prefix", "type": "bool", "desc": "自动生成参数名前缀"},
        ],
        "examples": [
            {
                "title": "定义自定义网络",
                "code": """import mindspore as ms
from mindspore import nn

class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Dense(64, 10)

    def construct(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.fc(x.flatten())
        return x

net = MyNet()
print(net.trainable_params())""",
            },
        ],
        "related": ["nn.SequentialCell", "nn.CellList"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.Cell.html",
    },

    "nn.SequentialCell": {
        "category": "cell",
        "category_cn": "网络单元",
        "description": "顺序容器，按顺序执行各层",
        "signature": "mindspore.nn.SequentialCell(*args)",
        "parameters": [],
        "examples": [
            {
                "title": "基础用法",
                "code": """import mindspore as ms
from mindspore import nn

# 顺序组合网络层
backbone = nn.SequentialCell(
    nn.Conv2d(3, 64, 3, pad_mode='same'),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 128, 3, pad_mode='same'),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1))
)

x = ms.Tensor.randn(1, 3, 224, 224)
output = backbone(x)
print(output.shape)  # (1, 128, 1, 1)""",
            },
        ],
        "related": ["nn.Cell", "nn.CellList"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.SequentialCell.html",
    },

    # ==================== 常用工具 (common) ====================
    "ms.load_checkpoint": {
        "category": "common",
        "category_cn": "模型加载保存",
        "description": "加载检查点文件",
        "signature": "mindspore.load_checkpoint(ckpt_file_name, net=None, strict_load=False, decide=ParameterFormat.DICT)",
        "parameters": [
            {"name": "ckpt_file_name", "type": "str", "desc": "检查点文件路径"},
            {"name": "net", "type": "Cell", "desc": "目标网络"},
            {"name": "strict_load", "type": "bool", "desc": "严格匹配参数名"},
        ],
        "examples": [
            {
                "title": "加载检查点",
                "code": """import mindspore as ms

# 加载参数到网络
param_dict = ms.load_checkpoint("./resnet50.ckpt")
ms.load_param_into_net(net, param_dict)
print("模型加载成功!")""",
            },
        ],
        "related": ["ms.save_checkpoint", "ms.load_param_into_net"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.load_checkpoint.html",
    },

    "ms.save_checkpoint": {
        "category": "common",
        "category_cn": "模型加载保存",
        "description": "保存检查点文件",
        "signature": "mindspore.save_checkpoint(ckpt_file_name, net=None, meta_desc=None, append_dict=None)",
        "parameters": [
            {"name": "ckpt_file_name", "type": "str", "desc": "保存路径"},
            {"name": "net", "type": "Cell", "desc": "网络实例"},
        ],
        "examples": [
            {
                "title": "保存检查点",
                "code": """import mindspore as ms

# 保存网络参数
ms.save_checkpoint(net, "./output/model.ckpt")

# 保存时附加信息
ms.save_checkpoint(
    net,
    "./output/model.ckpt",
    append_dict={
        "epoch": 50,
        "best_acc": 0.95,
        "optim": optimizer
    }
)""",
            },
        ],
        "related": ["ms.load_checkpoint", "ms.load_param_into_net"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.save_checkpoint.html",
    },

    # ==================== 算子 (ops) ====================
    "ops.cat": {
        "category": "ops",
        "category_cn": "算子",
        "description": "拼接张量",
        "signature": "mindspore.ops.cat(tensors, axis=0)",
        "parameters": [
            {"name": "tensors", "type": "list", "desc": "待拼接的张量列表"},
            {"name": "axis", "type": "int", "desc": "拼接轴"},
        ],
        "examples": [
            {
                "title": "拼接张量",
                "code": """import mindspore as ms
from mindspore import ops

a = ms.Tensor([[1, 2], [3, 4]], dtype=ms.float32)
b = ms.Tensor([[5, 6]], dtype=ms.float32)

# 按行拼接 (axis=0)
c = ops.cat([a, b], axis=0)
print(c)
# [[1. 2.]
#  [3. 4.]
#  [5. 6.]]

# 按列拼接 (axis=1)
d = ops.cat([a, a], axis=1)
print(d)
# [[1. 2. 1. 2.]
#  [3. 4. 3. 4.]]""",
            },
        ],
        "related": ["ops.stack", "ops.split"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-ops/mindspore.ops.cat.html",
    },

    "ops.stack": {
        "category": "ops",
        "category_cn": "算子",
        "description": "沿新轴堆叠张量",
        "signature": "mindspore.ops.stack(tensors, axis=0)",
        "parameters": [
            {"name": "tensors", "type": "list", "desc": "待堆叠的张量列表"},
            {"name": "axis", "type": "int", "desc": "新轴位置"},
        ],
        "examples": [
            {
                "title": "堆叠张量",
                "code": """import mindspore as ms
from mindspore import ops

a = ms.Tensor([1, 2, 3], dtype=ms.float32)
b = ms.Tensor([4, 5, 6], dtype=ms.float32)

# 沿新轴堆叠
c = ops.stack([a, b], axis=0)
print(c.shape)  # (2, 3)
print(c)
# [[1. 2. 3.]
#  [4. 5. 6.]]""",
            },
        ],
        "related": ["ops.cat"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-ops/mindspore.ops.stack.html",
    },

    "ops.Reshape": {
        "category": "ops",
        "category_cn": "算子",
        "description": "重塑张量形状",
        "signature": "mindspore.ops.Reshape()(x, shape)",
        "parameters": [
            {"name": "x", "type": "Tensor", "desc": "输入张量"},
            {"name": "shape", "type": "tuple|list", "desc": "目标形状"},
        ],
        "examples": [
            {
                "title": "重塑张量",
                "code": """import mindspore as ms
from mindspore import ops

x = ms.Tensor.randn(1, 3, 224, 224)
reshaped = ops.Reshape()(x, (1, -1))
print(reshaped.shape)  # (1, 150528)""",
            },
        ],
        "related": ["ops.Transpose", "ops.Squeeze", "ops.ExpandDims"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-ops/mindspore.ops.Reshape.html",
    },

    # ==================== 分布式训练 (distributed) ====================
    "nn.DistributedSampler": {
        "category": "distributed",
        "category_cn": "分布式训练",
        "description": "分布式数据采样器",
        "signature": "mindspore.dataset.DistributedSampler(num_shards, shard_id, shuffle=True, num_samples=None, offset=0)",
        "parameters": [
            {"name": "num_shards", "type": "int", "desc": "总进程数"},
            {"name": "shard_id", "type": "int", "desc": "当前进程号"},
        ],
        "examples": [
            {
                "title": "分布式数据加载",
                "code": """import mindspore.dataset as ds

# 创建分布式采样器
sampler = ds.DistributedSampler(num_shards=8, shard_id=0, shuffle=True)

# 使用采样器加载数据
dataset = ds.Cifar10Dataset(data_dir="./data", sampler=sampler)
dataset = dataset.batch(32)""",
            },
        ],
        "related": ["nn.ParallelMode", "ms.context.set_auto_parallel_context"],
        "official_doc": "https://www.mindspore.cn/docs/zh-CN/r2.6/api/api-layers/mindspore.nn.DistributedSampler.html",
    },
}
