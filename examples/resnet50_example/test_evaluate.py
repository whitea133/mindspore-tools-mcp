import mindspore as ms
import numpy as np
import os

# 1. 首先查看保存的模型参数结构
def inspect_model_weights(model_path):
    """查看模型权重结构"""
    print(f"检查模型: {model_path}")
    
    try:
        param_dict = ms.load_checkpoint(model_path)
        print(f"参数总数: {len(param_dict)}")
        
        print("\n参数名称和形状:")
        for i, (name, param) in enumerate(param_dict.items()):
            shape = param.shape if hasattr(param, 'shape') else 'unknown'
            print(f"{i+1:3d}. {name}: {shape}")
            
            # 只显示前10个参数
            if i >= 9:
                print("... (更多参数)")
                break
        
        # 特别查看全连接层参数
        print("\n全连接层参数:")
        fc_params = {k: v for k, v in param_dict.items() if 'fc' in k or 'dense' in k}
        for name, param in fc_params.items():
            shape = param.shape if hasattr(param, 'shape') else 'unknown'
            print(f"{name}: {shape}")
            
    except Exception as e:
        print(f"检查失败: {e}")

# 2. 根据参数形状推断模型结构
def create_matching_model(param_dict):
    """根据参数形状创建匹配的模型"""
    # 分析卷积层参数
    conv_layers = {}
    fc_layers = {}
    
    for name, param in param_dict.items():
        if hasattr(param, 'shape'):
            if 'conv' in name or 'weight' in name and len(param.shape) == 4:
                conv_layers[name] = param.shape
            elif 'fc' in name or 'dense' in name or ('weight' in name and len(param.shape) == 2):
                fc_layers[name] = param.shape
    
    print("\n推断的模型结构:")
    print("卷积层:")
    for name, shape in conv_layers.items():
        print(f"  {name}: {shape}")
    
    print("\n全连接层:")
    for name, shape in fc_layers.items():
        print(f"  {name}: {shape}")
    
    # 根据 fc1.weight 的形状推断输入尺寸
    if 'fc1.weight' in fc_layers:
        fc1_shape = fc_layers['fc1.weight']
        print(f"\n推断信息:")
        print(f"- fc1 输入维度: {fc1_shape[1]}")
        print(f"- fc1 输出维度: {fc1_shape[0]}")
        
        # 尝试推断图像尺寸
        # fc1 输入维度 = 通道数 * 高度 * 宽度
        # 假设通道数为64（从conv2的输出推断）
        if 'conv2.weight' in conv_layers:
            conv2_out_channels = conv_layers['conv2.weight'][0]
            print(f"- conv2 输出通道数: {conv2_out_channels}")
            
            # 计算特征图尺寸
            feature_size = fc1_shape[1] // conv2_out_channels
            height = int(np.sqrt(feature_size))
            width = height
            print(f"- 特征图尺寸: {height}×{width} (假设正方形)")
            print(f"- 原始图像尺寸: {height*4}×{width*4} (经过2次2×2池化)")
    
    return conv_layers, fc_layers

# 3. 创建匹配的模型类
class DynamicCNN(nn.Cell):
    """根据参数动态创建模型"""
    def __init__(self, param_dict):
        super(DynamicCNN, self).__init__()
        
        # 提取参数信息
        self.layers = nn.CellDict()
        
        # 按顺序构建层
        layer_counter = 1
        
        # 查找所有卷积层
        conv_params = {}
        for name, param in param_dict.items():
            if 'weight' in name and len(param.shape) == 4:
                # 这是卷积层权重
                layer_name = name.replace('.weight', '')
                conv_params[layer_name] = {
                    'weight': param,
                    'bias': param_dict.get(name.replace('weight', 'bias'), None)
                }
        
        # 构建卷积层
        for layer_name, params in sorted(conv_params.items()):
            weight = params['weight']
            bias = params['bias']
            
            in_channels = weight.shape[1]
            out_channels = weight.shape[0]
            kernel_size = weight.shape[2]  # 假设方形卷积核
            
            conv = nn.Conv2d(in_channels, out_channels, 
                            kernel_size=kernel_size,
                            pad_mode='pad',
                            padding=kernel_size//2)
            
            # 加载预训练权重
            conv.weight.set_data(weight)
            if bias is not None:
                conv.bias.set_data(bias)
            
            self.layers[layer_name] = conv
            self.layers[f'relu{layer_counter}'] = nn.ReLU()
            self.layers[f'pool{layer_counter}'] = nn.MaxPool2d(kernel_size=2, stride=2)
            layer_counter += 1
        
        # 添加展平层
        self.layers['flatten'] = nn.Flatten()
        
        # 构建全连接层
        fc_params = {}
        for name, param in param_dict.items():
            if ('fc' in name or 'dense' in name) and 'weight' in name and len(param.shape) == 2:
                layer_name = name.replace('.weight', '')
                fc_params[layer_name] = {
                    'weight': param,
                    'bias': param_dict.get(name.replace('weight', 'bias'), None)
                }
        
        # 构建全连接层
        for i, (layer_name, params) in enumerate(sorted(fc_params.items())):
            weight = params['weight']
            bias = params['bias']
            
            in_features = weight.shape[1]
            out_features = weight.shape[0]
            
            fc = nn.Dense(in_features, out_features)
            
            # 加载预训练权重
            fc.weight.set_data(weight)
            if bias is not None:
                fc.bias.set_data(bias)
            
            self.layers[layer_name] = fc
            
            # 如果不是最后一层，添加ReLU
            if i < len(fc_params) - 1:
                self.layers[f'fc_relu{i+1}'] = nn.ReLU()
    
    def construct(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

# 4. 主函数
def main():
    model_path = r"E:\CodeProject\mindspore-tools-mcp\current_run\simplecnn_best.ckpt"
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return
    
    print("="*60)
    print("模型结构分析")
    print("="*60)
    
    # 1. 检查模型权重
    inspect_model_weights(model_path)
    
    # 2. 加载参数
    param_dict = ms.load_checkpoint(model_path)
    
    # 3. 推断模型结构
    conv_layers, fc_layers = create_matching_model(param_dict)
    
    print("\n" + "="*60)
    print("尝试创建匹配的模型")
    print("="*60)
    
    # 4. 根据参数创建模型
    try:
        # 先尝试使用动态创建
        print("方法1: 动态创建模型...")
        model = DynamicCNN(param_dict)
        
        # 测试模型
        test_input = ms.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(test_input)
        print(f"模型测试成功！输出形状: {output.shape}")
        
        # 保存模型结构信息
        print("\n模型层结构:")
        for name, layer in model.layers.items():
            print(f"  {name}: {type(layer).__name__}")
        
        return model
        
    except Exception as e:
        print(f"动态创建失败: {e}")
        
        # 方法2: 尝试推断具体结构
        print("\n方法2: 根据参数推断标准结构...")
        
        # 从参数推断模型类型
        if 'fc1.weight' in fc_layers:
            fc1_shape = fc_layers['fc1.weight']
            
            # 根据常见的模型结构推断
            if fc1_shape == (256, 2048):
                print("推断模型结构: 类似 ResNet 的特征提取 + 全连接层")
                print("可能的结构: 卷积层 -> 全局池化 -> fc1(256, 2048) -> fc2")
                
                # 创建匹配的简单模型
                class InferredModel(nn.Cell):
                    def __init__(self):
                        super(InferredModel, self).__init__()
                        # 假设输入是 3x224x224
                        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3)
                        self.bn1 = nn.BatchNorm2d(64)
                        self.relu = nn.ReLU()
                        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
                        
                        # 根据 fc1 输入维度推断
                        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                        self.flatten = nn.Flatten()
                        self.fc1 = nn.Dense(2048, 256)  # 注意：权重需要转置
                        self.fc2 = nn.Dense(256, 10)    # 假设10分类
                    
                    def construct(self, x):
                        x = self.conv1(x)
                        x = self.bn1(x)
                        x = self.relu(x)
                        x = self.pool(x)
                        x = self.global_pool(x)
                        x = self.flatten(x)
                        x = self.fc1(x)
                        x = self.fc2(x)
                        return x
                
                model = InferredModel()
                print("创建推断模型成功！")
                return model
        
        return None

if __name__ == "__main__":
    from mindspore import nn
    main()