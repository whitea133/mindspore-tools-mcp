"""
msutils.analysis.visualization - 可视化工具

提供训练曲线、模型结构等可视化功能
"""

import numpy as np


def plot_training_curves(history, save_path: str = None):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史记录
        save_path: 保存路径
    
    Example:
        >>> from msutils.analysis import plot_training_curves
        >>> plot_training_curves(history, 'training_curves.png')
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Curves')
            axes[0].legend()
            axes[0].grid(True)
        
        # 准确率曲线
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Acc')
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy Curves')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_confusion_matrix(cm, class_names, save_path: str = None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")


def show_images(images, labels=None, predictions=None, num_images: int = 16, save_path: str = None):
    """
    显示图像网格
    
    Args:
        images: 图像数组
        labels: 真实标签
        predictions: 预测标签
        num_images: 显示数量
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        num = min(num_images, len(images))
        cols = 4
        rows = (num + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i in range(num):
            ax = axes[i]
            img = images[i]
            
            # 处理通道顺序
            if img.shape[0] in [1, 3]:  # CHW -> HWC
                img = img.transpose(1, 2, 0)
            
            if img.shape[-1] == 1:  # 灰度图
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(img)
            
            title = ''
            if labels is not None:
                title += f'True: {labels[i]}'
            if predictions is not None:
                title += f'\nPred: {predictions[i]}'
            
            ax.set_title(title)
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required. Install with: pip install matplotlib")


__all__ = [
    'plot_training_curves',
    'plot_confusion_matrix',
    'show_images',
    'plot_learning_rate',
    'plot_gradient_flow',
    'plot_class_distribution',
    'plot_roc_curve',
    'plot_precision_recall'
]


def plot_learning_rate(lr_history, save_path: str = None):
    """
    绘制学习率曲线
    
    Args:
        lr_history: 学习率历史
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        plt.plot(lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required")


def plot_gradient_flow(named_parameters, save_path: str = None):
    """
    绘制梯度流
    
    查看各层梯度分布
    
    Args:
        named_parameters: 模型参数
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        avg_grads = []
        max_grads = []
        layers = []
        
        for name, param in named_parameters:
            if param.grad is not None:
                layers.append(name)
                avg_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(avg_grads)), max_grads, alpha=0.5, lw=1, color='c', label='max')
        plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.5, lw=1, color='b', label='mean')
        plt.hlines(0, 0, len(avg_grads) + 1, lw=2, color='k')
        plt.xticks(range(len(avg_grads)), layers, rotation='vertical')
        plt.xlim(left=0, right=len(avg_grads))
        plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)
        plt.xlabel('Layers')
        plt.ylabel('Gradient')
        plt.title('Gradient Flow')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required")


def plot_class_distribution(labels, class_names=None, save_path: str = None):
    """
    绘制类别分布
    
    Args:
        labels: 标签数组
        class_names: 类别名称
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(unique, counts, color='steelblue')
        
        if class_names:
            plt.xticks(unique, [class_names[i] for i in unique], rotation=45, ha='right')
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required")


def plot_roc_curve(fpr, tpr, auc_score, save_path: str = None):
    """
    绘制 ROC 曲线
    
    Args:
        fpr: 假阳性率
        tpr: 真阳性率
        auc_score: AUC 分数
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required")


def plot_precision_recall(precision, recall, save_path: str = None):
    """
    绘制 Precision-Recall 曲线
    
    Args:
        precision: 精确率
        recall: 召回率
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("matplotlib is required")
