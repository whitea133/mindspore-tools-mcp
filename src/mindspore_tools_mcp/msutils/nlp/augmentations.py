"""
msutils.nlp.augmentations - 文本数据增强

提供多种文本增强方法
"""

import random
import re
from typing import List, Optional, Callable
import numpy as np


class TextAugmenter:
    """
    文本增强器基类
    """
    
    def augment(self, text: str) -> str:
        """
        增强单个文本
        
        Args:
            text: 输入文本
        
        Returns:
            增强后的文本
        """
        raise NotImplementedError
    
    def augment_batch(self, texts: List[str]) -> List[str]:
        """
        批量增强文本
        
        Args:
            texts: 文本列表
        
        Returns:
            增强后的文本列表
        """
        return [self.augment(text) for text in texts]


class RandomInsertion(TextAugmenter):
    """
    随机插入增强
    
    在文本中随机位置插入同义词
    
    Reference:
        Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for 
        Boosting Performance on Text Classification Tasks. 
        EMNLP-IJCNLP 2019. https://arxiv.org/abs/1901.11196
    
    Example:
        >>> aug = RandomInsertion(n=2)
        >>> new_text = aug.augment("hello world")
    """
    
    def __init__(
        self,
        n: int = 1,
        synonyms: Optional[dict] = None,
        random_state: Optional[int] = None
    ):
        """
        初始化增强器
        
        Args:
            n: 插入数量
            synonyms: 同义词词典
            random_state: 随机种子
        """
        self.n = n
        self.synonyms = synonyms or self._default_synonyms()
        self.random = random.Random(random_state)
    
    def _default_synonyms(self) -> dict:
        """默认同义词词典"""
        return {
            'good': ['great', 'excellent', 'nice'],
            'bad': ['terrible', 'awful', 'poor'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'mini'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['gradual', 'steady'],
            'old': ['ancient', 'historic'],
            'new': ['fresh', 'modern'],
            'happy': ['glad', 'pleased', 'joyful'],
            'sad': ['unhappy', 'sorrowful']
        }
    
    def augment(self, text: str) -> str:
        """随机插入"""
        words = text.split()
        
        for _ in range(self.n):
            if not words:
                break
            
            idx = self.random.randint(0, len(words) - 1)
            word = words[idx].lower()
            
            if word in self.synonyms:
                synonym = self.random.choice(self.synonyms[word])
                words.insert(idx, synonym)
        
        return ' '.join(words)


class RandomDeletion(TextAugmenter):
    """
    随机删除增强
    
    随机删除文本中的单词
    
    Reference:
        Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for 
        Boosting Performance on Text Classification Tasks. 
        EMNLP-IJCNLP 2019. https://arxiv.org/abs/1901.11196
    
    Example:
        >>> aug = RandomDeletion(p=0.1)
        >>> new_text = aug.augment("hello world")
    """
    
    def __init__(
        self,
        p: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        初始化增强器
        
        Args:
            p: 删除概率
            random_state: 随机种子
        """
        self.p = p
        self.random = random.Random(random_state)
    
    def augment(self, text: str) -> str:
        """随机删除"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = [w for w in words if self.random.random() > self.p]
        
        # 确保不为空
        if not new_words:
            return self.random.choice(words)
        
        return ' '.join(new_words)


class RandomSwap(TextAugmenter):
    """
    随机交换增强
    
    随机交换文本中的两个单词
    
    Reference:
        Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for 
        Boosting Performance on Text Classification Tasks. 
        EMNLP-IJCNLP 2019. https://arxiv.org/abs/1901.11196
    
    Example:
        >>> aug = RandomSwap(n=2)
        >>> new_text = aug.augment("hello world")
    """
    
    def __init__(
        self,
        n: int = 1,
        random_state: Optional[int] = None
    ):
        """
        初始化增强器
        
        Args:
            n: 交换次数
            random_state: 随机种子
        """
        self.n = n
        self.random = random.Random(random_state)
    
    def augment(self, text: str) -> str:
        """随机交换"""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        for _ in range(self.n):
            idx1 = self.random.randint(0, len(words) - 1)
            idx2 = self.random.randint(0, len(words) - 1)
            
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)


class SynonymReplacement(TextAugmenter):
    """
    同义词替换增强
    
    随机替换单词为其同义词
    
    Reference:
        Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for 
        Boosting Performance on Text Classification Tasks. 
        EMNLP-IJCNLP 2019. https://arxiv.org/abs/1901.11196
    
    Note:
        本实现使用简化的硬编码同义词词典。生产环境建议使用：
        - WordNet (英文): https://wordnet.princeton.edu/
        - 同义词词林 (中文): https://github.com/yaleimeng/Final_word_Similarity
        - 预训练词向量找近义词
    
    Example:
        >>> aug = SynonymReplacement(n=2)
        >>> new_text = aug.augment("good job")
    """
    
    def __init__(
        self,
        n: int = 1,
        synonyms: Optional[dict] = None,
        random_state: Optional[int] = None
    ):
        """
        初始化增强器
        
        Args:
            n: 替换数量
            synonyms: 同义词词典
            random_state: 随机种子
        """
        self.n = n
        self.synonyms = synonyms or self._default_synonyms()
        self.random = random.Random(random_state)
    
    def _default_synonyms(self) -> dict:
        """默认同义词词典"""
        return {
            'good': ['great', 'excellent', 'nice', 'fine'],
            'bad': ['terrible', 'awful', 'poor', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'mini'],
            'fast': ['quick', 'rapid', 'swift'],
            'slow': ['gradual', 'steady'],
            'old': ['ancient', 'historic'],
            'new': ['fresh', 'modern'],
            'happy': ['glad', 'pleased', 'joyful'],
            'sad': ['unhappy', 'sorrowful'],
            'beautiful': ['pretty', 'lovely', 'attractive'],
            'ugly': ['unattractive', 'hideous'],
            'smart': ['intelligent', 'clever', 'bright'],
            'stupid': ['foolish', 'silly', 'dumb']
        }
    
    def augment(self, text: str) -> str:
        """同义词替换"""
        words = text.split()
        
        # 找出可替换的单词
        replaceable = [
            (i, w.lower()) for i, w in enumerate(words)
            if w.lower() in self.synonyms
        ]
        
        if not replaceable:
            return text
        
        # 随机选择要替换的单词
        n_replace = min(self.n, len(replaceable))
        to_replace = self.random.sample(replaceable, n_replace)
        
        for idx, word in to_replace:
            synonyms = self.synonyms[word]
            words[idx] = self.random.choice(synonyms)
        
        return ' '.join(words)


class BackTranslation(TextAugmenter):
    """
    回译增强
    
    通过中间语言进行回译来增强文本
    
    Reference:
        Sennrich, R., Haddow, B., & Birch, A. (2016). Improving Neural Machine 
        Translation Models with Monolingual Data. ACL 2016. 
        https://arxiv.org/abs/1511.06709
    
    Note:
        当前为简化实现，实际使用需要接入翻译 API：
        - Google Cloud Translation API
        - Azure Translator
        - 或本地翻译模型 (如 MarianMT)
    
    Example:
        >>> aug = BackTranslation()
        >>> new_text = aug.augment("hello world")
    """
    
    def __init__(self):
        """初始化回译增强器"""
        # 简化实现：不做实际翻译
        pass
    
    def augment(self, text: str) -> str:
        """
        回译（简化版）
        
        这里只是返回原文作为占位符
        实际使用时需要接入翻译 API
        """
        # TODO: 接入 Google Translate 或其他翻译 API
        return text


class RandomCharacter(TextAugmenter):
    """
    随机字符增强
    
    随机替换、删除或交换字符
    
    Reference:
        字符级噪声是常见的文本增强方法，参考：
        Belinkov, Y., & Bisk, Y. (2018). Synthetic and Natural Noise Both Break 
        Neural Machine Translation. ICLR 2018. 
        https://arxiv.org/abs/1711.02173
    
    Example:
        >>> aug = RandomCharacter()
        >>> new_text = aug.augment("hello")
    """
    
    def __init__(
        self,
        p: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        初始化增强器
        
        Args:
            p: 操作概率
            random_state: 随机种子
        """
        self.p = p
        self.random = random.Random(random_state)
    
    def augment(self, text: str) -> str:
        """随机字符操作"""
        if len(text) < 2:
            return text
        
        chars = list(text)
        
        for i in range(len(chars)):
            if self.random.random() < self.p:
                action = self.random.choice(['swap', 'delete', 'upper'])
                
                if action == 'swap' and i < len(chars) - 1:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                elif action == 'delete':
                    chars[i] = ''
                elif action == 'upper':
                    chars[i] = chars[i].upper()
        
        return ''.join(chars)


class TextAugmentPipeline:
    """
    文本增强组合器
    
    组合多种增强方法
    
    Example:
        >>> aug = TextAugmentPipeline([
        ...     SynonymReplacement(n=1),
        ...     RandomSwap(n=1),
        ...     RandomDeletion(p=0.1)
        ... ])
        >>> new_text = aug.augment("hello world")
    """
    
    def __init__(
        self,
        augmenters: List[TextAugmenter],
        random_state: Optional[int] = None
    ):
        """
        初始化组合器
        
        Args:
            augmenters: 增强器列表
            random_state: 随机种子
        """
        self.augmenters = augmenters
        self.random = random.Random(random_state)
    
    def augment(self, text: str) -> str:
        """依次应用所有增强器"""
        result = text
        
        # 随机选择部分增强器
        selected = self.random.sample(
            self.augmenters,
            self.random.randint(1, len(self.augmenters))
        )
        
        for aug in selected:
            result = aug.augment(result)
        
        return result
    
    def augment_batch(self, texts: List[str]) -> List[str]:
        """批量增强"""
        return [self.augment(text) for text in texts]


def create_augmenter(
    augmenter_type: str = 'synonym',
    **kwargs
) -> TextAugmenter:
    """
    文本增强器工厂函数
    
    Args:
        augmenter_type: 增强器类型
        **kwargs: 增强器参数
    
    Returns:
        文本增强器实例
    
    Example:
        >>> aug = create_augmenter('synonym', n=2)
        >>> new_text = aug.augment("good job")
    """
    augmenters = {
        'insert': RandomInsertion,
        'delete': RandomDeletion,
        'swap': RandomSwap,
        'synonym': SynonymReplacement,
        'char': RandomCharacter,
        'compose': Text.Compose
    }
    
    if augmenter_type not in augmenters:
        raise ValueError(
            f"Unknown augmenter: {augmenter_type}. "
            f"Available: {list(augmenters.keys())}"
        )
    
    return augmenters[augmenter_type](**kwargs)


# 导出所有增强器
__all__ = [
    'TextAugmenter',
    'RandomInsertion',
    'RandomDeletion',
    'RandomSwap',
    'SynonymReplacement',
    'BackTranslation',
    'RandomCharacter',
    'create_augmenter'
]
