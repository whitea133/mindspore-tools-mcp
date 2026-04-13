"""
msutils.nlp.tokenizers - NLP 分词工具

提供多种分词方法和预处理工具
"""

import re
from typing import List, Dict, Optional, Tuple
import numpy as np


class BasicTokenizer:
    """
    基础分词器
    
    支持中英文分词、去标点、转小写等基础操作
    
    Example:
        >>> tokenizer = BasicTokenizer()
        >>> tokens = tokenizer.tokenize("Hello, world!")
        >>> print(tokens)  # ['hello', 'world']
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        language: str = 'english'
    ):
        """
        初始化分词器
        
        Args:
            lowercase: 是否转小写
            remove_punctuation: 是否移除标点
            remove_stopwords: 是否移除停用词
            language: 语言类型 ('english', 'chinese')
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.language = language
        
        # 英文停用词
        self.english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # 中文停用词
        self.chinese_stopwords = {
            '的', '一', '是', '在', '不', '了', '有', '和', '人', '这',
            '中', '大', '为', '上', '个', '国', '我', '以', '要', '他',
            '时', '来', '用', '们', '生', '到', '作', '地', '于', '出',
            '就', '分', '对', '成', '会', '可', '主', '发', '年', '动'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词结果列表
        """
        # 转小写
        if self.lowercase:
            text = text.lower()
        
        # 移除标点
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分词
        if self.language == 'chinese':
            tokens = list(text)  # 简单的字符级分词
        else:
            tokens = text.split()
        
        # 移除停用词
        if self.remove_stopwords:
            if self.language == 'chinese':
                tokens = [t for t in tokens if t not in self.chinese_stopwords]
            else:
                tokens = [t for t in tokens if t not in self.english_stopwords]
        
        # 移除空字符串
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        批量分词
        
        Args:
            texts: 文本列表
        
        Returns:
            分词结果列表
        """
        return [self.tokenize(text) for text in texts]


class WordPieceTokenizer:
    """
    WordPiece 分词器
    
    基于词表的子词分词方法，类似 BERT
    
    Example:
        >>> vocab = ['[UNK]', 'hello', 'world', '##ing', '##ed']
        >>> tokenizer = WordPieceTokenizer(vocab)
        >>> tokens = tokenizer.tokenize("hello")
        >>> print(tokens)  # ['hello']
    """
    
    def __init__(
        self,
        vocab: List[str],
        max_chars: int = 100,
        unknown_token: str = '[UNK]'
    ):
        """
        初始化 WordPiece 分词器
        
        Args:
            vocab: 词表列表
            max_chars: 最大字符数
            unknown_token: 未知词标记
        """
        self.vocab = set(vocab)
        self.vocab_list = vocab
        self.max_chars = max_chars
        self.unknown_token = unknown_token
        
        # 构建词表索引
        self.vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    def tokenize(self, text: str) -> List[str]:
        """
        WordPiece 分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词结果列表
        """
        tokens = []
        
        for word in text.split():
            word = word.lower()
            
            if len(word) > self.max_chars:
                tokens.append(self.unknown_token)
                continue
            
            # 贪心匹配
            start = 0
            while start < len(word):
                end = len(word)
                found = False
                
                while start < end:
                    substr = word[start:end]
                    
                    # 非首字符添加 ##
                    if start > 0:
                        substr = '##' + substr
                    
                    if substr in self.vocab:
                        tokens.append(substr)
                        found = True
                        break
                    
                    end -= 1
                
                if not found:
                    tokens.append(self.unknown_token)
                    start += 1
                else:
                    start = end
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        将 token 转换为 ID
        
        Args:
            tokens: token 列表
        
        Returns:
            ID 列表
        """
        return [
            self.vocab_dict.get(token, self.vocab_dict.get(self.unknown_token, 0))
            for token in tokens
        ]


class CharacterTokenizer:
    """
    字符级分词器
    
    将文本分解为字符序列
    
    Example:
        >>> tokenizer = CharacterTokenizer()
        >>> tokens = tokenizer.tokenize("hello")
        >>> print(tokens)  # ['h', 'e', 'l', 'l', 'o']
    """
    
    def __init__(self, lowercase: bool = False):
        """
        初始化字符分词器
        
        Args:
            lowercase: 是否转小写
        """
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """
        字符级分词
        
        Args:
            text: 输入文本
        
        Returns:
            字符列表
        """
        if self.lowercase:
            text = text.lower()
        
        return list(text)


class SentenceTokenizer:
    """
    句子分词器
    
    将文本分解为句子
    
    Example:
        >>> tokenizer = SentenceTokenizer()
        >>> sentences = tokenizer.tokenize("Hello world. How are you?")
        >>> print(sentences)  # ['Hello world.', 'How are you?']
    """
    
    def __init__(self):
        """初始化句子分词器"""
        self.sentence_endings = r'[.!?]+'
    
    def tokenize(self, text: str) -> List[str]:
        """
        句子分词
        
        Args:
            text: 输入文本
        
        Returns:
            句子列表
        """
        # 按句号、感叹号、问号分割
        sentences = re.split(self.sentence_endings, text)
        
        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


class NGramTokenizer:
    """
    N-gram 分词器
    
    生成 N-gram 序列
    
    Example:
        >>> tokenizer = NGramTokenizer(n=2)
        >>> ngrams = tokenizer.tokenize("hello world")
        >>> print(ngrams)  # [('hello', 'world')]
    """
    
    def __init__(self, n: int = 2):
        """
        初始化 N-gram 分词器
        
        Args:
            n: N-gram 的 N 值
        """
        self.n = n
    
    def tokenize(self, text: str) -> List[Tuple]:
        """
        生成 N-gram
        
        Args:
            text: 输入文本
        
        Returns:
            N-gram 列表
        """
        tokens = text.split()
        ngrams = []
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.append(ngram)
        
        return ngrams
    
    def get_ngram_counts(self, text: str) -> Dict[Tuple, int]:
        """
        获取 N-gram 计数
        
        Args:
            text: 输入文本
        
        Returns:
            N-gram 计数字典
        """
        ngrams = self.tokenize(text)
        counts = {}
        
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1
        
        return counts


class BPETokenizer:
    """
    字节对编码（Byte Pair Encoding）分词器
    
    Example:
        >>> tokenizer = BPETokenizer()
        >>> tokens = tokenizer.tokenize("hello world")
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        初始化 BPE 分词器
        
        Args:
            vocab_size: 词表大小
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def tokenize(self, text: str) -> List[str]:
        """
        BPE 分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词结果列表
        """
        # 简化实现：字符级分词
        tokens = list(text.lower())
        
        # 应用合并规则
        for merge in self.merges[:100]:  # 限制合并次数
            tokens = self._apply_merge(tokens, merge)
        
        return tokens
    
    def _apply_merge(self, tokens: List[str], merge: Tuple) -> List[str]:
        """
        应用一个合并规则
        
        Args:
            tokens: token 列表
            merge: 合并规则 (token1, token2)
        
        Returns:
            合并后的 token 列表
        """
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens


def create_tokenizer(
    tokenizer_type: str = 'basic',
    **kwargs
):
    """
    分词器工厂函数
    
    Args:
        tokenizer_type: 分词器类型
        **kwargs: 分词器参数
    
    Returns:
        分词器实例
    
    Example:
        >>> tokenizer = create_tokenizer('basic', lowercase=True)
        >>> tokens = tokenizer.tokenize("Hello World")
    """
    tokenizers = {
        'basic': BasicTokenizer,
        'wordpiece': WordPieceTokenizer,
        'character': CharacterTokenizer,
        'sentence': SentenceTokenizer,
        'ngram': NGramTokenizer,
        'bpe': BPETokenizer
    }
    
    if tokenizer_type not in tokenizers:
        raise ValueError(
            f"Unknown tokenizer: {tokenizer_type}. "
            f"Available: {list(tokenizers.keys())}"
        )
    
    return tokenizers[tokenizer_type](**kwargs)


# 导出所有分词器
__all__ = [
    'BasicTokenizer',
    'WordPieceTokenizer',
    'CharacterTokenizer',
    'SentenceTokenizer',
    'NGramTokenizer',
    'BPETokenizer',
    'create_tokenizer'
]
