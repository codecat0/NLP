# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : dataset_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from collections import defaultdict
import numpy as np
from paddlenlp import Taskflow

# 采用Taskflow作为切词工具
word_segmenter = Taskflow('word_segmentation')


def build_vocab(texts,
                stopwords=[],
                num_words=None,
                min_freq=10,
                unk_token='[UNK]',
                pad_token='[PAD]'):
    """
    Args:
        texts: 原始语料库数据
        stopwords: 停用词
        num_words: 词典中最大的单词数
        min_freq: 要保留词的最小词频
        unk_token: 对于未知词的特殊token表示
        pad_token: 对于pad操作的特殊token表示
    Returns:
        word_index: 原始语料库的字典
    """
    word_counts = defaultdict(int)
    for text in texts:
        if not text:
            continue
        for word in word_segmenter(text):
            if word in stopwords:
                continue
            word_counts[word] += 1

    wcounts = []
    for word, count in word_counts.items():
        if count < min_freq:
            continue
        wcounts.append((word, count))

    wcounts.sort(key=lambda x: x[1], reverse=True)
    # -2 是为了unk_token 和 pad_token
    if num_words is not None and len(wcounts) > (num_words - 2):
        wcounts = wcounts[:(num_words - 2)]
    sorted_voc = [pad_token, unk_token]
    sorted_voc.extend(wc[0] for wc in wcounts)
    word_index = dict(zip(sorted_voc, list(range(len(sorted_voc)))))
    return word_index


def convert_example(example, tokenizer, is_test=False):
    """
    Args:
        example: 输入数据列表，包含文本和标签
        tokenizer: 使用jieba来分割中文文本
        is_test: 输入数据是否为测试数据
    Returns:
        input_ids: 词id列表
        valid_length：输入文本有效长度
        label: 输入标签
    """
    input_ids = tokenizer.encode(example['text'])
    valid_length = np.array(len(input_ids), dtype='int64')
    input_ids = np.array(input_ids, dtype='int64')

    if not is_test:
        label = np.array(example['label'], dtype='int64')
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def preprocess_prediction_data(data, tokenizer):
    examples = []
    for text in data:
        ids = tokenizer.encode(text)
        examples.append([ids, len(ids)])
    return examples