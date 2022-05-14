# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : word_embed.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn


class WordEmbedding(nn.Layer):
    def __init__(self, vocab_size, emb_dim, bos_id=0):
        """
        :param vocab_size: 词表大小
        :param emb_dim: embeding向量大小
        :param bos_id: padding id
        """
        super(WordEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.word_embedding  = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=bos_id,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0., emb_dim**-0.5)
            )
        )

    def forward(self, word):
        word_emb = self.emb_dim**0.5 * self.word_embedding(word)
        return word_emb