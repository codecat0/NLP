# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : position_embed.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
import paddle
import paddle.nn as nn


def position_encoding_init(n_postion, d_pos_vec, dtype='float32'):
    """"
    :param n_postion: 句子序列的最大位置
    :param d_pos_vec: embedding向量大小
    :param dtype:
    :return:
    """
    channels = d_pos_vec
    position = np.arange(n_postion)
    num_timescales = channels // 2
    log_timesacles_increment = (np.log(float(1e4) / float(1)) / (num_timescales - 1))

    inv_timescales = np.exp(
        np.arange(num_timescales) * -log_timesacles_increment
    )
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], mode='constant')
    position_enc = signal
    return position_enc.astype(dtype)


class PositionEmbedding(nn.Layer):
    def __init__(self, emb_dim, max_length):
        """
        :param emb_dim:  embedding向量大小
        :param max_length: 句子序列的最大长度
        """
        super(PositionEmbedding, self).__init__()
        self.emb_dim = emb_dim

        self.pos_encoder = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=emb_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    position_encoding_init(max_length, emb_dim)
                )
            )
        )

    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        pos_emb.stop_gradient = True
        return pos_emb