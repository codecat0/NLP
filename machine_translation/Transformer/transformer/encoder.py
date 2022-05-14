# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : encoder.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .multi_head_attention import MultiHeadAttention, _convert_attention_mask


class TransformerEncoderLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation='relu',
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        """
        :param d_model: 输入和输出的特征维度
        :param nhead: 多头注意力机制的头数
        :param dim_feedforward: 前向神经网络的隐藏层大小
        :param dropout: 在多头注意力和前向神经网络前处理和后处理数据时drop概率
        :param activation: 前向神经网络后激活函数类型
        :param attn_dropout: 多头注意力机制中dropout层的设置
        :param act_dropout: dropout层后的激活函数
        :param normalize_before: LayerNorm放在多头注意力和前向神经网络前还是后面
        """
        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout
        )

        self.linear1 = nn.Linear(
            d_model, dim_feedforward
        )
        self.dropout = nn.Dropout(
            p=act_dropout
        )
        self.linear2 = nn.Linear(
            dim_feedforward, d_model
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(
            p=dropout
        )
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None):
        """
        :param src: (batch_size, sequence_length, d_model)
        :param src_mask: (batch_size, n_head, sequece_length, sequence_length)
        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        redidual = src
        if self.normalize_before:
            src = self.norm1(src)
        # 多头注意力
        src = self.self_attn(
            src, src, src, src_mask
        )

        src = redidual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        redidual = src
        if self.normalize_before:
            src = self.norm2(src)
        # 前向神经网络
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = redidual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)

        return src



class TransformerEncoder(nn.Layer):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.LayerList(
            [
                encoder_layer for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        output = src
        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


if __name__ == '__main__':
    batch_size = 2
    sequence_length = 4
    d_model = 128
    n_head = 2
    enc_input = paddle.rand((batch_size, sequence_length, d_model))
    attn_mask = paddle.rand((batch_size, n_head, sequence_length, sequence_length))
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=412
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=8
    )
    enc_output = encoder(enc_input, attn_mask)
    print(enc_output.shape)

