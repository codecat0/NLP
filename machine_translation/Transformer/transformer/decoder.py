# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : decoder.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from multi_head_attention import MultiHeadAttention, _convert_attention_mask


class TransformerDecoderLayer(nn.Layer):
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
        :param d_model: 输入输出的特征维度
        :param nhead: 多头注意力机制的头数
        :param dim_feedforward: 前向神经网络中隐藏层大小
        :param dropout: 在多头注意力和前向神经网络前处理和后处理数据时drop概率
        :param activation: 前向神经网络后激活函数类型
        :param attn_dropout: 多头注意力机制中dropout层的设置
        :param act_dropout: 激活函数后的dropout层
        :param normalize_before: LayerNorm放在多头注意力和前向神经网络前还是后面
        """
        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout
        )

        self.linear1 = nn.Linear(
            in_features=d_model, out_features=dim_feedforward
        )
        self.dropout = nn.Dropout(
            p=act_dropout, mode='upscale_in_train'
        )
        self.linear2 = nn.Linear(
            in_features=dim_feedforward, out_features=d_model
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(
            p=dropout
        )
        self.dropout2 = nn.Dropout(
            p=dropout
        )
        self.dropout3 = nn.Dropout(
            p=dropout
        )
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        :param tgt: Transformer decoder layer输入：(batch_size, target_length, d_model)
        :param memory: Transformer encoder layer输出：(batch_size, source_length, d_model)
        :param tgt_mask: (bacth_size, n_head, target_length, target_length)
        :param memory_mask: (bacth_size, n_head, target_length, source_length)
        :return:
        """
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)
        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Layer):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.LayerList(
            [
                decoder_layer for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        output = tgt
        for i, mod in enumerate(self.layers):
            output = mod(output,
                         memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output


if __name__ == '__main__':
    batch_size = 4
    tgt_len = 4
    d_model = 128
    src_len = 6
    n_head = 2
    dec_input = paddle.rand((batch_size, tgt_len, d_model))
    enc_output = paddle.rand((batch_size, src_len, d_model))
    self_attn_mask = paddle.rand((batch_size, n_head, tgt_len, tgt_len))
    cross_attn_mask = paddle.rand((batch_size, n_head, tgt_len, src_len))
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=512
    )
    decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=8
    )
    output = decoder(
        dec_input,
        enc_output,
        self_attn_mask,
        cross_attn_mask
    )
    print(output.shape)
