# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : transformer.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import numpy as np
import paddle
import paddle.nn as nn

from multi_head_attention import _convert_attention_mask
from encoder import TransformerEncoderLayer, TransformerEncoder
from decoder import TransformerDecoderLayer, TransformerDecoder


class Transformer(nn.Layer):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        """
        :param d_model: 编码器和解码器输入和输出的特征维度
        :param nhead: 多头注意力机制的头数
        :param num_encoder_layers: 编码器的个数
        :param num_decoder_layers: 解码器的个数
        :param dim_feedforward: 前向神经网络隐藏层大小
        :param dropout: 在多头注意力和前向神经网络前处理和后处理数据时drop概率
        :param activation: 前向神经网络后激活函数类型
        :param attn_dropout: 多头注意力机制中dropout层的设置
        :param act_dropout: dropout层后的激活函数
        :param normalize_before: LayerNorm放在多头注意力和前向神经网络前还是后面
        """
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            normalize_before=normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        memory = self.encoder(
            src, src_mask=src_mask
        )

        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return output

    @staticmethod
    def generate_square_subsequent_mask(length):
        return paddle.tensor.triu(
            (paddle.ones(
                shape=(length, length), dtype=paddle.get_default_dtype()
            )) * -np.inf
        )


if __name__ == '__main__':
    batch_size = 2
    src_length = 4
    tgt_length = 6
    d_model = 128
    n_head = 2

    enc_input = paddle.rand((batch_size, src_length, d_model))
    dec_input = paddle.rand((batch_size, tgt_length, d_model))
    enc_self_attn_mask = paddle.rand((batch_size, n_head, src_length, src_length))
    dec_self_attn_mask = paddle.rand((batch_size, n_head, tgt_length, tgt_length))
    dec_cross_attn_mask = paddle.rand((batch_size, n_head, tgt_length, src_length))
    transformer = Transformer(
        d_model=d_model,
        nhead=n_head
    )
    output = transformer(
        enc_input,
        dec_input,
        enc_self_attn_mask,
        dec_self_attn_mask,
        dec_cross_attn_mask
    )
    print(output.shape)