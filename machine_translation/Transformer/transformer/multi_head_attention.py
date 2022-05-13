# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : multi_head_attention.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import tensor
from paddle.fluid.data_feeder import convert_dtype


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
        return attn_mask


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'

        self.q_proj = nn.Linear(
            embed_dim, embed_dim
        )
        self.k_proj = nn.Linear(
            self.kdim, embed_dim
        )
        self.v_proj = nn.Linear(
            self.vdim, embed_dim
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim
        )

    def _prepare_qkv(self, query, key, value):
        """
        :param query: (batch_size, sequence_length, embed_dim)
        :param key: (batch_size, sequence_length, kdim)
        :param value: (batch_size, sequence_length, vdim)
        :return: (bacth_size, num_heads, sequence_length, embed_dim // num_heads)
        """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=(0, 0, self.num_heads, self.head_dim))
        q = tensor.transpose(x=q, perm=(0, 2, 1, 3))

        k = self.k_proj(key)
        k = tensor.reshape(x=k, shape=(0, 0, self.num_heads, self.head_dim))
        k = tensor.transpose(x=k, perm=(0, 2, 1, 3))

        v = self.v_proj(value)
        v = tensor.reshape(x=v, shape=(0, 0, self.num_heads, self.head_dim))
        v = tensor.transpose(x=v, perm=(0, 2, 1, 3))

        return q, k, v

    def forward(self, query, key=None, value=None, attn_mask=None):
        """
        :param query: (batch_size, sequence_length, embed_dim)
        :param key: (batch_size, sequence_length, kdim)
        :param value: (batch_size, sequecne_length, vdim)
        :param attn_mask: (bacth_size, num_head, sequence_length, sequence_length)
        :return:
        """
        key = query if key is None else key
        value = query if value is None else value
        # 获取q，k，v
        q, k, v = self._prepare_qkv(query, key, value)

        # 计算 q*k/sqrt(d) 注意力权重
        product = tensor.matmul(
            x=q, y=k, transpose_y=True
        ) / self.head_dim ** -0.5

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)

        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode='upscale_in_train'
            )

        out = tensor.matmul(weights, v)

        # 合并 heads
        out = tensor.transpose(x=out, perm=(0, 2, 1, 3))
        out = tensor.reshape(x=out, shape=(0, 0, out.shape[2] * out.shape[3]))

        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        return out if len(outs) == 1 else tuple(outs)


