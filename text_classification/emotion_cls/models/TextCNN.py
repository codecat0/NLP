# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : TextCNN.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CNNEncoder(nn.Layer):
    """
    Convolutional Neural Networks for Sentence Classification
    """
    def __init__(self,
                 emb_dim,
                 num_filter,
                 ngram_filter_sizes=(2, 3, 4, 5),
                 conv_layer_activatation=nn.Tanh(),
                 output_dim=None,
                 **kwargs):
        """
        :param emb_dim: 输入序列每个向量的维度
        :param num_filter: 每个卷积层输出维度
        :param ngram_filter_sizes: 卷积层的数量及卷积核的大小
        :param conv_layer_activatation: 卷积层后的激活函数
        :param output_dim: 输出维度大小
        :param kwargs:
        """
        super(CNNEncoder, self).__init__()
        self._emb_dim = emb_dim
        self._num_filter = num_filter
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activatation
        self._output_dim = output_dim

        self.convs = nn.LayerList([
            nn.Conv2D(
                in_channels=1,
                out_channels=self._num_filter,
                kernel_size=(i, self._emb_dim),
                **kwargs
            ) for i in self._ngram_filter_sizes
        ])

        maxpool_output_dim = self._num_filter * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(
                in_features=maxpool_output_dim,
                out_features=self._output_dim
            )
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    @property
    def get_input_dim(self):
        return self._emb_dim

    @property
    def get_output_dim(self):
        return self._output_dim

    def forward(self, inputs, mask=None):
        """
        :param inputs: (batch_size, num_tokens, emb_dim) 输入序列的特征
        :param mask: (batch_size, num_tokens) 标记输入序列每个token是否为填充，如果是为False，否则为True
        :return: (batch_size, output_dim)
        """
        if mask is not None:
            inputs = inputs * mask

        # (batch_size, num_tokens, emb_dim) -> (batch_size, 1, num_tokens, emb_dim)
        inputs = inputs.unsqueeze(1)

        convs_out = [
            self._activation(conv(inputs)).squeeze(3) for conv in self.convs
        ]
        maxpool_out = [
            F.adaptive_avg_pool1d(
                t, output_size=1
            ).squeeze(2) for t in convs_out
        ]
        result = paddle.concat(maxpool_out, axis=1)

        if self.projection_layer is not None:
            result = self.projection_layer(result)
        return result


class TextCNNModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emd_dim=128,
                 padding_idx=0,
                 num_filter=128,
                 ngram_filter_sizes=(1, 2, 3),
                 fc_hidden_size=96):
        super(TextCNNModel, self).__init__()
        self.embedder = nn.Embedding(
            vocab_size, emd_dim, padding_idx=padding_idx
        )
        self.encoder = CNNEncoder(
            emb_dim=emd_dim,
            num_filter=num_filter,
            ngram_filter_sizes=ngram_filter_sizes
        )
        self.fc = nn.Linear(self.encoder.get_output_dim, fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len=None):
        # (batch_size, num_tokens) -> (batch_size, num_tokens, emb_dim)
        embedded_text = self.embedder(text)

        # (batch_size, num_tokens, emb_dim) -> (batch_size, encoder.get_output_dim)
        encoder_out = self.encoder(embedded_text)
        encoder_out = paddle.tanh(encoder_out)

        fc_out = self.fc(encoder_out)
        logits = self.output_layer(fc_out)

        probs = F.softmax(logits, axis=1)
        return probs


if __name__ == '__main__':
    model = TextCNNModel(vocab_size=100, num_classes=2)
    text = paddle.randint(low=1, high=10, shape=(2, 10), dtype='int32')
    probs = model(text)
    print(probs)