# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : TextRNN.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LSTMEncoder(nn.Layer):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction='forward',
                 dropout=0.0,
                 pooling_type=None,
                 **kwargs):
        super(LSTMEncoder, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._direction = direction
        self._pooling_type = pooling_type

        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            direction=direction,
            dropout=dropout,
            **kwargs
        )

    @property
    def get_input_dim(self):
        return self._input_size

    @property
    def get_output_dim(self):
        if self._direction == 'bidirect':
            return self._hidden_size * 2
        else:
            return self._hidden_size

    def forward(self, inputs, sequence_length):
        """
        :param inputs: (batch_size, num_tokens, input_size) 输入句子的特征
        :param sequence_length: 输入句子的有效长度
        :return:
            output: (batch_size, hidden_size * num_directions) 对于每一个LSTM层的最后一个时间步的隐藏状态
                    num_directions = 2 if direction is 'bidirect' else 1
        """
        encoded_text, (last_hidden, last_cell) = self.lstm_layer(inputs, sequence_length=sequence_length)
        if not self._pooling_type:
            if self._direction != 'bidirect':
                output = last_hidden[-1, :, :]
            else:
                output = paddle.concat(
                    (last_hidden[-2, :, :], last_hidden[-1, :, :]), axis=1
                )
        else:
            if self._pooling_type == 'sum':
                output = paddle.sum(encoded_text, axis=1)
            elif self._pooling_type == 'max':
                output = paddle.max(encoded_text, axis=1)
            elif self._pooling_type == 'mean':
                output = paddle.mean(encoded_text, axis=1)
            else:
                raise ValueError(
                    'Unexpected pooling type %s. Pooling type must be one of sum, max and mean' % self._pooling_type
                )
        return output


class LSTMModel(nn.Layer):
    """
    Recurrent Convolutional Neural Networks for Text Classification
    """
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward',
                 lstm_layers=1,
                 dropout_rate=0.0,
                 pooling_type=None,
                 fc_hidden_size=96):
        super(LSTMModel, self).__init__()

        # 首先将输入word id 查表后映射为word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx
        )

        # 将word embedding经过LSTM变换到文本语义表征空间中
        self.lstm_encoder = LSTMEncoder(
            input_size=emb_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            pooling_type=pooling_type
        )

        self.fc = nn.Linear(
            in_features=self.lstm_encoder.get_output_dim,
            out_features=fc_hidden_size
        )

        self.output_layer = nn.Linear(
            in_features=fc_hidden_size,
            out_features=num_classes
        )

    def forward(self, text, seq_len):
        """
        :param text: (batch_size, num_tokens) 每个句子文本的id信息
        :param seq_len: (batch_size, 1) 每个句子有效长度
        :return:
            probs: (batch_size, num_classes) : 每个句子属于不同类别的概率
        """
        # (batch_size, num_tokens) -> (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)

        # (batch_size, num_tokens, embedding_dim) -> (batch_size, hidden_size * num_directions)
        # num_directions = 2 if direction is 'bidirect' else 1
        text_repr = self.lstm_encoder(embedded_text, seq_len)

        # (batch_size, hidden_size * num_directions) -> (batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))

        # (batch_size, fc_hidden_size) -> (batch_size, num_classes)
        logits = self.output_layer(fc_out)

        probs = F.softmax(logits)
        return probs


if __name__ == '__main__':
    model = LSTMModel(vocab_size=100, num_classes=2)
    text = paddle.randint(low=1, high=10, shape=(2, 10), dtype='int64')
    seq_len = paddle.to_tensor([10, 10])
    out = model(text, seq_len)
    print(out)