# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : TextBiLSTM_att.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

INF = 1e6


class SelfAttention(nn.Layer):
    """
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    """
    def __init__(self, hidden_size=196):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weight = self.create_parameter(
            shape=(1, hidden_size, 1), dtype='float32'
        )

    def forward(self, input, mask=None):
        """
        :param input: (batch_size, seq_len, hidden_size) 输入序列的特征
        :param mask: (batch_size, seq_len) 每一个元素标记输入的单词id是否为pad token
        """
        forward_input, backward_input = paddle.chunk(input, chunks=2, axis=2)

        # elementwise-sum 来连接 forward_input 和 backward_input
        # (batch_size, seq_len, hidden_size
        h = paddle.add_n([forward_input, backward_input])
        # (batch_size, hidden_size, 1)
        att_weight = self.att_weight.tile(
            repeat_times=(paddle.shape(h)[0], 1, 1)
        )
        # (batch_size, seq_len, 1)
        att_score = paddle.bmm(paddle.tanh(h), att_weight)
        if mask is not None:
            mask = paddle.cast(mask, dtype='float32')
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = paddle.full(
                shape=mask.shape, dtype='float32', fill_value=-INF
            )
            att_score = paddle.multiply(att_score, mask) + paddle.multiply(inf_tensor, (1 - mask))

        # (batch_size, seq_len, 1)
        att_weight = F.softmax(att_score, axis=1)
        reps = paddle.bmm(h.transpose(perm=(0, 2, 1)), att_weight).squeeze(axis=-1)
        reps = paddle.tanh(reps)
        return reps, att_weight


class SelfInteractiveAttention(nn.Layer):
    """
    Hierarchical Attention Networks for Document Classiﬁcation
    """
    def __init__(self, hidden_size=196):
        super(SelfInteractiveAttention, self).__init__()
        self.input_weight = self.create_parameter(
            shape=(1, hidden_size, hidden_size), dtype='float32'
        )
        self.bias = self.create_parameter(
            shape=(1, 1, hidden_size), dtype='float32'
        )
        self.att_context_vector = self.create_parameter(
            shape=(1, hidden_size, 1), dtype='float32'
        )

    def forward(self, input, mask=None):
        """
        :param input: (batch_size, seq_len, hidden_size) 输入序列的特征
        :param mask: (batch_size, seq_len) 每一个元素标记输入的单词id是否为pad token
        """
        weight = self.input_weight.tile(
            repeat_times=(paddle.shape(input)[0], 1, 1)
        )
        bias = self.bias.tile(
            repeat_times=(paddle.shape(input)[0], 1, 1)
        )
        # (batch_size, seq_len, hidden_size)
        word_squish = paddle.bmm(input, weight) + bias

        att_context_vector = self.att_context_vector.tile(
            repeat_times=(paddle.shape(input)[0], 1, 1)
        )
        # (batch_size, seq_len, 1)
        att_score = paddle.bmm(word_squish, att_context_vector)
        if mask is not None:
            mask = paddle.cast(mask, dtype='float32')
            mask = mask.unsqueeze(axis=-1)
            inf_tensor = paddle.full(
                shape=mask.shape, dtype='float32', fill_value=-INF
            )
            att_score = paddle.multiply(att_score, mask) + paddle.multiply(inf_tensor, (1 - mask))

        # (batch_size, seq_len, 1)
        att_weight = F.softmax(att_score, axis=1)
        reps = paddle.bmm(input.transpose(perm=(0, 2, 1)), att_weight).squeeze(axis=-1)
        return reps, att_weight


class BiLSTMAttentionModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 padding_idx=0,
                 emb_dim=128,
                 lstm_hidden_size=196,
                 lstm_layer=1,
                 dropout_rate=0.0,
                 attention_layer=SelfAttention(),
                 fc_hidden_size=96
                 ):
        super(BiLSTMAttentionModel, self).__init__()
        self.padding_idx = padding_idx

        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx
        )

        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layer,
            dropout=dropout_rate,
            direction='bidirect'
        )

        if isinstance(attention_layer, SelfAttention):
            self.attention = SelfAttention(hidden_size=lstm_hidden_size)
            self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        elif isinstance(attention_layer, SelfInteractiveAttention):
            self.attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
            self.fc = nn.Linear(2 * lstm_hidden_size, fc_hidden_size)

        else:
            raise ValueError(
                'Unkonwn attention type %s. ' % attention_layer.__class__.__name__
            )

        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        mask = text != self.padding_idx
        embedded_text = self.embedder(text)
        encoded_text, (last_hidden, last_cell) = self.bilstm(embedded_text, sequence_length=seq_len)
        hidden, att_weights = self.attention(encoded_text, mask)
        fc_out = paddle.tanh(self.fc(hidden))
        logits = self.output_layer(fc_out)
        probs = F.softmax(logits, axis=1)
        return probs


if __name__ == '__main__':
    model = BiLSTMAttentionModel(vocab_size=100, num_classes=2, attention_layer=SelfInteractiveAttention())
    text = paddle.randint(low=1, high=10, shape=(2, 10), dtype='int64')
    seq_len = paddle.to_tensor([10, 10])
    out = model(text, seq_len)
    print(out)