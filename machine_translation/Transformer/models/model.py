# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from word_embed import WordEmbedding
from position_embed import PositionEmbedding
from ..transformer.transformer import Transformer


class TransformerDecodeCell(nn.Layer):
    def __init__(self,
                 decoder,
                 word_embedding=None,
                 pos_embedding=None,
                 linear=None,
                 dropout=0.1):
        super(TransformerDecodeCell, self).__init__()
        self.decoder = decoder
        self.word_embedding = word_embedding
        self.pos_embedding = pos_embedding
        self.linear = linear
        self.dropout = dropout

    def forward(self, inputs, trg_src_attn_bias, memory):
        """
        :param inputs: 列表or元组 包含target ids和position ids
        :param trg_src_attn_bias: mask操作，防止模型看到未来时刻的词
        :param memory: encoder输出
        :return:
        """

        if self.word_embedding:
            if not isinstance(inputs, (list, tuple)):
                raise ValueError('when Word Embedding is not None, inputs must include target ids and position ids')

            word_emb = self.word_embedding[inputs[0]]
            pos_emb = self.pos_embedding(inputs[1])
            word_emb = word_emb + pos_emb
            inputs = F.dropout(
                x=word_emb,
                p=self.dropout,
                training=False
            ) if self.dropout else word_emb

            cell_outputs = self.decoder(
                inputs, memory, None, trg_src_attn_bias
            )
        else:
            cell_outputs = self.decoder(
                inputs, memory, None, trg_src_attn_bias
            )

        if self.linear:
            cell_outputs = self.linear(cell_outputs)

        return cell_outputs


class TransformerModel(nn.Layer):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 num_encoder_layers,
                 num_decoder_layers,
                 n_head,
                 d_model,
                 d_inner_hid,
                 dropout,
                 attn_dropout=None,
                 act_dropout=None,
                 bos_id=0,
                 eos_id=1):
        """
        :param src_vocab_size: 源词典大小
        :param trg_vocab_size: 目标词典大小
        :param max_length: 输入句子序列最大长度
        :param num_encoder_layers: encoder layer个数
        :param num_decoder_layers: decoder layer个数
        :param n_head: 多头注意力机制头数
        :param d_model: word embedding维数
        :param d_inner_hid: position-wise 前向神经网络隐藏层大小
        :param dropout: Dropout概率
        :param attn_dropout:
        :param act_dropout:
        :param bos_id: 开始token id和padding id
        :param eos_id: 结束token id
        """
        super(TransformerModel, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size =trg_vocab_size
        self.emb_dim = d_model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dropout = dropout

        self.src_word_embedding = WordEmbedding(
            vocab_size=src_vocab_size, emb_dim=d_model, bos_id=bos_id
        )
        self.src_pos_embedding = PositionEmbedding(
            emb_dim=d_model, max_length=max_length
        )

        self.trg_word_embedding = WordEmbedding(
            vocab_size=trg_vocab_size, emb_dim=d_model, bos_id=bos_id
        )
        self.trg_pos_embedding = PositionEmbedding(
            emb_dim=d_model, max_length=max_length
        )

        self.transformer = Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            attn_dropout=attn_dropout,
            act_dropout=act_dropout,
            activation='relu',
            normalize_before=True
        )

        self.linear = nn.Linear(
            in_features=d_model,
            out_features=trg_vocab_size
        )

    def forward(self, src_word, trg_word):
        """
        :param src_word: 源序列ids  (batch_size, source_sequence_len)
        :param trg_word: 目标序列ids (batch_size, target_sequence_len)
        :return: (batch_size, sequence_len, vocab_size)
        """
        src_max_len = paddle.shape(src_word)[-1]
        trg_max_len = paddle.shape(trg_word)[-1]

        # Transformer Encoder部分不考虑padding ids
        src_self_attn_mask = paddle.cast(
            src_word == self.bos_id,
            dtype=paddle.get_default_dtype()
        ).unsqueeze((1, 2)) * -1e4
        src_self_attn_mask.stop_gradient = True

        # Transformer Decoder self attention不考虑未来时刻的词
        trg_self_attn_mask = self.transformer.generate_square_subsequent_mask(
            length=trg_max_len
        )
        trg_self_attn_mask.stop_gradient = True

        trg_src_attn_mask = src_self_attn_mask

        src_pos = paddle.cast(
            src_word != self.bos_id, dtype=src_word.dtype
        ) * paddle.arange(0, src_max_len, dtype=src_word.dtype)

        trg_pos = paddle.cast(
            trg_word != self.bos_id, dtype=trg_word.dtype
        ) * paddle.arange(0, trg_max_len, dtype=trg_word.dtype)

        src_emb = self.src_word_embedding(src_word)
        src_pos_emb = self.src_pos_embedding(src_pos)
        src_emb = src_emb + src_pos_emb
        enc_input = F.dropout(
            src_emb, p=self.dropout, training=self.training
        ) if self.dropout else src_emb

        trg_emb = self.trg_word_embedding(trg_word)
        trg_pos_emb = self.trg_pos_embedding(trg_pos)
        trg_emb = trg_emb + trg_pos_emb
        dec_input = F.dropout(
            trg_emb, p=self.dropout, training=self.training
        ) if self.dropout else trg_emb

        output = self.transformer(
            src=enc_input,
            tgt=dec_input,
            src_mask=src_self_attn_mask,
            tgt_mask=trg_self_attn_mask,
            memory_mask=trg_src_attn_mask
        )

        predict = self.linear(output)

        return predict


if __name__ == '__main__':
    transformer = TransformerModel(
        src_vocab_size=30000,
        trg_vocab_size=30000,
        max_length=257,
        num_encoder_layers=6,
        num_decoder_layers=6,
        n_head=8,
        d_model=512,
        d_inner_hid=2048,
        dropout=0.1,
        bos_id=0,
        eos_id=1
    )

    batch_size = 4
    src_len = 10
    trg_len = 12
    predict = transformer(
        src_word=paddle.randint(low=2, high=30000, shape=(batch_size, src_len)),
        trg_word=paddle.randint(low=2, high=30000, shape=(batch_size, trg_len))
    )
    print(predict.shape)