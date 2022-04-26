# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : predict.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import argparse

import paddle
from paddlenlp.data import JiebaTokenizer, Stack, Tuple, Pad, Vocab
from paddlenlp.datasets import load_dataset

from models.TextRNN import LSTMModel
from models.TextCNN import TextCNNModel
from models.TextBiLSTM_Att import BiLSTMAttentionModel, SelfAttention, SelfInteractiveAttention
from utils.dataset_utils import preprocess_prediction_data


def predict(model, data, label_map, batch_size=1, pad_token_id=0):
    # 将数据分割为mini-batch
    batches = [
        data[idx: idx + batch_size] for idx in range(0, len(data), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        # input_ids
        Pad(axis=0, pad_val=pad_token_id),
        # seq len
        Stack(dtype='int64')
    ): fn(samples)

    results = []
    model.eval()
    for batch in batches:
        texts, seqs_len = batchify_fn(batch)
        texts = paddle.to_tensor(texts)
        seqs_len = paddle.to_tensor(seqs_len)
        probs = model(texts, seqs_len)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


def main(args):
    paddle.set_device(args.device)

    # 导入词典
    vocab = Vocab.from_json(args.vocab_path)
    label_map = {
        0: 'negative',
        1: 'positive'
    }

    network = args.network.lower()
    vocab_size = len(vocab)
    num_classes = len(label_map)
    pad_token_id = vocab.to_indices('[PAD]')
    if network == 'textrnn':
        model = LSTMModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            direction='bidirect',
            padding_idx=pad_token_id
        )
    elif network == 'textcnn':
        model = TextCNNModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            padding_idx=pad_token_id
        )
    elif network == 'textbilstm_att':
        model = BiLSTMAttentionModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            padding_idx=pad_token_id,
            attention_layer=SelfAttention()
        )
    elif network == 'textbilstm_inter_att':
        model = BiLSTMAttentionModel(
            vocab_size=vocab_size,
            num_classes=num_classes,
            padding_idx=pad_token_id,
            attention_layer=SelfInteractiveAttention()
        )
    else:
        raise ValueError(
            "Unknown network: %s, it must be one of textrnn, textcnn, textbilstm_att and textbilstm_inter_att." % network
        )

    # 加载模型参数
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)

    # 加载测试数据
    text = []
    test_dv = load_dataset('chnsenticorp', splits=['test'])
    for item in test_dv:
        text.append(item['text'])

    tokenizer = JiebaTokenizer(vocab)
    data = preprocess_prediction_data(text, tokenizer)

    results = predict(
        model=model,
        data=data,
        label_map=label_map,
        batch_size=args.batch_size,
        pad_token_id=vocab.token_to_idx.get('[PAD]', 0)
    )
    # 打印前5条数据
    for idx, text in enumerate(text[:5]):
        print('Data: {} \t Label: {}'.format(text, results[idx]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="cpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--batch_size", type=int, default=4, help="Total examples' number of a batch for training.")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json", help="The file path to vocabulary.")
    parser.add_argument('--network',
                        choices=['textrnn', 'textcnn', 'textbilstm_att', 'textbilstm_inter_att'],
                        default="textbilstm_att", help="Select which network to train, defaults to textbilstm_att.")
    parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams',
                        help="The path of model parameter to be loaded.")
    args = parser.parse_args()
    main(args)

