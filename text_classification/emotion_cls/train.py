# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
from functools import partial
import argparse
import random

import numpy as np
import paddle
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset
from visualdl import LogWriter

from models.TextRNN import LSTMModel
from models.TextCNN import TextCNNModel
from models.TextBiLSTM_Att import BiLSTMAttentionModel, SelfAttention, SelfInteractiveAttention
from utils.dataset_utils import convert_example, build_vocab
from utils.train_val_utils import train_one_epoch, eval_one_epoch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None) -> paddle.io.DataLoader:
    """
    Args:
        dataset: 数据集
        trans_fn: 将数据样本转为input_ids, seq len, label
        mode: 是否为训练模式
        batch_size: mini-batch的大小
        batchify_fn: 将mini-batch数据合并为一个列表
    Returns:
        dataloader: 用于生成batch的dataloader
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=batchify_fn
    )
    return dataloader


def main(args):
    paddle.set_device(args.device)
    set_seed()
    log_writer = LogWriter(logdir=args.logdir)

    # Load dataset
    train_ds, dev_ds = load_dataset('chnsenticorp', splits=['train', 'dev'])
    texts = []
    for data in train_ds:
        texts.append(data['text'])
    for data in dev_ds:
        texts.append(data['text'])

    # 停用词
    stopwords = {'的', '吗', '吧', '呀', '呜', '呢', '呗', ',', '，', '。', '？', '.', ';', ':', '!', ' '}

    # Build vocab
    word2idx = build_vocab(
        texts=texts,
        stopwords=stopwords,
        min_freq=5
    )
    vocab = Vocab.from_dict(
        token_to_idx=word2idx,
        unk_token='[UNK]',
        pad_token='[PAD]'
    )
    # save vocab
    vocab.to_json(args.vocab_path)

    # build dataloader
    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        # input_ids
        Pad(pad_val=vocab.token_to_idx.get('[PAD]', 0)),
        # valid_length (seq len)
        Stack(dtype='int64'),
        # label
        Stack(dtype='int64')
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(
        dataset=train_ds,
        trans_fn=trans_fn,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn
    )
    dev_loader = create_dataloader(
        dataset=dev_ds,
        trans_fn=trans_fn,
        mode='validation',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn
    )

    # Build Network
    network = args.network.lower()
    vocab_size = len(vocab)
    num_classes = len(train_ds.label_list)
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

    # Define optimizer and loss
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr)
    loss_function = paddle.nn.CrossEntropyLoss()

    # model.to(device=args.device)

    for epoch in range(args.epochs):
        print('Epoch {} / {}'.format(epoch+1, args.epochs))
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            dataloader=train_loader,
            device=args.device
        )

        val_loss, val_acc = eval_one_epoch(
            model=model,
            loss_function=loss_function,
            dataloader=dev_loader,
            device=args.device
        )

        # visualdl log
        tags = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        log_writer.add_scalar(tags[0], train_loss, epoch)
        log_writer.add_scalar(tags[1], train_acc, epoch)
        log_writer.add_scalar(tags[2], val_loss, epoch)
        log_writer.add_scalar(tags[3], val_acc, epoch)


    paddle.save(model.state_dict(), path=args.weight_path + network + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Number of epoches for training.")
    parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="cpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
    parser.add_argument("--weight_path", type=str, default='./checkpoints/', help="Directory to save model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json", help="The file path to save vocabulary.")
    parser.add_argument('--network',
                        choices=['textrnn', 'textcnn', 'textbilstm_att', 'textbilstm_inter_att'],
                        default="textrnn", help="Select which network to train, defaults to textrnn.")
    parser.add_argument("--logdir", type=str, default='./log/train', help="Directory to save train log")
    args = parser.parse_args()
    print(args)
    main(args)