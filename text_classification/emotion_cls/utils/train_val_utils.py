# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : train_val_utils.py
# Author     ：CodeCat
# version    ：python 3.7
# Software   ：Pycharm
"""
import paddle
from tqdm import tqdm


def train_one_epoch(model, optimizer, loss_function, dataloader, device):
    model.train()
    loss_sum = paddle.zeros(shape=[1])
    acc_sum = paddle.zeros(shape=[1])
    # dataloader = tqdm(dataloader)
    for step, data in enumerate(dataloader):
        texts, seqs_len, labels = data
        labels = labels.reshape((-1, 1))
        predicts = model(texts, seqs_len)

        loss = loss_function(predicts, labels)
        acc = paddle.metric.accuracy(predicts, labels)

        loss_sum += loss.detach()
        acc_sum += acc.detach()

        loss.backward()

        # dataloader.desc = "step {}/{} - loss: {:.4f} - acc: {:.4f}".format(step+1, len(dataloader), loss.item(), acc.item())

        if (step + 1) % 10 == 0:
            print("step {}/{} - loss: {:.4f} - acc: {:.4f}".format(step+1, len(dataloader), loss.item(), acc.item()))

        optimizer.step()

        optimizer.clear_grad()
    return loss_sum / len(dataloader), acc_sum / len(dataloader)


@paddle.no_grad()
def eval_one_epoch(model, dataloader, loss_function, device):
    model.eval()

    loss_sum = paddle.zeros(shape=[1])
    acc_sum = paddle.zeros(shape=[1])
    # dataloader = tqdm(dataloader)
    for step, data in enumerate(dataloader):
        texts, seqs_len, labels = data
        predicts = model(texts, seqs_len)
        labels = labels.reshape((-1, 1))
        loss = loss_function(predicts, labels)
        acc = paddle.metric.accuracy(predicts, labels)

        loss_sum += loss.detach()
        acc_sum += acc.detach()

        # dataloader.desc = "step {}/{} - loss: {:.4f} - acc: {:.4f}".format(step + 1, len(dataloader), loss.item(), acc.item())

        if (step + 1) % 10 == 0:

            print("step {}/{} - loss: {:.4f} - acc: {:.4f}".format(step + 1, len(dataloader), loss.item(), acc.item()))

    return loss_sum / len(dataloader), acc_sum / len(dataloader)




