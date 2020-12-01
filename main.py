import os
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BERTEncoder, ProtoNet
from data import FewRelDataset, get_loader


def main(args):
    def train():
        # Init
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=300, num_training_steps=train_iter)

        # Training
        start_iter = 0
        best_acc = 0
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        model.train()

        with tqdm(range(start_iter, start_iter + train_iter)) as bar:
            for it in bar:
                support, query, label = next(train_data_loader)
                for k in support:
                    support[k] = support[k].to(device)
                for k in query:
                    query[k] = query[k].to(device)
                label = label.to(device)

                logits, pred = model(support, query, trainN,
                                     K, Q * trainN + na_rate * Q)
                loss = model.loss(logits, label) / float(grad_iter)
                right = model.accuracy(pred, label)
                loss.backward()

                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                iter_loss += loss.item()
                iter_right += right.item()
                iter_sample += 1
                bar.set_postfix_str('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(
                    it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample))

                if (it + 1) % eval_every == 0:
                    acc = eval(val_data_loader)
                    model.train()
                    if acc > best_acc:
                        print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, ckpt)
                        best_acc = acc
                    iter_loss = 0.
                    iter_right = 0.
                    iter_sample = 0.

    def eval(loader):
        model.eval()

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad(), trange(val_iter) as bar:
            for it in bar:
                support, query, label = next(loader)
                for k in support:
                    support[k] = support[k].to(device)
                for k in query:
                    query[k] = query[k].to(device)
                label = label.to(device)
                _, pred = model(support, query, N, K, Q * N + Q * na_rate)

                right = model.accuracy(pred, label)
                iter_right += right.item()
                iter_sample += 1

                bar.set_postfix_str('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(
                    it + 1, 100 * iter_right / iter_sample))
        return iter_right / iter_sample

    # Load configs
    trainN = args.trainN
    dataset = args.dataset
    N, K, Q = args.N, args.K, args.Q
    train_iter, val_iter, eval_every = args.train_iter, args.val_iter, args.eval_every
    batch_size = args.batch_size
    max_length = args.max_length
    grad_iter = args.grad_iter
    na_rate = args.na_rate
    pretrain_ckpt = 'bert-base-uncased'
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/proto-bert-{}-{}_{}.pth'.format(args.dataset, N, K)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    sentence_encoder = BERTEncoder(pretrain_ckpt, max_length)
    model = ProtoNet(sentence_encoder, dot=args.dot)
    model.to(device)
    lr = 2e-5 if args.lr == -1 else args.lr

    # Load data
    train_data_loader = get_loader('train_{}'.format(dataset), sentence_encoder,
                                   N=trainN, K=K, Q=Q, na_rate=args.na_rate, batch_size=batch_size)
    val_data_loader = get_loader('val_{}'.format(dataset), sentence_encoder,
                                 N=N, K=K, Q=Q, na_rate=args.na_rate, batch_size=batch_size)
    test_data_loader = get_loader('val_{}'.format(dataset), sentence_encoder,
                                  N=N, K=K, Q=Q, na_rate=args.na_rate, batch_size=batch_size)
    if args.load_model:
        model.load_state_dict(torch.load(
            args.load_model, map_location=device)['state_dict'])

    # Do training
    if args.train:
        print("Start training...")
        train()
        print("Finish training")

    # Do testing
    if args.test:
        print("Start evaluating...")
        acc = eval(test_data_loader)
        print("Result: %.2f" % (acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='wiki', help='name of dataset')
    parser.add_argument('--trainN', default=10, type=int, help='N in train')
    parser.add_argument('--N', default=5, type=int, help='N way')
    parser.add_argument('--K', default=1, type=int, help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--train_iter', default=30000,
                        type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000,
                        type=int, help='num of iters in testing')
    parser.add_argument('--eval_every', default=2000, type=int,
                        help='evaluate after training how many iters')
    parser.add_argument('--max_length', default=128,
                        type=int, help='max length')
    parser.add_argument('--lr', default=-1, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5,
                        type=float, help='weight decay')
    parser.add_argument('--dropout', default=0.0,
                        type=float, help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
                        help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--hidden_size', default=230,
                        type=int, help='hidden size')
    parser.add_argument('--load_model', default=None,
                        help='where to load trained model')
    parser.add_argument('--train', action='store_true',
                        help='whether do training')
    parser.add_argument('--test', action='store_true',
                        help='whether do testing')
    parser.add_argument('--dot', action='store_true',
                        help='use dot instead of L2 distance for proto')
    main(parser.parse_args())
