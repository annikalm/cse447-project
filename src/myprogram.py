#!/usr/bin/env python
import argparse
import json
import os
import random
import string
from pathlib import Path

import torch

from data import Vocab, build_vocab, load_lines, make_training_batches
from inference import predict_batch, predict_file
from model import CharGRU
from train_utils import save_checkpoint, train_model


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    train_p = subparsers.add_parser('train', help='train model')
    train_p.add_argument('--work_dir', default='work', help='where to save artifacts')
    train_p.add_argument('--train_data', default='example/input.txt', help='training text file')
    train_p.add_argument('--vocab_size', type=int, default=2000, help='max characters in vocab')
    train_p.add_argument('--seq_len', type=int, default=64, help='sequence length')
    train_p.add_argument('--batch_size', type=int, default=128, help='batch size')
    train_p.add_argument('--epochs', type=int, default=20, help='training epochs')
    train_p.add_argument('--emb_dim', type=int, default=256, help='embedding size')
    train_p.add_argument('--hidden_dim', type=int, default=512, help='hidden size')
    train_p.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    test_p = subparsers.add_parser('test', help='run inference')
    test_p.add_argument('--work_dir', default='work', help='where artifacts live')
    test_p.add_argument('--test_data', default='example/input.txt', help='test input file')
    test_p.add_argument('--test_output', default='pred.txt', help='where to write predictions')
    test_p.add_argument('--batch_size', type=int, default=4096, help='batch size for inference')

    return parser.parse_args()


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


class MyModel:
    """
    char level GRU model, returns random chars if training is missing
    """

    def __init__(self, model=None, vocab: Vocab | None = None, config: dict | None = None, device=None):
        self.model = model
        self.vocab = vocab
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def load_training_data(cls, fname):
        return load_lines(fname)

    @classmethod
    def load_test_data(cls, fname):
        return load_lines(fname)

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write(f'{p}\n')

    def run_train(self, lines, work_dir, args):
        vocab = build_vocab(lines, max_size=args.vocab_size)
        train_batches = make_training_batches(lines, vocab, seq_len=args.seq_len, batch_size=args.batch_size)

        model = CharGRU(len(vocab), emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(self.device)

        config = {
            'vocab_size': len(vocab),
            'emb_dim': args.emb_dim,
            'hidden_dim': args.hidden_dim,
            'seq_len': args.seq_len,
            'batch_size': args.batch_size,
        }

        train_model(model, train_batches, device=self.device, epochs=args.epochs, lr=args.lr)
        self.model = model
        self.vocab = vocab
        self.config = config
        self.save(work_dir)

    def run_pred(self, data_lines, batch_size: int = 64):
        # if nothing trained yet then js return random things
        if self.model is None or self.vocab is None:
            chars = string.ascii_letters or 'abc'
            return [''.join(random.choice(chars) for _ in range(3)) for _ in data_lines]

        self.model.eval()
        # disable gradient engine for maximum speed and memory efficiency
        with torch.inference_mode():
            return predict_batch(self.model, self.vocab, data_lines, device=self.device, k=3)

    def save(self, work_dir):
        if self.model is not None and self.vocab is not None:
            save_checkpoint(work_dir, self.model, self.vocab, self.config)
        else:
            # make sure right dir
            Path(work_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, work_dir):
        artifacts_path = Path(work_dir)
        model_path = artifacts_path / 'model.pt'
        vocab_path = artifacts_path / 'vocab.json'
        config_path = artifacts_path / 'config.json'

        if not (model_path.exists() and vocab_path.exists() and config_path.exists()):
            return cls(model=None, vocab=None, config=None)

        with open(config_path) as f:
            config = json.load(f)

        vocab = Vocab.load(vocab_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CharGRU(len(vocab), emb_dim=config.get('emb_dim', 128), hidden_dim=config.get('hidden_dim', 256))
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return cls(model=model, vocab=vocab, config=config, device=device)


def main():
    args = parse_args()
    set_seed(0)
    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir, exist_ok=True)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data(args.train_data)
        print('Training')
        model.run_train(train_data, args.work_dir, args)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data, batch_size=args.batch_size)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        MyModel.write_pred(pred, args.test_output)


if __name__ == '__main__':
    main()
