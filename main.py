from corpus import Corpus, CorpusLoader
from extractor import EmbeddingExtractor
from network import Network
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import numpy as np


def single_epoch(model:Network, loader:DataLoader, optimizer:Optimizer, loss_fn:torch.nn.Module, train:bool=True):
    for param in model.parameters():
        param.requires_grad = train

    total_loss = 0
    for x, target in tqdm(loader):
        if train:
            model.zero_grad()
        pred = model(x)
        target = model.extractor.get_packed_sequence(target)
        loss = loss_fn(pred.data, target.data).mean()
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss


def generate(model:Network, words:list):
    pred = model([words], to_str=True)
    return pred[0]


def main():
    parser = argparse.ArgumentParser('LM main')
    parser.add_argument('--corpus', type=str, default='tiny_corpus.txt', help='corpus to train')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=.99)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    corpus = Corpus(args.corpus)
    loader = CorpusLoader(corpus, args.batch_size, True, args.num_workers)
    extractor = EmbeddingExtractor(corpus.vocab, 64, args.device)
    network = Network(extractor).to(args.device)
    optimizer = torch.optim.SGD(network.parameters(), args.lr, args.momentum)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    for epoch in range(args.epochs):
        idx = np.random.randint(len(corpus))
        words, target = corpus[idx]
        pred = generate(network, words)
        print(' '.join(pred))
        print(' '.join(target))

        loss = single_epoch(network, loader, optimizer, loss_fn)
        print(loss)
        scheduler.step(loss)


if __name__ == '__main__':
    main()
