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
    for x, target in (loader):
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


def generate(model:Network, device:str='cpu', limit:int=100):
    x = Corpus.TOKEN_BOS
    hid = None
    sentence = []
    for _ in range(limit):
        idx = model.extractor.word2idx[x]
        vec = model.extractor.embedding(torch.LongTensor([idx]).to(device))
        out, hid = model.rnn(vec.view(1,1,-1), hid)
        out = model.linear(out.view(1,-1))
        idx = out.argmax(-1)
        word = model.extractor.vocab[idx]
        if word == Corpus.TOKEN_EOS:
            break
        sentence.append(word)
    return sentence


def main():
    parser = argparse.ArgumentParser('LM main')
    parser.add_argument('--corpus', type=str, default='tiny_corpus.txt', help='corpus to train')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=.99)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    corpus = Corpus(args.corpus)
    loader = CorpusLoader(corpus, args.batch_size, True, args.num_workers)
    extractor = EmbeddingExtractor(corpus.vocab, args.emb_dim, args.device)
    network = Network(extractor, args.num_layers).to(args.device)
    optimizer = torch.optim.SGD(network.parameters(), args.lr, args.momentum)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    for epoch in range(args.epochs):
        pred = generate(network, device=args.device)
        print(' '.join(pred))

        loss = single_epoch(network, loader, optimizer, loss_fn)
        print(loss)
        scheduler.step(loss)


if __name__ == '__main__':
    main()
