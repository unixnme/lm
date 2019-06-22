from corpus import Corpus, CorpusLoader
from extractor import EmbeddingExtractor
from network import Network
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
import kenlm


def single_epoch(model:Network, loader:DataLoader, optimizer:Optimizer, loss_fn:torch.nn.Module, norm:float, train:bool=True):
    for param in model.parameters():
        param.requires_grad = train

    total_loss = 0
    model.train(train)
    for x, target in (loader):
        if train:
            model.zero_grad()
        pred = model(x)
        target = model.extractor.get_packed_sequence(target)
        loss = loss_fn(pred.data, target.data)
        if train:
            loss.backward()
            #clip_grad_norm_(model.parameters(), norm)
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def generate(model:Network, device:str='cpu', limit:int=100):
    x = Corpus.TOKEN_BOS
    hid = None
    sentence = []
    model.eval()
    with torch.no_grad():
        for _ in range(limit):
            idx = model.extractor.word2idx[x]
            vec = model.extractor.embedding(torch.LongTensor([idx]).to(device))
            out, hid = model.rnn(vec.view(1,1,-1), hid)
            out = model.linear(out.view(1,-1))
            prob = torch.softmax(out, -1).view(-1)
            # word2p = [(model.extractor.vocab[idx], p.item()) for idx,p in enumerate(prob)]
            # word2p.sort(key=lambda x: x[1], reverse=True)
            idx = prob.multinomial(1)
            word = model.extractor.vocab[idx]
            if word == Corpus.TOKEN_EOS:
                break
            sentence.append(word)
    return sentence


def main():
    parser = argparse.ArgumentParser('LM main')
    parser.add_argument('--corpus', type=str, default='tiny_corpus.txt', help='corpus to train')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=.99)
    parser.add_argument('--clip_norm', type=float, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=str, default='model.pt')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--arpa', type=str, default='tiny_corpus.arpa')
    args = parser.parse_args()

    corpus = Corpus(args.corpus)
    loader = CorpusLoader(corpus, args.batch_size, True, args.num_workers)
    if args.load is None:
        extractor = EmbeddingExtractor(corpus.vocab, args.emb_dim, args.device)
        network = Network(extractor, args.num_layers, drop=args.drop).to(args.device)
    else:
        network = torch.load(args.load, map_location=args.device)
        network.extractor.device = args.device
        network.rnn.flatten_parameters()

    ken_lm = kenlm.LanguageModel(args.arpa)
    optimizer = torch.optim.RMSprop(network.parameters(), args.lr)
    #optimizer = torch.optim.SGD(network.parameters(), args.lr, args.momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=.5)

    min_loss = float('inf')
    for epoch in range(args.epochs):
        pred = generate(network, device=args.device)
        gen_sentence = ' '.join(pred)
        ppl = ken_lm.perplexity(gen_sentence)
        print('%s\nPPL:\t%f' % (gen_sentence, ppl))

        loss = single_epoch(network, loader, optimizer, loss_fn, args.clip_norm)
        print('epochs %d \t loss %.3f' % (epoch, loss))
        scheduler.step(loss)

        if min_loss > loss:
            min_loss = loss
            print('saving to %s' % args.save)
            torch.save(network, args.save)
        print()


if __name__ == '__main__':
    main()
