import torch
from torch.nn.utils.rnn import PackedSequence
from extractor import Extractor
from utils import unpack

class Network(torch.nn.Module):
    def __init__(self, extractor:Extractor, num_layers:int, drop:float):
        super(Network, self).__init__()
        self.extractor = extractor
        self.dropout = torch.nn.Dropout(drop)
        self.rnn = torch.nn.LSTM(self.extractor.emb_dim,
                                self.extractor.emb_dim,
                                num_layers,
                                dropout=drop
                                )
        self.linear = torch.nn.Linear(self.extractor.emb_dim,
                                      len(self.extractor.vocab),
                                      False)

    def forward(self, sentences:list, to_str:bool=False) -> PackedSequence:
        packed = self.extractor(sentences)
        packed = PackedSequence(self.dropout(packed.data), packed.batch_sizes)
        packed, _ = self.rnn(packed)
        data = self.linear(packed.data)
        if to_str:
            max_idx = data.argmax(-1)
            word_pack = [self.extractor.vocab[i] for i in max_idx.cpu()]
            return unpack(word_pack, packed.batch_sizes)
        else:
            return PackedSequence(data, packed.batch_sizes)


if __name__ == '__main__':
    from corpus import Corpus, CorpusLoader
    from extractor import EmbeddingExtractor

    corpus_file = 'tiny_corpus.txt'
    corpus = Corpus(corpus_file)
    loader = CorpusLoader(corpus, 100, False, 8)
    extractor = EmbeddingExtractor(corpus.vocab, 64, 'cpu')
    network = Network(extractor)
    print(corpus.vocab)
    for x,y in loader:
        pred = network(x, to_str=True)
        print(pred, y)
