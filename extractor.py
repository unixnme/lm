import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from collections import Iterable

class Extractor(torch.nn.Module):
    def __init__(self, device:str):
        super(Extractor, self).__init__()
        self.device = device

    def forward(self, words:list) -> PackedSequence:
        '''
        :param words: batch of seq of words
        '''
        raise NotImplemented


class EmbeddingExtractor(Extractor):
    def __init__(self, vocab:list, emb_dim:int, device:str):
        super(EmbeddingExtractor, self).__init__(device)
        self.word2idx = {word:idx for idx,word in enumerate(vocab)}
        self.embedding = torch.nn.Embedding(len(vocab), emb_dim)

    def forward(self, words:list) -> PackedSequence:
        pack = self.get_packed_sequence(words)
        vectors = self.embedding(pack.data)
        return PackedSequence(vectors, pack.batch_sizes)

    def get_packed_sequence(self, words:list):
        seqs = [torch.LongTensor([self.word2idx[word] for word in s]) for s in words]
        return pack_sequence(seqs).to(self.device)


if __name__ == '__main__':
    from corpus import Corpus
    from torch.utils.data import DataLoader

    corpus_file = 'corpus.txt'
    corpus = Corpus(corpus_file)
    dataloader = DataLoader(corpus, batch_size=2)
    extractor = EmbeddingExtractor(corpus.vocab, 64, 'cpu')
    for x in dataloader:
        print(extractor(x))