import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from collections import Iterable

class Extractor(torch.nn.Module):
    def __init__(self, vocab:list, emb_dim:int, device:str):
        super(Extractor, self).__init__()
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.device = device
        self.word2idx = {word:idx for idx,word in enumerate(vocab)}

    def forward(self, sentences:list) -> PackedSequence:
        '''
        :param sentences: batch of seq of words
        '''
        raise NotImplemented


class EmbeddingExtractor(Extractor):
    def __init__(self, vocab:list, emb_dim:int, device:str):
        super(EmbeddingExtractor, self).__init__(vocab, emb_dim, device)
        self.embedding = torch.nn.Embedding(len(vocab), emb_dim)

    def forward(self, sentences:list) -> PackedSequence:
        pack = self.get_packed_sequence(sentences)
        vectors = self.embedding(pack.data)
        return PackedSequence(vectors, pack.batch_sizes)

    def get_packed_sequence(self, sentences:list):
        seqs = [torch.LongTensor([self.word2idx[word] for word in s]) for s in sentences]
        return pack_sequence(seqs).to(self.device)


if __name__ == '__main__':
    from corpus import Corpus, CorpusLoader

    corpus_file = 'tiny_corpus.txt'
    corpus = Corpus(corpus_file)
    loader = CorpusLoader(corpus, 100, False, 8)
    extractor = EmbeddingExtractor(corpus.vocab, 64, 'cpu')
    print(corpus.vocab)
    for x,y in loader:
        pack = extractor(x)
        print(len(x), pack.data.shape)
