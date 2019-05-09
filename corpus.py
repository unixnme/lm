from torch.utils.data import Dataset, DataLoader

class Corpus(Dataset):
    TOKEN_BOS = '<s>'
    TOKEN_EOS = '</s>'
    TOKEN_UNK = '<unk>'

    def __init__(self, corpus_file:str):
        with open(corpus_file, 'r') as f:
            self.data = [line.split() for line in f]

        self.vocab = {self.TOKEN_BOS, self.TOKEN_EOS, self.TOKEN_UNK}
        for tokens in self.data:
            self.vocab = self.vocab.union(set(tokens))
        self.vocab = sorted(list(self.vocab))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item:int):
        return [self.TOKEN_BOS] + self.data[item],\
               self.data[item] + [self.TOKEN_EOS]


class CorpusLoader(DataLoader):
    def __init__(self, corpus:Corpus, batch_size, shuffle:bool, num_workers:int):
        super(CorpusLoader, self).__init__(corpus, batch_size, shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    def collate_fn(self, batch:list):
        return list(zip(*batch))


if __name__ == '__main__':
    corpus_file = 'corpus.txt'
    corpus = Corpus(corpus_file)
    loader = CorpusLoader(corpus, 3, False, 2)
    print(corpus.vocab)
    for x,y in loader:
        for i,o in zip(x,y):
            print(i,o)