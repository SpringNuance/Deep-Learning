from io import open
import unicodedata
import re
import pickle
import os
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive


# Prepare the data
SOS_token = 0  # Start-of-sentence token
EOS_token = 1  # End-of-sentence token
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang:
    """A class that encodes words with one-hot vectors."""
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class TranslationDataset(Dataset):
    download_url_prefix = 'https://users.aalto.fi/~alexilin/dle'
    zip_filename = 'translation_data.zip'

    source_lang_file = 'fra_lang.pkl'
    target_lang_file = 'eng_lang.pkl'
    pairs_file = 'eng-fra_pairs_train.pkl'
    train_pairs_file = 'eng-fra_pairs_test.pkl'
    test_pairs_file = 'eng-fra_pairs.pkl'

    def __init__(self, root, train=None):
        self.root = root
        self._folder = folder = os.path.join(root, 'translation_data')
        self._fetch_data(root)
        
        self.input_lang = pickle.load(open(os.path.join(folder, self.source_lang_file), "rb"))
        self.output_lang = pickle.load(open(os.path.join(folder, self.target_lang_file), "rb"))
        if train is None:
            self.pairs = pickle.load(open(os.path.join(folder, self.pairs_file), "rb"))
        elif train:
            self.pairs = pickle.load(open(os.path.join(folder, self.train_pairs_file), "rb"))
        else:
            self.pairs = pickle.load(open(os.path.join(folder, self.test_pairs_file), "rb"))

    def _preprocess(self, lang1='eng', lang2='fra'):
        reverse = True
        print('Preprpocess the data')
        input_lang, output_lang, pairs = readLangs(path, lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        print(pairs[0])
        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        # Split into training and test set
        train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=1, shuffle=True)
        print('Training pairs:', len(train_pairs))
        print('Test pairs:', len(test_pairs))
        pickle.dump(input_lang, open(source_lang_file, "wb"))
        pickle.dump(output_lang, open(target_lang_file, "wb"))
        pickle.dump(pairs, open(pairs_file, "wb"))
        pickle.dump(train_pairs, open(train_pairs_file, "wb"))
        pickle.dump(test_pairs, open(test_pairs_file, "wb"))

        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = train_pairs if train else test_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_seq = tensorFromSentence(self.input_lang, pair[0])
        output_seq = tensorFromSentence(self.output_lang, pair[1])
        return (input_seq, output_seq)

    def _check_integrity(self):
        files = [self.source_lang_file, self.target_lang_file, self.pairs_file, self.train_pairs_file, self.test_pairs_file]
        for file in files:
            if not os.path.isfile(os.path.join(self._folder, file)):
                return False
        return True

    def _fetch_data(self, data_dir):
        if self._check_integrity():
            return
        
        url = self.download_url_prefix + '/' + self.zip_filename
        download_and_extract_archive(url, data_dir, filename=self.zip_filename, remove_finished=True)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(path, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(os.path.join(path, '%s-%s.txt' % (lang1, lang2)), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

