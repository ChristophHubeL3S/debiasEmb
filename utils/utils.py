import gzip
from collections import *
import random
import numpy as np


def construct_skipgram_training_instances(words, n_words):
    """
    construct the training instances
    :param words: corpus
    :param n_words: learn most common n_words
    :return:
        - data: [word_index]
        - count: [ [word_index, word_count], ]
        - dictionary: {word_str: word_index}
        - reversed_dictionary: {word_index: word_str}
    """
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # UNK index is 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def read_data(data_path):
    print('Reading data file %s...' % data_path)

    if data_path.endswith('gz'):
        fin = gzip.open(data_path, 'rt')
    else:
        fin = open(data_path, 'rt')

    data = fin.read().split()
    print('Finished reading data file. The corpus contains %d sentences' % len(data))
    return data


def noise(vocabs, word_count):
    """
    generate noise distribution
    :param vocabs:
    :param word_count:
    :return:
    """
    Z = 0.001
    unigram_table = []
    num_total_words = sum([c for w, c in word_count])
    for vo in vocabs:
        # print (word_count[vo][1] / float(num_total_words)) ** 0.75

        unigram_table.extend([vo] * int(((word_count[vo][1] / float(num_total_words)) ** 0.75) / Z))

    print("vocabulary size", len(vocabs))
    print("unigram_table size:", len(unigram_table))
    return unigram_table


class DataPipeline:
    def __init__(self, data, vocabs, word_count, data_index=0, use_noise_neg=True):
        self.data = data
        self.data_index = data_index
        if use_noise_neg:
            self.unigram_table = noise(vocabs, word_count)
        else:
            self.unigram_table = vocabs

    def get_neg_data(self, batch_size, num, target_inputs):
        """
        sample the negative data. Don't use np.random.choice(), it is very slow.
        :param batch_size: int
        :param num: int
        :param target_inputs: []
        :return:
        """
        neg = np.zeros((num))
        for i in range(batch_size):
            delta = random.sample(self.unigram_table, num)
            while target_inputs[i] in delta:
                delta = random.sample(self.unigram_table, num)
            neg = np.vstack([neg, delta])
        return neg[1: batch_size + 1]

    def generate_batch(self, batch_size, num_skips, skip_window):
        """
        get the data batch
        :param batch_size:
        :param num_skips:
        :param skip_window:
        :return: target batch and context batch
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window, target, skip_window ]
        buffer = deque(maxlen=span)

        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels
