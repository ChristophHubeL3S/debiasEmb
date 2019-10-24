import gzip
import random
from collections import *

import numpy as np


def read_data(data_path):
    print('Reading data file %s...' % data_path)

    word_list = []
    if data_path.endswith('gz'):
        fin = gzip.open(data_path, 'rt')
    else:
        fin = open(data_path, 'rt')

    data = []
    for idx, line in enumerate(fin):
        if idx >= 1000000:
            break
        sentence_tokens = line.lower().split()
        word_list.extend(sentence_tokens)
        data.append(sentence_tokens)

    print('Finished reading data file. The corpus contains %d sentences and %d words' % (len(data), len(word_list)))
    return data, word_list


def noise(vocabs, word_count):
    """
    generate noise distribution
    :param vocabs:
    :param word_count:
    :return:
    """
    Z = 0.001
    unigram_table = []
    num_total_words = sum(word_count.values())
    for vo in vocabs:
        unigram_table.extend([vo] * int(((word_count[vo] / float(num_total_words)) ** 0.75) / Z))

    print ("vocabulary size", len(vocabs))
    print ("unigram_table size:", len(unigram_table))
    return unigram_table


def construct_skipgram_training_instances(sentence_words, words, n_words):
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

    word_freq = Counter(words)
    count = [['UNK', -1]]
    count.extend([[el, word_freq[el]] for el in word_freq if word_freq[el] >= n_words])

    dictionary = dict()
    counts_idx = dict()
    counts_idx[0] = -1
    for word, _ in count:
        dictionary[word] = len(dictionary)
        counts_idx[dictionary[word]] = _
    data = []
    unk_count = 0

    for sentence in sentence_words:
        sentence_data = []
        for word in sentence:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # UNK index is 0
                unk_count += 1
            sentence_data.append(index)
        data.append(sentence_data)
    counts_idx[0] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counts_idx, dictionary, reversed_dictionary


class DataPipeline:
    def __init__(self, data, vocabs, word_count, use_noise_neg=True):
        self.sentence_pointer = 0

        self.data = data
        if use_noise_neg:
            self.unigram_table = noise(vocabs, word_count)
        else:
            self.unigram_table = vocabs

    def generate_batch(self, batch_size, skip_window, neg_items=10, batch_step=100000):
        """
        get the data batch
        :param batch_size:
        :param skip_window:
        :return: target batch and context batch
        """
        batch = np.zeros(batch_size, dtype=np.int32)
        labels = np.zeros(batch_size, dtype=np.int32)

        batch_list = []
        labels_list = []
        neg_list = []

        counter = 0
        batch_slot_counter = 0

        for s_idx, sentence in enumerate(self.data):
            if s_idx < self.sentence_pointer:
                continue
            self.sentence_pointer += 1

            if len(batch_list) >= batch_step and batch_slot_counter == batch_size:
                break

            # generate the word pairs for training
            for idx, center_word in enumerate(sentence):
                for cx_idx in range(idx - skip_window, skip_window + idx + 1):
                    if cx_idx < 0 or cx_idx >= len(sentence) or cx_idx == idx:
                        continue

                    batch[counter] = center_word
                    labels[counter] = sentence[cx_idx]

                    counter += 1
                    batch_slot_counter += 1

                    if counter % batch_size == 0:
                        batch_list.append(batch)
                        labels_list.append(labels)

                        neg_list.append(np.random.choice(self.unigram_table, size=(batch_size, neg_items)))

                        counter = 0
                        batch_slot_counter = 0

                        batch = np.zeros(batch_size, dtype=np.int32)
                        labels = np.zeros(batch_size, dtype=np.int32)

        return batch_list, labels_list, neg_list

    def generate_batch_step_name_dict(self, skip_window, batch_size=10000, batch_step=100000, names_dict=None):
        """
        get the data batch
        :param batch_size:
        :param skip_window:
        :return: target batch and context batch
        """
        batch_list = []
        labels_list = []
        names_dict_batch = []

        current_center = []
        current_labels = []
        current_name_dict = []

        for s_idx, sentence in enumerate(self.data):
            if s_idx < self.sentence_pointer:
                continue

            # keep track of which sentence we have parsed already
            self.sentence_pointer += 1

            if self.sentence_pointer % batch_step == 0:
                break

            center, labels, names = self.add_instances_to_batch_names_dict(sentence, skip_window, names_dict)

            current_center.extend(center)
            current_labels.extend(labels)

            current_name_dict.extend(names)

            if len(current_center) >= batch_size:
                batch_list.append(np.asarray(current_center))
                labels_list.append(np.asarray(current_labels))
                names_dict_batch.append(np.asarray(current_name_dict))

                current_labels = []
                current_center = []
                current_name_dict = []

        if len(current_center) != 0:
            batch_list.append(np.asarray(current_center))
            labels_list.append(np.asarray(current_labels))
            names_dict_batch.append(np.asarray(current_name_dict))

        return batch_list, labels_list, names_dict_batch

    def generate_batch_step(self, skip_window, batch_size=10000, batch_step=100000):
        """
        get the data batch
        :param batch_size:
        :param skip_window:
        :return: target batch and context batch
        """
        batch_list = []
        labels_list = []

        current_center = []
        current_labels = []

        for s_idx, sentence in enumerate(self.data):
            if s_idx < self.sentence_pointer:
                continue

            # keep track of which sentence we have parsed already
            self.sentence_pointer += 1

            if self.sentence_pointer % batch_step == 0:
                break

            center, labels = self.add_instances_to_batch(sentence, skip_window)

            current_center.extend(center)
            current_labels.extend(labels)

            if len(current_center) >= batch_size:
                batch_list.append(np.asarray(current_center))
                labels_list.append(np.asarray(current_labels))

                current_labels = []
                current_center = []

        if len(current_center) != 0:
            batch_list.append(np.asarray(current_center))
            labels_list.append(np.asarray(current_labels))

        return batch_list, labels_list

    '''
        Construct the positive and negative pairs from this sentence.
    '''

    def add_instances_to_batch(self, sentence, skip_window):
        center = []
        labels = []

        for idx, center_word in enumerate(sentence):
            for cx_idx in range(idx - skip_window, skip_window + idx + 1):
                if cx_idx < 0 or cx_idx >= len(sentence) or cx_idx == idx:
                    continue

                center.append(center_word)
                labels.append(sentence[cx_idx])

        return center, labels

    '''
        Construct the positive and negative pairs from this sentence.
    '''

    def add_instances_to_batch_names_dict(self, sentence, skip_window, names_dict):
        center = []
        labels = []
        names = []

        for idx, center_word in enumerate(sentence):
            in_dict_center = center_word in names_dict
            for cx_idx in range(idx - skip_window, skip_window + idx + 1):
                if cx_idx < 0 or cx_idx >= len(sentence) or cx_idx == idx:
                    continue

                center.append(center_word)
                labels.append(sentence[cx_idx])

                # in_dict_label = sentence[cx_idx] in names_dict
                names.append(1 if in_dict_center else 0)

        return center, labels, names

    def get_neg_data(self, batch_size, num, target_inputs, context_inputs):
        """
        sample the negative data. Don't use np.random.choice(), it is very slow.
        :param batch_size: int
        :param num: int
        :param target_inputs: []
        :return:
        """

        # testing
        print("Batch_size: ", batch_size)
        print("Number of negative samples: ", num)
        print(target_inputs)

        neg = np.zeros((num))
        for i in range(batch_size):
            delta = random.sample(self.unigram_table, num)
            while target_inputs[i] in delta or context_inputs[i] in delta:
                delta = random.sample(self.unigram_table, num)
            neg = np.vstack([neg, delta])
        return neg[1: batch_size + 1]

        # print(self.unigram_table)
        #
        # neg = np.zeros((num))
        # for i in range(batch_size):
        #     delta = random.sample(self.unigram_table, num)
        #     while target_inputs[i] in delta:
        #         # print("delta: ", delta)
        #         # print("target input: ", target_inputs[i])
        #         delta = random.sample(self.unigram_table, num)
        #     neg = np.vstack([neg, delta])
        # return neg[1: batch_size + 1]
