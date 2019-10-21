#!/usr/bin/env python3

import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import *
from keras.layers import Embedding, Dense
from keras.models import *
from keras.optimizers import *
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import asarray


class DownstreamClassifier:

    def __init__(self, dataset_path, embeddings_path, model_output_path, emb_dim, dropout_rate, epochs, batch_size, learning_rate):

        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.model_output_path = model_output_path
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        sentences, labels = self.read_input()
        print("Number of sentences: {}".format(len(sentences)))

        vocab_w2i, vocab_i2w, word2vec = self.read_word_emb()
        sentence_indexes = []
        max_len = 0

        for sentence in sentences:
            sentence_idx = self.convert_to_index(sentence, vocab_w2i)
            sentence_indexes.append(sentence_idx)
            if len(sentence_idx) > max_len:
                max_len = len(sentence_idx)

        self.X = pad_sequences(sentence_indexes, max_len, value=vocab_w2i['UNK'], padding='pre')

        # class labels
        self.classes = set(labels)  # get unique labels
        class2vec = {}
        for i, class_x in enumerate(self.classes):
            vec = np.zeros(len(self.classes), dtype='int8')
            vec[i] = 1
            class2vec[class_x] = vec

        self.y = []
        for label in labels:
            self.y.append(class2vec[label])

        self.embeddings = self.create_embedding_matrix(vocab_i2w, vocab_w2i, word2vec)
        model = self.build_model_FeedForward(max_len)

        model = self.train_model(model)
        model.save(self.model_output_path)

    def read_word_emb(self):
        print("loading embeddings...")
        word2vec = KeyedVectors.load_word2vec_format(self.embeddings_path, binary=False)
        vocab_w2i = {key: idx + 1 for idx, key in enumerate(word2vec.vocab)}
        vocab_w2i['UNK'] = 0
        vocab_i2w = {}
        for key in vocab_w2i:
            vocab_i2w[vocab_w2i[key]] = key
        print("Index of word test: {}".format(vocab_w2i["test"]))
        return vocab_w2i, vocab_i2w, word2vec

    def convert_to_index(self, input_string, vocab_w2i):
        string = input_string.lower()
        string = re.sub('[^0-9a-z ]+', '', string)
        parts = string.split(' ')
        indexes = []
        for word in parts:
            if word not in vocab_w2i:
                indexes.append(vocab_w2i['UNK'])  # represent unknown word as UNK index
            else:
                indexes.append(vocab_w2i[word])  # represent word as index
        return indexes

    def read_input(self):
        # file should contain two columns: sentence, label
        df = pd.read_csv(self.dataset_path, delimiter="\t")
        df = df.drop(df[df.label == "neut"].index)
        print(df.head())
        sentences = df["sentence"]
        labels = df["label"].tolist()
        return sentences, labels

    def create_embedding_matrix(self, vocab_i2w, vocab_w2i, word2vec):
        # create embedding matrix for using pre-trained embeddings
        embeddings = 1 * np.random.randn(len(vocab_i2w) + 1, self.emb_dim)  # initializing matrix
        embeddings[0] = 0  # So that the padding will be ignored
        for word, index in vocab_w2i.items():  # items outputs (key, value) pairs
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)[:self.emb_dim]
        return embeddings

    def build_model_FeedForward(self, max_len):
        # build model
        print("building model...")
        main_input = Input(shape=(max_len,), dtype='float', name='main_input')
        tensor = Embedding(len(self.embeddings), self.emb_dim, weights=[self.embeddings], input_length=max_len, trainable=False)(main_input)
        tensor = Dense(80, activation="relu")(tensor)
        tensor = Dropout(self.dropout_rate)(tensor)
        tensor = Dense(30, activation="relu")(tensor)
        tensor = Dropout(self.dropout_rate)(tensor)
        tensor = Dense(10, activation="relu")(tensor)
        tensor = Dropout(self.dropout_rate)(tensor)
        tensor = Flatten()(tensor)
        output = Dense(len(self.classes), activation='sigmoid')(tensor)  # use sigmoid for >2 classes

        model = Model(input=main_input, output=output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate),
                           metrics=['acc'])  # use categorical_crossentropy for >2 classes
        return model

    def get_class_labels(self, predictions):
        predicted_labels = []
        for p in predictions:
            if not isinstance(p, list):
                p = p.tolist()
            label = p.index(max(p))
            predicted_labels.append(label)
        return predicted_labels

    def train_model(self, model):
        print("training model...")
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1)  # data will be shuffled before splitting
        model.fit(asarray(X_train), asarray(y_train), epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

        # test
        prob_predictions = model.predict(asarray(X_test))
        print(prob_predictions[0])
        class_predictions = self.get_class_labels(prob_predictions)
        y_test_cleaned = self.get_class_labels(y_test)
        print("test accuracy: {}".format(accuracy_score(class_predictions, y_test_cleaned)))

        label_counts = {}
        for label in y_test_cleaned:
            if label in label_counts.keys():
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        print(label_counts)
        return model










