import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd

import random
import torch
import pickle

'''
    Code for training and testing a word-level sentiment classifier based on positive and negative words.
'''


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to train the oracle sentiment classifier.')
    parser.add_argument('-e', '--emb', help='The path to the W2V embedding file.', required=True)
    parser.add_argument('-p', '--pos', help='The path to the positive words.', required=True)
    parser.add_argument('-n', '--neg', help='The path to the negative words.', required=True)
    parser.add_argument('-ev', '--eval', help='The path to the list of words to be evaluated (e.g. names).', required=True)
    parser.add_argument('-o', '--out', help='The path to the output folder where we store the weights.', required=True)
    parser.add_argument('-t', '--tag',
                        help='The tag with which we will assign the file names to distinguish the different embeddings.',
                        required=False, default='')
    parser.add_argument('-f', '--flag', default='train', required=False)
    parser.add_argument('-c', '--classifier', default='', required=False)

    return parser.parse_args()


def read_word_lexicon(path, indexes):
    words = [w.lower().strip() for w in open(path, 'rt').readlines() if w.strip().lower() in indexes]
    return words


def vec(w, emb):
    try:
        return emb.loc[w].as_matrix()
    except:
        return None


def train_model(emb_file, pos_file, neg_file, out_dir, tag):
    tag = os.path.basename(emb_file) + '' + tag

    # read word embeddings as DataFrame
    emb = pd.read_table(emb_file, sep=' ', header=None, skiprows=1, index_col=0)
    indexes = emb.index.values

    # prepare training data
    pos_words = read_word_lexicon(pos_file, indexes)
    neg_words = read_word_lexicon(neg_file, indexes)

    # balance datasets
    if len(pos_words) > len(neg_words):
        random.shuffle(pos_words)
        pos_words = pos_words[:len(neg_words)]
    else:
        random.shuffle(neg_words)
        neg_words = neg_words[:len(pos_words)]
    words = pos_words + neg_words
    print("Number of training examples (pos/neg words) per class: ", len(words))

    # convert to word vectors
    word_vectors = [vec(w, emb) for w in words]

    labels = [1 for pos_word in pos_words] + [-1 for neg_word in neg_words]
    assert len(word_vectors) == len(labels)
    train_X, test_X, train_y, test_y = train_test_split(word_vectors, labels, test_size=0.1, random_state=1)

    # train and predict
    clf = LogisticRegression(solver='liblinear', random_state=1)
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)

    # save weights
    tensor_weight = torch.Tensor(clf.coef_)
    pickle.dump(tensor_weight, open(out_dir + '/' + tag + '_weights.pckl', 'wb'), pickle.HIGHEST_PROTOCOL)
    outstr = str(clf.intercept_) + '\t'
    for val in tensor_weight[0]:
        outstr += str(val.item()) + '\t'
    open(p.out + '/' + tag + '_weights.txt', 'wt').write(outstr)
    # print('Intercept: %.3f' % clf.intercept_)

    # save model
    pickle.dump(clf, open(out_dir + '/' + tag + '_model.clf', 'wb'), pickle.HIGHEST_PROTOCOL)

    # performance
    print("Accuracy: ", accuracy_score(test_y, y_pred, normalize=True, sample_weight=None))
    print(classification_report(test_y, y_pred, digits=4))
    print("Confusion matrix: ", confusion_matrix(test_y, y_pred))

    return clf, emb


if __name__ == '__main__':
    random.seed(1)
    p = get_arguments()

    if p.flag == 'train':
        clf, emb = train_model(p.emb, p.pos, p.neg, p.out, p.tag)
    elif p.flag == 'test':
        clf = pickle.load(open(p.classifier, 'rb'))
        emb = pd.read_table(p.emb, sep=' ', header=None, skiprows=1, index_col=0)
    print("---------------------------------------")

    # sentiment scores
    eval_words = [e.strip().lower() for e in open(p.eval, 'rt').readlines()]
    out_str = 'word\tPOS\tNEG\tdiff\n'
    # print(out_str)

    avg_pos = 0
    avg_neg = 0
    avg_diff = 0
    counter = 0
    for word in eval_words:
        # check if eval word exists in embeddings
        word_vec = vec(word, emb)
        if word_vec is None:
            continue

        # predict class probabilities
        counter += 1
        avg_pos += clf.predict_proba([word_vec])[:, 1]
        avg_neg += clf.predict_proba([word_vec])[:, 0]
        avg_diff += abs(clf.predict_proba([word_vec])[:, 1] - .5)
        out_str += '%s\t%.3f\t%.3f\t%.3f\n' % (word, avg_pos, avg_neg, avg_diff)
        # print('%s\t%.3f\t%.3f\t%.3f' % (word, avg_pos, avg_neg, avg_diff))

    # compute average
    out_str += '\nAVG\t%.3f\t%.3f\t%.3f\n' % (avg_pos/counter, avg_neg/counter, avg_diff/counter)
    print('\nAVG\t%.3f\t%.3f\t%.3f\n' % (avg_pos / counter, avg_neg / counter, avg_diff / counter))

    # output the results
    base = os.path.basename(p.emb) + '_results.txt'
    open(base, 'wt').write(out_str)
