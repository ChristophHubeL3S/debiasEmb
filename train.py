from embeddings.word2vec import Word2Vec
from embeddings.debias_word2vec import DebiasWord2Vec
import torch
import argparse
import os
import numpy as np
import re

'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to train different types of word embeddings.')
    parser.add_argument('-w', '--weights', help='The logistic regression weights.', required=False)
    parser.add_argument('-uc', '--is_cuda', help='Check if it should use the GPUs', required=False, default=False, type=bool)
    parser.add_argument('-i', '--input', help='The path to the input text file that the embeddings will be trained on.', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=True)
    parser.add_argument('-n', '--names', help='The list of names for which we want to debias the embeddings',
                        required=False)
    parser.add_argument('-m', '--model', help='The type of embedding we want to train.', required=True, default='w2v')
    parser.add_argument('-v', '--vocab_size', help='The size of the vocabulary.', required=False, default=10000,
                        type=int)
    parser.add_argument('-e', '--epochs', help='Training epochs.', required=False, default=10000, type=int)
    parser.add_argument('-sw', '--skip_gram_window', help='Skip-Gram window.', required=False, default=5, type=int)
    parser.add_argument('-nn', '--neg_samples', help='Negative samples.', required=False, default=20, type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size for training the embeddings.', required=False,
                        default=100, type=int)
    parser.add_argument('-d', '--emb_dim', help='Embeddings dimensions', required=False, default=100, type=int)
    parser.add_argument('-ef', '--emb_file', help='Embedding file name', required=False, default='vector_emb.txt')
    parser.add_argument('-c', '--check_names', help='Debias only specific names', required=False, default=False)
    parser.add_argument('-es', '--emb_states', help='Embedding states', required=False, default='')

    return parser.parse_args()


if __name__ == '__main__':
    p = get_arguments()

    if p.model == 'w2v':
        print("Using w2v skip-gram approach.")
        model = Word2Vec(data_path=p.input, vocab_size=p.vocab_size, emb_size=p.emb_dim, output_dir=p.out_dir,
                         is_cuda=p.is_cuda, emb_model_states=p.emb_states)
    elif p.model == 'debiasEmb':
        print("Using debiasEmb")
        lr_weights = np.array(open(p.weights, 'rt').read().strip().split('\t'))
        intercept = float(re.sub(r'\[|\]', '', lr_weights[0]))
        lr_weights = lr_weights[1:].astype(np.float)
        lr_weights = torch.tensor(lr_weights, dtype=torch.float)

        check_names = p.check_names == 'True'
        names_dict = set([line.strip().lower() for line in open(p.names, 'rt').readlines()])

        print('Dictionary names length %d' % len(names_dict))
        print(lr_weights.shape)
        model = DebiasWord2Vec(data_path=p.input, vocab_size=p.vocab_size, emb_size=p.emb_dim, output_dir=p.out_dir,
                               is_cuda=p.is_cuda, sent_lr_weights=lr_weights, check_names=check_names,
                               intercept=intercept, names_dict=names_dict, emb_model_states=p.emb_states)

    # train model
    model.train(train_steps=p.epochs, skip_window=p.skip_gram_window, num_neg=p.neg_samples,
                batch_size=p.batch_size, batch_step=p.batch_size, model=p.model, emb_file=p.emb_file)

    # save vector txt file
    if not os.path.exists(p.out_dir + '/' + p.model):
        os.mkdir(p.out_dir + '/' + p.model)

    model.save_vector_txt(path_file=p.out_dir + '/' + p.model + '/' + p.emb_file)
