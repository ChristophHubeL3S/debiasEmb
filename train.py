from embeddings.word2vec import Word2Vec
from embeddings.debias_word2vec import DebiasWord2Vec
import pickle
import argparse
import os

'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to train different types of word embeddings.')
    parser.add_argument('-w', '--weights', help='The logistic regression weights.', required=False)
    parser.add_argument('-i', '--input', help='The input path to the training data.', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=True)
    parser.add_argument('-n', '--names', help='The list of names for which we want to debias the embeddings',
                        required=False)
    parser.add_argument('-m', '--model', help='The type of embedding we want to train.', required=False, default='w2v')
    parser.add_argument('-v', '--vocab_size', help='The size of the vocabulary.', required=False, default=10000,
                        type=int)
    parser.add_argument('-e', '--epochs', help='Training epochs.', required=False, default=10000, type=int)
    parser.add_argument('-sw', '--skip_gram_window', help='Skip-Gram window.', required=False, default=5, type=int)
    parser.add_argument('-nn', '--neg_samples', help='Negative samples.', required=False, default=20, type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size for training the embeddings.', required=False,
                        default=100, type=int)
    parser.add_argument('-d', '--emb_dim', help='Embeddings dimensions', required=False, default=100, type=int)
    parser.add_argument('-ef', '--emb_file', help='Embedding file name', required=False, default='vector_emb.txt')
    parser.add_argument('-c', '--cuda', help='Use Cuda?', required=False, default=False, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    p = get_arguments()

    if p.model == 'w2v':
        model = Word2Vec(data_path=p.input, vocab_size=p.vocab_size, emb_size=p.emb_dim, output_dir=p.out_dir, cuda=p.cuda)
    else:
        lr_weights = pickle.load(open(p.weights, 'rb'))  # weights required when using debias!!!
        names_dict = set(open(p.names, 'rt').readlines())
        model = DebiasWord2Vec(data_path=p.input, vocab_size=p.vocab_size, emb_size=p.emb_dim, output_dir=p.out_dir,
                               sent_lr_weights=lr_weights, names_dict=names_dict, cuda=p.cuda)
        # get the indexes of the names in our dictionary
        names_dict_idx = set()
        for word in names_dict:
            if word.strip() in model.word2index:
                names_dict_idx.add(model.word2index[word.strip()])

        print(names_dict_idx)

    # train model
    model.train(train_steps=p.epochs, skip_window=p.skip_gram_window, num_skips=2, num_neg=p.neg_samples,
                batch_size=p.batch_size)

    # save vector txt file
    if not os.path.exists(p.out_dir + '/' + p.model):
        os.mkdir(p.out_dir + '/' + p.model)

    model.save_vector_txt(path_file=p.out_dir + '/' + p.model + '/' + p.emb_file)
