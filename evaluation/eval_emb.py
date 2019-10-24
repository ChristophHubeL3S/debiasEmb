import pandas as pd
import pickle
import argparse

'''
    Evaluate word embeddings.
'''


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to evaluate the sentiment bias of word embeddings.')
    parser.add_argument('-e', '--emb', help='The input path to the word embedding.', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=True)
    parser.add_argument('-ev', '--eval', help='The list of eval words (e.g. names) for which we want to debias the embeddings', required=False)
    parser.add_argument('-m', '--model', help='The path to the pre-trained Logistic Regression model.', required=True)
    parser.add_argument('-t', '--emb_type', help='The type of the word embedding.', required=True, default='Debiased')

    return parser.parse_args()


def vec(w, emb):
    try:
        return emb.loc[w].as_matrix()
    except Exception as e:
        return None


if __name__ == '__main__':
    p = get_arguments()

    emb = pd.read_table(p.emb, sep=' ', header=None, skiprows=0, index_col=0)

    # load classifier
    clf = pickle.load(open(p.model, 'rb'))

    # get the indexes of eval words
    eval_words = set(open(p.eval, 'rt').readlines())

    outstr = ''
    avg = []
    for word in eval_words:
        w2v = vec(word.strip(), emb)

        if w2v is None:
            continue

        pred_score = clf.predict_proba([w2v])[:, 1]
        print('Sentiment score for\t%s\t%s\t%.3f' % (word.strip(), p.emb_type.strip(), pred_score))
        outstr += 'Sentiment score for\t%s\t%s\t%.3f\n' % (word.strip(), p.emb_type.strip(), pred_score)

        avg.extend(pred_score)

    print('Average sentiment score\t%s\t%.3f' % (p.emb_type.strip(), (sum(avg)/len(avg))))
    outstr += 'Average sentiment score\t%s\t%.3f\n' % (p.emb_type.strip(), (sum(avg) / len(avg)))
    open(p.out_dir, 'wt').write(outstr)
