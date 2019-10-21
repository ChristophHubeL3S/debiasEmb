import argparse
from evaluation.downstream_classifier import DownstreamClassifier


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to train evaluate the trained embeddings.')
    parser.add_argument('-d', '--data', help='The labeled dataset for training and testing.', required=True)
    parser.add_argument('-e', '--embeddings', help='The embeddings to evaluate.', required=True)
    parser.add_argument('-o', '--out', help='The output path for the model.', required=False, default="./model_output.h5")
    parser.add_argument('-dim', '--embeddings_dimension', help='Dimension of the input embeddings.', required=False, default=100, type=int)
    parser.add_argument('-dr', '--dropout_rate', help='Dropout rate to be used during training.', required=False, default=0.5, type=float)
    parser.add_argument('-ep', '--epochs', help='Number of epochs.', required=False, default=50, type=int)
    parser.add_argument('-bs', '--batch_size', help='Batch size to be used during training.', required=False, default=64, type=int)
    parser.add_argument('-lr', '--learning_rate', help='Initial learning rate.', required=False, default=0.001, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    p = get_arguments()
    cl = DownstreamClassifier(dataset_path=p.data, embeddings_path=p.embeddings, model_output_path=p.out,
                              emb_dim=p.embeddings_dimension, dropout_rate=p.dropout_rate, epochs=p.epochs,
                              batch_size=p.batch_size, learning_rate=p.learning_rate)