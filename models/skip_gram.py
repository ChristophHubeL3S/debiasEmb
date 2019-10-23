import torch
import torch.nn as nn

"""The skip-gram model itself"""


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dimension):
        super(SkipGram, self).__init__()

        # set the parameters for the embedding model
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension

        # the embeddings for the center and context words
        self.center_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        self.log_sigmoid = nn.LogSigmoid()

        initrange = 0.5 / self.emb_dimension

        # init the weight as original word2vec do.
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(0, 0)

    def forward(self, center_input, context_output, negative_samples):
        """
       :param target_input: [batch_size]
       :param context: [batch_size]
       :param neg: [batch_size, neg_size]
       :return:
       """

        v = self.center_embeddings(center_input)
        u = self.context_embeddings(context_output)

        # check the similarity between the center and context words
        pos_val = self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()

        # u_hat: [batch_size, neg_size, emb_dim]
        u_negative = self.context_embeddings(negative_samples)
        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]

        # neg_vals: [batch_size, neg_size]
        neg_vals = torch.bmm(u_negative, v.unsqueeze(2)).squeeze(2)

        # neg_val: [batch_size]
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = pos_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.center_embeddings(inputs)




