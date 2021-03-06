import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dimension, is_cuda=True):
        super(SkipGram, self).__init__()

        # set the parameters for the embedding model
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.is_cuda = is_cuda

        # the embeddings for the center and context words
        self.center_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        if self.is_cuda:
            self.center_embeddings = self.center_embeddings.cuda()
            self.context_embeddings = self.context_embeddings.cuda()

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
        score = torch.mul(v, u)
        score = torch.sum(score, dim=1)
        pos_val = F.logsigmoid(score).squeeze()

        # u_hat: [batch_size, neg_size, emb_dim]
        u_negative = self.context_embeddings(negative_samples)
        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]

        # neg_val: [batch_size]
        neg_score = torch.bmm(u_negative, v.unsqueeze(2)).squeeze()
        neg_val = torch.sum(neg_score, dim=1)
        neg_val = F.logsigmoid(-1 * neg_val).squeeze()

        loss = pos_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.center_embeddings(inputs)




