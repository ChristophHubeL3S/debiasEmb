import torch
import torch.nn as nn


class DebiasedSkipGram(nn.Module):
    def __init__(self, vocab_size, sent_word_model_weights, emb_dimension, names_dict, lr_intercept=1.10, cuda=False):
        super(DebiasedSkipGram, self).__init__()

        # set the parameters for the embedding model
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension

        # the target names for which we wanna debias the embeddings
        self.names_dict = names_dict

        # the embeddings for the center and context words
        if cuda:
            self.center_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True).cuda()
            self.context_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True).cuda()
        else:
            self.center_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
            self.context_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        # here we store the LR weights which are used to predetermine the sentiment of a word
        if cuda:
            self.word_semantics = sent_word_model_weights.cuda()
        else:
            self.word_semantics = sent_word_model_weights

        self.intercept = lr_intercept
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()
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

        # here we see if the embedding of the center word has any prior sentiment value associated with it
        sent_val = self.sigmoid(torch.sum(v * self.word_semantics, dim=1) + self.intercept)
        # for idx in enumerate(center_input):
        #     if idx not in self.names_dict:
        #         sent_val[idx] = .5

        # our target is to minimize the loss value such that the sigmoid function produces a value of 0.5,
        # which means that the model is not able to classify correctly the sentiment of the embedding
        sent_val = - (abs(sent_val - .5) * 1)

        # print '%.3f\t%.3f\t%.3f' % (pos_val, sent_val, neg_val)
        loss = pos_val + sent_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.center_embeddings(inputs)
