import torch
import torch.nn as nn
import torch.nn.functional as F


class DebiasedSkipGram(nn.Module):
    def __init__(self, vocab_size, sent_word_model_weights=None, emb_dimension=100, names_dict=None, lr_intercept=1.10,
                 check_names=False, is_cuda=True):
        super(DebiasedSkipGram, self).__init__()

        # set the parameters for the embedding model
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.is_cuda = is_cuda

        # the target names for which we wanna debias the embeddings
        self.names_dict = names_dict
        self.check_names = check_names

        # the embeddings for the center and context words
        self.center_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.context_embeddings = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        # here we store the LR weights which are used to predetermine the sentiment of a word
        self.word_semantics = sent_word_model_weights
        self.intercept = lr_intercept

        if self.is_cuda:
            self.center_embeddings = self.center_embeddings.cuda()
            self.context_embeddings = self.context_embeddings.cuda()

            self.word_semantics = self.word_semantics.cuda()

        initrange = 0.5 / self.emb_dimension

        # init the weight as original word2vec do.
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(0, 0)

    def forward(self, center_input, context_output, negative_samples, names):
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

        # here we see if the embedding of the center word has any prior sentiment value associated with it
        sent_val = F.sigmoid(torch.sum(v * self.word_semantics, dim=1))
        sent_val = - (abs(sent_val - .5))
        if self.check_names:
            # for idx, center_idx in enumerate(center_input):
            #     if center_idx not in self.names_dict and context_output[idx] not in self.names_dict:
            #         sent_val.data[idx] = .5
            sent_val = sent_val * names

        # our target is to minimize the loss value such that the sigmoid function produces a value of 0.5,
        # which means that the model is not able to classify correctly the sentiment of the embedding
        # sent_val = - (abs(sent_val - .5))

        loss = pos_val + sent_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.center_embeddings(inputs)
