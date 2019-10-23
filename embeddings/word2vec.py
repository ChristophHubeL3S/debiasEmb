import os

from torch.optim import SGD

from models.skip_gram import SkipGram
from utils.utils import DataPipeline, read_data, construct_skipgram_training_instances
from utils.vector_handle import *

"""Code for training and working with a skip-gram model"""


class Word2Vec:
    def __init__(self, data_path, vocab_size, emb_size, learning_rate=0.1, output_dir='', cuda=False):
        self.corpus = read_data(data_path)

        self.data, self.word_count, self.word2index, self.index2word = construct_skipgram_training_instances(self.corpus, vocab_size)
        self.vocabs = list(set(self.data))
        self.cuda = cuda

        if self.cuda:
            self.model = SkipGram(vocab_size, emb_size).cuda()
        else:
            self.model = SkipGram(vocab_size, emb_size)

        self.model_optim = SGD(self.model.parameters(), lr=learning_rate)
        self.output_dir = output_dir

    def train(self, train_steps, skip_window=1, num_skips=2, num_neg=20, batch_size=128, data_offest=0):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        avg_loss = 0
        pipeline = DataPipeline(self.data, self.vocabs, self.word_count, data_offest)

        for step in range(train_steps):
            batch_inputs, batch_labels = pipeline.generate_batch(batch_size, num_skips, skip_window)
            batch_neg = pipeline.get_neg_data(batch_size, num_neg, batch_inputs)

            if self.cuda:
                batch_inputs = torch.tensor(batch_inputs, dtype=torch.long).cuda()
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
                batch_neg = torch.tensor(batch_neg, dtype=torch.long).cuda()
            else:
                batch_inputs = torch.tensor(batch_inputs, dtype=torch.long)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long)
                batch_neg = torch.tensor(batch_neg, dtype=torch.long)

            loss = self.model(batch_inputs, batch_labels, batch_neg)
            self.model_optim.zero_grad()
            loss.backward()
            self.model_optim.step()

            avg_loss += loss.item()

            if step % 2000 == 0 and step > 0:
                avg_loss /= 2000
                print('Average loss at step ', step, ': ', avg_loss)
                avg_loss = 0

            # checkpoint
            if step % 100000 == 0 and step > 0:
                torch.save(self.model.state_dict(), self.output_dir + '/model_step%d.pt' % step)

        # save model at last
        torch.save(self.model.state_dict(), self.output_dir + '/model_step%d.pt' % train_steps)

    def save_model(self, out_path):
        torch.save(self.model.state_dict(), out_path + '/model.pt')

    def get_list_vector(self):
        sd = self.model.state_dict()
        return sd['center_embeddings.weight'].tolist()

    def save_vector_txt(self, path_file):
        embeddings = self.get_list_vector()
        fo = open(path_file, 'w')
        for idx in range(len(embeddings)):
            word = self.index2word[idx]
            embed = embeddings[idx]
            embed_list = [str(i) for i in embed]
            line_str = ' '.join(embed_list)
            fo.write(word + ' ' + line_str + '\n')
        fo.close()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def vector(self, index):
        self.model.predict(index)

    def most_similar(self, word, top_k=8):
        index = self.word2index[word]
        index = torch.tensor(index, dtype=torch.long).unsqueeze(0)
        emb = self.model.predict(index)
        sim = torch.mm(emb, self.model.center_embeddings.weight.transpose(0, 1))
        nearest = (-sim[0]).sort()[1][1: top_k + 1]
        top_list = []
        for k in range(top_k):
            close_word = self.index2word[nearest[k].item()]
            top_list.append(close_word)
        return top_list
