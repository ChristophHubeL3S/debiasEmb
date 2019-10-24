import os

import time
from torch.optim import SGD

from models.skip_gram import SkipGram
from utils.utils_new import DataPipeline, read_data, construct_skipgram_training_instances
from utils.vector_handle import *


class Word2Vec:
    def __init__(self, data_path, vocab_size, emb_size, learning_rate=0.01, output_dir='', is_cuda=True,
                 emb_model_states=''):
        self.data, self.word_list = read_data(data_path)

        self.data, self.word_count, self.word2index, self.index2word = construct_skipgram_training_instances(self.data,
                                                                                                             self.word_list,
                                                                                                             vocab_size)
        self.is_cuda = is_cuda == 'True'
        self.model = SkipGram(len(self.word_count), emb_size, is_cuda=self.is_cuda)
        if self.is_cuda:
            self.model = self.model.cuda()

        if len(emb_model_states):
            self.model.load_state_dict(torch.load(emb_model_states))

        self.model_optim = SGD(self.model.parameters(), lr=learning_rate)
        self.output_dir = output_dir

    def train(self, train_steps, skip_window=1, num_neg=20, batch_size=128, batch_step=10000, model='w2v',
              emb_file='model'):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        pipeline = DataPipeline(self.data, self.index2word.keys(), self.word_count)

        # check how many iterations we will need for the entire data per epoch
        no_iter_per_epoch = int(float(len(pipeline.data)) / batch_step)

        print ('Extracted the training data, now starting to train for %d batches of size %d' % (
        no_iter_per_epoch, batch_size))
        for step in range(train_steps):
            pipeline.sentence_pointer = 0
            start_time = time.time()

            print('Started training for epoch %d' % step)

            # iterate over all the batches in the data
            avg_loss = 0
            counter = 0
            while pipeline.sentence_pointer < len(pipeline.data):
                batch_inputs, batch_labels = pipeline.generate_batch_step(skip_window=skip_window,
                                                                          batch_step=batch_step, batch_size=batch_step)

                for batch_idx, _ in enumerate(batch_inputs):
                    center_batch = batch_inputs[batch_idx]
                    context_batch = batch_labels[batch_idx]

                    neg_labels = pipeline.get_neg_data(len(center_batch), num=num_neg, target_inputs=center_batch,
                                                       context_inputs=context_batch)

                    center_batch = torch.tensor(center_batch, dtype=torch.long)
                    context_batch = torch.tensor(context_batch, dtype=torch.long)
                    neg_labels = torch.tensor(neg_labels, dtype=torch.long)

                    if self.is_cuda:
                        center_batch = center_batch.cuda()
                        context_batch = context_batch.cuda()
                        neg_labels = neg_labels.cuda()

                    self.model_optim.zero_grad()
                    loss = self.model(center_batch, context_batch, neg_labels)
                    loss.backward()
                    self.model_optim.step()

                    avg_loss += loss.item()
                    counter += 1

                    print ('Trained for epoch %d and %d -th batch, processed up to %d sentences with a loss of %.3f' % (
                        step, batch_idx, pipeline.sentence_pointer, loss.item()))

            avg_loss /= counter
            end_time = time.time() - start_time
            print (
            'Finished processing training for epoch %d in %s with a loss of %.3f' % (step, str(end_time), avg_loss))

            self.save_vector_txt(path_file=self.output_dir + '/' + emb_file + '_' + model + '_' + str(step) + '.emb')
            self.save_model(out_path=self.output_dir + '/' + emb_file + '_' + model + '_' + str(step) + '.model')

    def save_model(self, out_path):
        torch.save(self.model.state_dict(), out_path)

    def get_list_vector(self):
        sd = self.model.state_dict()
        return sd['center_embeddings.weight'].tolist()

    def save_vector_txt(self, path_file):
        embeds = self.model.center_embeddings.weight.data.tolist()
        fo = open(path_file, 'w')
        fo.write(str(len(embeds)) + ' ' + str(len(embeds[0])) + '\n')
        for idx in range(len(embeds)):
            word = self.index2word[idx]
            embed = ' '.join(map(str, embeds[idx]))
            fo.write(word + ' ' + embed + '\n')

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
