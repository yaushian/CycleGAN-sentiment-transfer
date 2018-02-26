import os
import numpy as np
import json
import re
import gensim
import random

class data_utils():
    def __init__(self,args):
        self.data_path = 'data/feature_twitter.txt'
        self.num_batch = 12
        self.sent_length = args.sequence_length
        self.batch_size = args.batch_size
        
        self.set_dictionary('word2vec_model/dictionary.json')
        self.set_word2vec_model('word2vec_model/model')


    def set_dictionary(self, dict_file):
        if os.path.exists(dict_file):
            fp = open(dict_file,'r')
            self.word_id_dict = json.load(fp)
            print('word number:',len(self.word_id_dict))

            self.BOS_id = self.word_id_dict['__BOS__']
            self.EOS_id = self.word_id_dict['__EOS__']

            self.id_word_dict = [[]]*len(self.word_id_dict)
            for word in self.word_id_dict:
                self.id_word_dict[self.word_id_dict[word]] = word
        else:
            print('where is dictionary file QQ?')


    def set_word2vec_model(self,name):
        word_array = []
        self.word2vec_model = gensim.models.Word2Vec.load(name)
        for i in range(len(self.id_word_dict)):
            word = self.id_word_dict[i]
            word_array.append(self.word2vec_model[word])

        self.word_array = np.array(word_array)


    def sent2id(self,sent):
        vec = np.zeros((self.sent_length),dtype=np.int32) + self.EOS_id
        pat = re.compile('(\W+)')
        sent_list = ' '.join(re.split(pat,sent.lower().strip())).split()
        i = 0
        for word in sent_list:
            if word in self.word_id_dict:
                vec[i] = self.word_id_dict[word]
                i += 1
            if i>=self.sent_length:
                break
        return vec


    def vec2sent(self,vecs):
        sent = []
        for vec in vecs:
            possible_words = self.word2vec_model.most_similar([vec],topn=10)
            word = possible_words[0][0]
            sent.append(word)

        return ' '.join(sent)


    def id2sent(self,indices):
        sent = []
        for index in indices:
            sent.append(self.id_word_dict[index])
        return ' '.join(sent)


    def data_generator(self,class_id):
        while(1):
            with open(self.data_path) as fp:
                for line in fp:
                    s = line.strip().split('+++$+++')
                    if int(s[0])==class_id and random.randint(0,10) >= 2:
                        yield self.sent2id(s[1].strip())

    def X_data_generator(self):
        return self.data_generator(0)


    def Y_data_generator(self):
        return self.data_generator(1)


    def gan_data_generator(self):
        one_X_batch = []
        one_Y_batch = []

        for one_X,one_Y in zip(self.X_data_generator(),self.Y_data_generator()):
            one_X_batch.append(one_X)
            one_Y_batch.append(one_Y)
            if len(one_X_batch) == self.batch_size*self.num_batch:
                one_X_batch = np.array(one_X_batch).reshape(self.num_batch,self.batch_size,-1)
                one_Y_batch = np.array(one_Y_batch).reshape(self.num_batch,self.batch_size,-1)
                yield one_X_batch,one_Y_batch
                one_X_batch = []
                one_Y_batch = []


    def pretrain_generator_data_generator(self):
        one_X_batch = []
        one_Y_batch = []

        for one_X,one_Y in zip(self.X_data_generator(),self.Y_data_generator()):
            one_X_batch.append(one_X)
            one_Y_batch.append(one_Y)
            if len(one_X_batch) == self.batch_size:
                one_X_batch = np.array(one_X_batch)
                one_Y_batch = np.array(one_Y_batch)
                yield one_X_batch,one_Y_batch
                one_X_batch = []
                one_Y_batch = []


    def test_data_generator(self):
        one_batch = np.zeros([self.batch_size,self.sent_length])
        batch_count = 0
        for line in open('seq2seq_test.txt'):
            one_batch[batch_count] = self.sent2id(line.strip())
            batch_count += 1
            if batch_count == self.batch_size:
                yield one_batch
                batch_count = 0
                one_batch = np.zeros([self.batch_size,self.sent_length])

        if batch_count >= 1:
            yield one_batch