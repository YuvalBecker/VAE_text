import requests
from bs4 import BeautifulSoup
from nltk.tokenize import  word_tokenize
import numpy as np
import re

import string
import json

# -> create my own vocabulary
# Loading the specific embeddings matrixes 50 and 100 dimentions: (can choose which one is better)
# Suppose to be generic for every transcript :

UNKNOWN = 2

class Vocab:
    def __init__(self, name,contain_non_embeddings=True,embedding_size=50,embedding_path=None,word2idx_path='/home/yuval/PycharmProjects/NLP_FINAL_PROJ/EMbeddings/wordsidx.txt'):
        self.name = name
        self.word2index = {}
        self.contain_non_embeddings= contain_non_embeddings
        self.embedding= np.zeros(( 1,embedding_size ) )
        self.index2word = {UNKNOWN: '__unk__'}
        self.n_words = 1
        self.vec_embbedding = np.load(embedding_path)

        # Loading the mapper:
        self.wordidx = json.load(open(word2idx_path))

    def index_word(self, word, write=True):
        word =word.replace(" ", "")
        if word not in self.word2index:
            if write:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                # Creating the word embedding:
                if word in self.wordidx:
                    vec_embeddings=  self.vec_embbedding[ self.wordidx[word] ]
                    self.embedding= np.concatenate([self.embedding,np.expand_dims(vec_embeddings,axis=0) ],axis=0)
                    self.n_words = self.n_words + 1

                else: # not in the pretrained embeddings :
                    print('Not include in the embedding space: '+ word)
                    if self.contain_non_embeddings:
                        self.embedding= np.concatenate([self.embedding,np.zeros((1,50))],axis=0)
                        self.n_words = self.n_words + 1
                    else:
                        return -1

            else:
                return UNKNOWN
        return self.word2index[word]

class Data_for_train(Vocab):
    def __init__(self, url_list,charecters = ['pumbaa', 'timon'],contain_non_embbedding = True,embedding_size=None,embedding_path='C:\\Users\\yuval\\PycharmProjects\\Gen_vae_text\\glove_vectors_50d.npy',word2idx_path='C:\\Users\\yuval\\PycharmProjects\\Gen_vae_text\\wordsidx.txt'):
        self.url_list= url_list
        self.charecters =charecters # Which charecters to train of.
        self.dict ={ }
        self.contain_non_embbedding =contain_non_embbedding
        self.embedding_size =embedding_size

        super().__init__(name='lion',contain_non_embeddings= self.contain_non_embbedding,embedding_size=self.embedding_size,embedding_path=embedding_path,word2idx_path=word2idx_path)

    def Data_clean(self,text):
        text= re.sub(r'{.+?}', '', text)
        text= re.sub(r'[.]', '', text)

        text= re.sub(r'\(.+?\)', '', text)

        text= re.sub("[\.][ ]*[\.][ ]*[\.]", " ", text)
        text= re.sub("[ ][ ]*]", " ", text)

        text= re.sub("\\(.*\\)", "", text)
        text= re.sub("\\(.*\\n", "", text)

        text= re.sub("\\[.*\\]", "", text)

        text= re.sub(r'\'', '', text)
        text=text.lower()

        text = re.sub(r"([,])", r" ", text)
        text = re.sub(r"([-])", r" ", text)
        return text

    def create_trainning_Data(self):

        for url in self.url_list:
            html = requests.get(url).text


            soup = BeautifulSoup(html, "lxml")
            Text_parse = soup.text

            L = len(Text_parse)
            Text_parse = self.Data_clean(Text_parse)

            # Choosing the charecters to train:
            for ind, c in enumerate(self.charecters):
                if c not in self.dict:
                    self.dict[c] = {}
                    self.dict[c+'str'] = {}

                start = 0
                while (start != -1):
                    finish = 0

                    # Search where the charecter speaks :
                    start = Text_parse.find(c + ':', start + 1, L)
                    #end = Text_parse.find('\n', start+len(c) + 1, L)
                    indexes_iter= re.finditer('\n.*:',Text_parse[start+len(c) + 1: L])
                    try:
                        val = next(indexes_iter)
                        end = val.start()
                    except:
                        finish =1
                        start =-1
                    if (finish==0):
                        add_sentence = Text_parse[start + len(c) + 1:start+end - 1+ len(c) + 1]
                        if len(add_sentence) > 1:
                            words = word_tokenize(add_sentence.rstrip(''))
                            list_words_idx = []
                            self.dict[c + 'str'][len(self.dict[c + 'str'])] = add_sentence +'\n'

                            if len(words) > 3: # threshold over minimum length of sentence
                                for word in words:
                                    if word[0] == '':  # because it adds empty charecter at the first word
                                        word = word[1:]
                                        word.maketrans('', '', string.punctuation)


                                    index = self.index_word(word)
                                    if self.contain_non_embbedding:
                                        list_words_idx.append(index)
                                    else:
                                        if index != -1: #
                                            list_words_idx.append(index)
                                self.dict[c][len(self.dict[c])] = list_words_idx
def freinds_parsing():
    # func to read all friends episodes :
    num_seasons = 10
    base_str = 'https://fangj.github.io/friends/season/'
    url_list=[]
    for i in range(1,num_seasons,1):
        for ep in range(1,22,1): # will take only 22 episodes from each season
            if ep > 9:
                url_list.append(base_str + '0'+str(i) +str(ep) + '.html')

            else:
                url_list.append(base_str + '0'+str(i) + '0'+str(ep) + '.html')
    return  url_list


if __name__ == '__main__':
    # Add the url lists containning the transcripts:
    url_list = [r'https://transcripts.fandom.com/wiki/The_Lion_King',
                r'https://transcripts.fandom.com/wiki/The_Lion_King_II:_Simba%27s_Pride',
                r'https://transcripts.fandom.com/wiki/Never_Everglades',
                r'http://www.lionking.org/scripts/TLK1.5-Script.html#specialpower']
    # Add the names of the charecters you want to train:
    trained_charecters = ['timon','pumbaa']
    train_prepare = Data_for_train(url_list=url_list,charecters=trained_charecters)
    train_prepare.create_trainning_Data()
    dict_Data =train_prepare.dict
    print(np.shape( train_prepare.embedding ) )
    print(train_prepare.n_words )

    # Check if the embedding went well:
    vecs50 = np.load("/home/yuval/PycharmProjects/NLP_FINAL_PROJ/EMbeddings/glove_vectors_50d.npy")
    print( train_prepare.embedding[train_prepare.word2index['and']] - vecs50[train_prepare.wordidx[ 'and']]  )
