
import pandas as pd
import scipy.io
from ast import literal_eval
import numpy as np
from keras.utils import pad_sequences


class data_generator():

    def __init__(self,dataset='flickr8k'):
        self._dataset = dataset
    
    def build_vocab(self,word_count_threshold=5):
        print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))

        sentences = pd.read_csv(self._dataset+'_dataset.txt', delimiter='\t')['captions'].to_list()
        word_counts = {}
        nsents = 0
        max_len = 0
        for sent in sentences:
            tokened = literal_eval(sent)
            if len(tokened) > max_len:
                max_len = len(tokened)
            nsents += 1
            for w in tokened:
                word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

        ixtoword = {}
        ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
        wordtoix = {}
        wordtoix['#START#'] = 0 # make first vector be the start token
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        self._wordtoix = wordtoix
        self._ixtoword = ixtoword
        self._max_len = len(wordtoix)

        print(f'Length of the word index: {self._max_len}')
    
    def build_generator(dataset,wordtoix,max_len,batch_size =32):
        captions = []
        images = []

        df = pd.read_csv(dataset+'_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
                x = next(iter)
                c.append(literal_eval(x[1][3]))
                imgs.append(x[1][1])
        features_path = f'data/{dataset}/vgg_feats.mat'
        features_struct = scipy.io.loadmat(features_path)['feats']
        count = 0
        while True:
            for text,im in zip(c,imgs):
                current_image = features_struct[:,im]
                word_idx = [wordtoix[i] for i in text if i in wordtoix]
                word_idx.append(0)
                captions.append(word_idx)
                count+=1
                images.append(current_image)
                if count>=batch_size:
                    images = np.asarray(images)
                    captions = pad_sequences(captions, maxlen=max_len, padding='post')
                    yield [[images, captions], captions]
                    captions = []
                    images = []
                    count = 0

