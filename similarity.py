import os
import pandas as pd
import numpy as np
import random
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchUtilities import *
from sklearn.neighbors import NearestNeighbors
from pandasUtilities import dfChunks
import networkx as nx
import regex as re

import matplotlib.pyplot as plt
import seaborn as sb

from defaults import *


def stringToChars(s,max_len=200):
    return b'<' + stringToAscii(s)[:max_len] + b'>'


class VectorModel(nn.Module):
    def __init__(self,d_in=96,d_recurrent=300,d_out=300,recurrent_layers=1,bidirectional=False):
        super().__init__()

        d_gru_out = d_recurrent*recurrent_layers*(1+int(bidirectional))

        self.char_embedding = nn.Linear(d_in,d_recurrent)
        self.embedding_dropout = nn.Dropout()
        self.gru = nn.GRU(d_recurrent,d_recurrent,recurrent_layers,bidirectional=bidirectional,batch_first=True,dropout=0.5 if recurrent_layers > 1 else 0)
        self.dropout = nn.Dropout()
        self.projection = nn.Linear(d_gru_out,d_out)

    def forward(self,W):
        X = PackedSequence(self.embedding_dropout(self.char_embedding(W.data)),W.batch_sizes)
        H,h = self.gru(X)

        #Concat layer and direction h vectors
        v = h.permute(1,0,2).contiguous().view(h.shape[1],-1)
        v = self.dropout(v)
        v = self.projection(v)

        return v



class SimilarityModel():
    def __init__(self,cuda=False,d=300,d_recurrent=None,recurrent_layers=1,bidirectional=False):
        self.args = locals()
        del self.args['self']

        if d_recurrent is None:
            self.d_recurrent = d
        else:
            self.d_recurrent = d_recurrent

        self.model = VectorModel(d_recurrent=d,d_out=d,recurrent_layers=recurrent_layers,bidirectional=bidirectional)

        if cuda:
            self.model = self.model.cuda()

        self.params = {'params':self.model.parameters()}

        self.optimizer = torch.optim.Adam([self.params])

        self.lossHistory = pd.DataFrame()


    def trainMinibatch(self,componentData,componentWeightsDF,batch_size):
        cuda = next(self.model.parameters()).is_cuda

        if not batch_size <= 0.5*len(componentData):
            raise Exception('Batch size must be smaller than half the number of components')

        h = {'batch_size':batch_size,'between_loss':0,'within_loss':0,'within_size':0}

        # Make a little utility vector that contains 0 and 1 for entering into the loss function
        target = torch.tensor([0,1]).float()
        if cuda:
            target = target.cuda()

        self.optimizer.zero_grad()
        for pair in np.random.choice(componentWeightsDF['component'],size=(batch_size,2),p=componentWeightsDF['weight'],replace=False):
            pairData = [componentData[i] for i in pair]

            if cuda:
                pairData = [packedToCuda(packed) for packed in pairData]

            # Compute vectors for each pair
            v = [self.model(packed) for packed in pairData]

            # Compute component mean vector for each pair
            v_mean = [x.mean(dim=0) for x in v]

            # Accumulate within-component gradient
            for i in 0,1:
                if v[i].shape[0] > 1:

                    d = ((v[i] - v_mean[i].detach())**2).sum()
                    score = (-d).exp()
                    score = torch.clamp(score,min=0,max=1)

                    withinLoss = F.binary_cross_entropy(score,target[1]) / float(batch_size)
                    withinLoss.backward(retain_graph=True)

                    h['within_loss'] += withinLoss.item()
                    h['within_size'] += v[i].shape[0]

            # Accumulate between-component gradient
            d = ((v_mean[0] - v_mean[1])**2).sum()
            score = (-d).exp()
            score = torch.clamp(score,min=0,max=1)

            betweenLoss = F.binary_cross_entropy(score,target[0]) / float(batch_size)
            betweenLoss.backward()

            h['between_loss'] += betweenLoss.item()


        nn.utils.clip_grad_norm_(self.model.parameters(),5)
        self.optimizer.step()

        return h



    def train(self,matcher,epochs=10,epoch_size=100,minibatch_size=1,lr=0.001,weight_decay=1e-6,save_as=None,verbose=True):
        cuda = next(self.model.parameters()).is_cuda
        self.model.train()

        componentsDF = matcher.componentsDF()
        componentsDF['count'] = componentsDF['string'].apply(lambda s: matcher.counts[s])

        componentWeightsDF = componentsDF.groupby('component')['count'].sum().to_frame('weight').reset_index()
        componentWeightsDF['weight'] = componentWeightsDF['weight'] / componentWeightsDF['weight'].sum()

        # Compile each component into a packed sequence
        componentData = []
        for i,df in componentsDF.groupby('component'):
            chars = df['string'].apply(stringToChars)
            packed,chars = bytesToPacked1Hot(chars,clamp_range=(31,126))

            assert i == len(componentData)
            componentData.append(packed)


        # Set up batch size and learning rate schedules
        try:
            lr_start,lr_end = lr
            lr_schedule = np.geomspace(lr_start,lr_end,epochs)
        except:
            lr_schedule = lr*np.ones(epochs)

        try:
            b_start,b_end = minibatch_size
            b_schedule = np.linspace(b_start,b_end,epochs).astype(int)
        except:
            b_schedule = minibatch_size*np.ones(epochs).astype(int)

        bar_freq = epoch_size//100

        startingEpoch = self.lossHistory['epoch'].max() + 1 if len(self.lossHistory) else 0
        for i,epoch in enumerate(range(startingEpoch,startingEpoch + epochs)):
            if verbose: print('\nTraining epoch {}'.format(epoch))

            for g in self.optimizer.param_groups:
                g['lr'] = lr_schedule[i]
                g['weight_decay'] = weight_decay


            epochHistory = []
            for b in range(epoch_size):
                h = self.trainMinibatch(componentData,componentWeightsDF,b_schedule[i])
                h['batch'] = b
                epochHistory.append(h)

                if not b % bar_freq:
                    if verbose: print('|',end='')

            epochHistoryDF = pd.DataFrame(epochHistory)
            epochHistoryDF['epoch'] = epoch
            epochHistoryDF['within_loss'] = epochHistoryDF['within_loss'] / epochHistoryDF['within_size']
            epochHistoryDF.loc[epochHistoryDF['within_size']==0,'within_loss'] = np.nan

            if verbose: print('\nMean loss: Between={:.3f}, Within={:.3f}'.format(*epochHistoryDF[['between_loss','within_loss']].mean()))

            self.lossHistory = self.lossHistory.append(epochHistoryDF)

            if save_as:
                self.save(save_as)
                if verbose: print('Model saved as ',save_as)

        return self.lossHistory[self.lossHistory['epoch']>=startingEpoch]



    def vectorizeStrings(self,strings,batch_size=100,max_len=200):
        self.model.eval()
        cuda = next(self.model.parameters()).is_cuda

        chars = [stringToChars(s,max_len=max_len) for s in strings]
        vecs = []
        for i in range(0,len(chars),batch_size):
            batchChars = chars[i:i+batch_size]
            packed,sorted_chars = bytesToPacked1Hot(batchChars,clamp_range=(31,126))
            chars_to_id = {s:i for i,s in enumerate(sorted_chars)}
            if cuda:
                packed = packedToCuda(packed)
            sorted_vecs = self.model(packed).data.cpu().numpy()
            selector = np.array([chars_to_id[s] for s in batchChars])
            batch_vecs = sorted_vecs[selector,:]
            vecs.append(batch_vecs)

        return np.vstack(vecs)


    def findSimilar(self,strings,min_score=0.95,batch_size=100,drop_duplicates=True):
        strings = sorted(set(strings))

        vecs = self.vectorizeStrings(strings,batch_size=batch_size)

        radius = -np.log(max(min_score,1e-8))

        nearestNeighbors = NearestNeighbors(radius=radius,metric='euclidean')
        nearestNeighbors.fit(vecs)

        distances,matches = nearestNeighbors.radius_neighbors(vecs)

        matchPairs = [(i,j) for i,query_matches in enumerate(matches) for j in query_matches]
        pairDistances = np.array([d for query_distances in distances for d in query_distances])
        scores = np.exp(-pairDistances**2)

        if drop_duplicates:
            matchDF = pd.DataFrame([(strings[i],strings[j]) for i,j in matchPairs],columns=['string0','string1'])
            matchDF['score'] = scores

        if drop_duplicates:
            matchPairs = np.sort(np.array(matchPairs),axis=1)

            matchDF = pd.DataFrame([(strings[i],strings[j]) for i,j in matchPairs],columns=['string0','string1'])
            matchDF['score'] = scores

            matchDF = matchDF.drop_duplicates(['string0','string1'])

        matchDF = matchDF[matchDF['string0'] != matchDF['string1']].copy()

        return matchDF


    def save(self,filename):
        state = {
        'args':self.args,
        'model_state': self.model.state_dict(),
        'optimizer_state': self.optimizer.state_dict(),
        'loss_history':self.lossHistory
        }
        torch.save(state,filename)



def loadSimilarityModel(filename,cuda=False):
    state = torch.load(filename)

    state['args'].update({'cuda':cuda})

    similarityModel = SimilarityModel(**state['args'])
    similarityModel.model.load_state_dict(state['model_state'])
    similarityModel.optimizer.load_state_dict(state['optimizer_state'])

    similarityModel.lossHistory = state['loss_history']

    return similarityModel



def plotLossHistory(historyDF):
    historyDF = historyDF.sort_values(['epoch','batch']).reset_index()
    historyDF['t'] = historyDF.index.get_level_values(0)

    ax = plt.subplot(3,1,1)
    historyDF.groupby('epoch').mean().plot(x='t',y=['between_loss','within_loss'],ax=ax)

    if len(historyDF) > 1000:
        sampleDF = historyDF.sample(1000)
    else:
        sampleDF = historyDF

    plt.subplot(3,1,2)
    plt.scatter(sampleDF['t'],sampleDF['between_loss'],s=3)

    plt.subplot(3,1,3)
    plt.scatter(sampleDF['t'],sampleDF['within_loss'],s=3,color='C1')



# if __name__ == '__main__':
#
#     Test code
#
#     import nama
#     from nama.matcher import Matcher
#     df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
#     df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])
#
#     # Initialize the matcher
#     matcher = Matcher()
#
#     # Add the strings we want to match to the match graph
#     matcher.addStrings(df1['name'])
#     matcher.addStrings(df2['name'])
#     matcher.addStrings(['other'])
#
#
#     matcher.matchHash(nama.hashes.corpHash)
#
#     matcher.G.edges()
#
#     similarityModel = SimilarityModel(cuda=True,d=100,d_recurrent=100,recurrent_layers=2,bidirectional=True)
#
#     matcher.matchSimilar(similarityModel)
#
#     matcher.matchesDF()
#
#     import time
#
#     t0 = time.time()
#     newHistoryDF = similarityModel.train(matcher,epochs=3)
#     print(time.time()-t0)
#
#     similarityModel.save(os.path.join(modelDir,'test_model.bin'))
#
#     loadedModel = loadSimilarityModel(os.path.join(modelDir,'test_model.bin'))
#
#     loadedModel.train(matcher,epochs=3)
