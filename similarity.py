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


def stringToChars(s,max_len=200):
    return b'<' + stringToAscii(s)[:max_len] + b'>'


# def batchedNearestNeighbors(vecs,radius,metric,batch_size=100000):
#     for i in range(0,vecs.shape[0],batch_size):
#
#         nearestNeighbors = NearestNeighbors(radius=radius,metric=metric)
#         nearestNeighbors.fit(vecs[i:i+batch_size])
#
#         for j in range(0,vecs.shape[0],batch_size):
#             distances,matches = nearestNeighbors.radius_neighbors(vecs[j:j+batch_size])
#
#             for q,(query_distances,query_matches) in enumerate(zip(distances,matches)):
#                 for d,k in zip(query_distances,query_matches):
#                     yield (i+k,j+q),d
#
#
# def batchedNearestNeighbors(vecs,radius,metric,batch_size=100000):
#     for b in range(0,vecs.shape[0],batch_size):
#
#         nearestNeighbors = NearestNeighbors(radius=radius,metric=metric)
#         nearestNeighbors.fit(vecs[b:b+batch_size])
#
#         distances,matches = nearestNeighbors.k_neighbors(vecs)
#
#         for j,(query_distances,query_matches) in enumerate(zip(distances,matches)):
#             for d,i in zip(query_distances,query_matches):
#                 yield (b+i,j),d


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


    def trainMinibatch(self,components,componentWeights,batch_size):
        cuda = next(self.model.parameters()).is_cuda

        if not batch_size <= 0.5*len(components):
            raise Exception('Batch size must be smaller than half the number of components')

        h = {'batch_size':batch_size,'between_loss':0,'within_loss':0,'within_size':0}

        # Make a little utility vector that contains 0 and 1 for entering into the loss function
        target = torch.tensor([0,1]).float()
        if cuda:
            target = target.cuda()

        self.optimizer.zero_grad()
        for pair in np.random.choice(components,size=(batch_size,2),p=componentWeights,replace=False):
            pairChars = [set(stringToChars(s) for s in component) for component in pair]

            with torch.no_grad():
                pairData = [bytesToPacked1Hot(chars,clamp_range=(31,126))[0] for chars in pairChars]

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

        components = list(matcher.components())
        componentWeights = np.array([sum(matcher.counts[s] for s in component) for component in components])
        componentWeights = componentWeights / componentWeights.sum()

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
                h = self.trainMinibatch(components,componentWeights,b_schedule[i])
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


    def findSimilar(self,strings,min_score=0,n=10,batch_size=100,leaf_size=50,drop_duplicates=True):
        strings = sorted(set(strings))
        n = min(n,len(strings))

        vecs = self.vectorizeStrings(strings,batch_size=batch_size)

        # radius = np.sqrt(-np.log(max(min_score,1e-8)))

        # print(list(batchedNearestNeighbors(vecs,radius=radius,metric='l2',batch_size=neighbor_batch_size)))

        # matchPairs,matchDistances = zip(*batchedNearestNeighbors(vecs,radius=radius,metric='l2',batch_size=neighbor_batch_size))
        # matchScores = np.exp(-np.array(matchDistances)**2)

        nearestNeighbors = NearestNeighbors(n_neighbors=n,algorithm='ball_tree',leaf_size=leaf_size)
        nearestNeighbors.fit(vecs)

        distances,matches = nearestNeighbors.kneighbors(vecs)

        matchPairs = np.vstack([np.kron(np.arange(len(strings)),np.ones(n).astype(int)),matches.ravel()]).T

        print(matchPairs)
        matchPairs = np.sort(np.array(matchPairs),axis=1)

        matchScores = np.exp(-distances.ravel()**2)

        # radius = np.sqrt(-np.log(max(min_score,1e-8)))
        #
        # nearestNeighbors = NearestNeighbors(radius=radius,algorithm='ball_tree',leaf_size=leaf_size)
        # nearestNeighbors.fit(vecs)
        #
        # distances,matches = nearestNeighbors.radius_neighbors(vecs)
        #
        # matchPairs = [(i,j) for i,query_matches in enumerate(matches) for j in query_matches]
        # # pairDistances = np.array([d for query_distances in distances for d in query_distances])
        # pairDistances = np.hstack(distances)
        # matchScores = np.exp(-pairDistances**2)

        matchDF = pd.DataFrame(matchPairs,columns=['string0','string1'])
        matchDF['score'] = matchScores

        matchDF = matchDF[matchDF['string0'] != matchDF['string1']].copy()

        if min_score > 0:
            matchDF = matchDF[matchDF['score'] >= min_score]

        if drop_duplicates:
            matchDF = matchDF.drop_duplicates(['string0','string1'])

        for c in 'string0','string1':
            matchDF[c] = matchDF.apply(lambda i: strings[i])

        matchDF = matchDF.reset_index(drop=True)

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



if __name__ == '__main__':

    # Test code
    import nama
    from nama.matcher import Matcher
    import cProfile as profile

    # Initialize the matcher
    matcher = Matcher(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO','ABC Inc.','XYZ Co.'])

    # Add some corpHash matches
    matcher.matchHash(nama.hashes.corpHash)

    # Initalize a new, untrained similarity model
    similarityModel = SimilarityModel(cuda=True,d=20,d_recurrent=20,recurrent_layers=2,bidirectional=True)

    matcher.suggestMatches(similarityModel,min_score=0)


    profile.run('similarityModel.train(matcher,epochs=1)',sort='tottime')

    profile.run('matcher.suggestMatches(similarityModel,min_score=0)',sort='tottime')



    df0 = matcher.suggestMatches(similarityModel,min_score=0,leaf_size=2)


    df0 = matcher.suggestMatches(similarityModel,min_score=0)
    df1 = matcher.suggestMatches(similarityModel,min_score=0,neighbor_batch_size=2)
    df2 = matcher.suggestMatches(similarityModel,min_score=0,neighbor_batch_size=3)


np.array([[1,2,3],[4,5,6]]).ravel()


np.kron(np.arange(5),np.ones(3).astype(int))
