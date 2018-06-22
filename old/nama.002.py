import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchUtilities import *
from torchModels import Chars2Vec, Scaler
import random
from unidecode import unidecode
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import NearestNeighbors
from pandasUtilities import dfChunks
from itertools import combinations



def charIdsToPacked1Hot(chars):
    chars = sorted(set(chars),key=len,reverse=True)
    if not chars[-1]:
        raise Exception('Cannot pack empty bytestring')

    tensors = [charsToTensor(s) for s in chars]
    packed = nn.utils.rnn.pack_sequence(tensors)
    packed = packedCharIdsTo1Hot(packed)

    return packed,chars


# minibatchDF = trainingDF.sample(3)
# random_samples= 20
# rand_sample_prob =0.5
def generateMinibatches(trainingDF,cuda=False,max_len=100,base_size=10,resample_prob=0.5,shuffle=True):
    trainingDF = trainingDF[trainingDF['match'].notnull()].copy()

    if shuffle:
        trainingDF = trainingDF.sample(frac=1)

    for c in 'query','candidate':
        trainingDF['{}_chars'.format(c)] = trainingDF['{}_string'.format(c)].apply(lambda s: stringToChars(s)[:max_len] + b'\n')

    # #Track match pairs (in either order)
    # pairs = {(c0,c1) for i,c0,c1 in trainingDF[['query_chars','candidate_chars']].itertuples()}
    # pairs.update({(c1,c0) for c0,c1 in pairs})

    matches = {(c0,c1) for i,c0,c1,m in trainingDF[['query_chars','candidate_chars','match']].itertuples() if m>0}
    matches.update({(c1,c0) for c0,c1 in matches})

    for minibatchDF in dfChunks(trainingDF,base_size):
        batchPairs = {(c0,c1) for c0,c1 in zip(minibatchDF['query_chars'],minibatchDF['candidate_chars'])}
        batchPairs.update({(c1,c0) for c0,c1 in batchPairs})

        with torch.no_grad():
            packed,chars = charIdsToPacked1Hot((s for c in ('query','candidate') for s in minibatchDF['{}_chars'.format(c)]))

            selector = np.vstack(np.triu_indices(len(chars),k=1)).T

            inBatch = np.array([(chars[i],chars[j]) in batchPairs for i,j in selector])
            randPairs = np.random.uniform(0,1,size=selector.shape[0]) < resample_prob

            selector = selector[inBatch | randPairs,:]

            match = np.array([(chars[i],chars[j]) in matches for i,j in selector]).astype(float)

            # selected =
            #
            # randSelector = allPairs[np.random.choice(allPairs.shape[0],size=random_samples),:]
            # randMatch = np.array([(chars[i],chars[j]) in matches for i,j in randSelector]).astype(float)
            #
            # selector = np.vstack([selector,randSelector])
            # match = np.hstack([match,randMatch])
            #
            #
            #

            # chars_to_id = {s:i for i,s in enumerate(chars)}
            # selector = np.array([[chars_to_id[s] for s in minibatchDF['{}_chars'.format(c)]] for c in ('query','candidate')]).T
            # match = minibatchDF['match'].values

            #Add random samples, reusing the packed chars and assuming out-of-sample pairs are not matches
            # if random_samples:
            #     allPairs = np.vstack(np.triu_indices(len(chars),k=1)).T
            #
            #     randSelector = allPairs[np.random.choice(allPairs.shape[0],size=random_samples),:]
            #     randMatch = np.array([(chars[i],chars[j]) in matches for i,j in randSelector]).astype(float)
            #
            #     selector = np.vstack([selector,randSelector])
            #     match = np.hstack([match,randMatch])

            selector = torch.from_numpy(selector).long()
            match = torch.from_numpy(match).float()

        if cuda:
            packed.data.pin_memory()
            selector.data.pin_memory()
            match.data.pin_memory()

            packed = packedToCuda(packed)
            selector = selector.cuda()
            match = match.cuda()

        yield packed,selector,match


def trainMinibatch(batchData,modelPackage):
    packed,selector,match = batchData

    modelPackage['model'].train()

    sorted_vecs = modelPackage['model'](packed)
    vecs = [sorted_vecs.index_select(0,selector[:,i]) for i in [0,1]]
    dist = F.pairwise_distance(vecs[0],vecs[1])

    match_prob =(-dist**2).exp()
    match_prob = torch.clamp(match_prob,min=0,max=1)

    loss = F.binary_cross_entropy(match_prob,match,size_average=False)

    modelPackage['optimizer'].zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(modelPackage['model'].parameters(),5)

    modelPackage['optimizer'].step()

    return {'size':match.size(0),'loss':float(loss.data.cpu())}


class vectorModel(nn.Module):
    def __init__(self,d_in=96,d_hidden=300,recurrent_layers=1,bidirectional=False):
        super().__init__()

        d_projection = d_hidden*recurrent_layers*(1+int(bidirectional))

        self.char_embedding = nn.Linear(d_in,d_hidden)
        self.embedding_dropout = nn.Dropout()
        self.gru = nn.GRU(d_hidden,d_hidden,recurrent_layers,bidirectional=bidirectional,batch_first=True,dropout=0.5 if recurrent_layers > 1 else 0)
        self.dropout = nn.Dropout()
        self.projection = nn.Linear(d_projection,d_projection)
        self.dropout2 = nn.Dropout()
        self.projection2 = nn.Linear(d_projection,d_projection)

    def forward(self,W):
        X = PackedSequence(self.embedding_dropout(self.char_embedding(W.data)),W.batch_sizes)
        H,h = self.gru(X)
        #Concat layer and direction h vectors
        h = h.permute(1,0,2).contiguous().view(h.shape[1],-1)
        v = self.dropout(h)
        v = self.projection(F.tanh(v)) + v
        v = self.dropout2(h)
        v = self.projection2(F.tanh(v)) + v
        return v


def newModel(cuda=False,d=300,recurrent_layers=1,bidirectional=False,lr=1e-3,weight_decay=1e-6):
    modelPackage = {'args':locals(),'loss_history':pd.DataFrame()}


    modelPackage['model'] = vectorModel(d_hidden=d,recurrent_layers=recurrent_layers,bidirectional=bidirectional)

    if cuda:
        modelPackage['model'] = modelPackage['model'].cuda()

    modelPackage['params'] = {'params':modelPackage['model'].parameters(),'lr':lr,'weight_decay':weight_decay}
    modelPackage['optimizer'] = torch.optim.Adam([modelPackage['params']])

    return modelPackage



def saveModelPackage(modelPackage,filename):
    state = {
    'args':modelPackage['args'],
    'loss_history':modelPackage['loss_history'],
    'model_state': modelPackage['model'].state_dict(),
    'optimizer_state': modelPackage['optimizer'].state_dict(),
    }
    torch.save(state,filename)



def loadModelPackage(filename):
    state = torch.load(filename)

    modelPackage = newModel(**state['args'])
    modelPackage['model'].load_state_dict(state['model_state'])
    modelPackage['optimizer'].load_state_dict(state['optimizer_state'])

    modelPackage['args'] = state['args']
    modelPackage['loss_history'] = state['loss_history']

    return modelPackage



def trainModel(modelPackage,trainingDF,minibatch_size=10,resample_prob=0.5,epochs=10,exit_function=None,verbose=False):
    cuda = next(modelPackage['model'].parameters()).is_cuda
    epochBatchIterator = (generateMinibatches(trainingDF,cuda=cuda,base_size=minibatch_size,resample_prob=resample_prob) for epoch in range(epochs))
    bar_freq = max(1,int((len(trainingDF)/minibatch_size)/100))
    modelPackage['loss_history'] = trainWithHistory(lambda b: trainMinibatch(b,modelPackage),epochBatchIterator,modelPackage['loss_history'],
                                    exit_function=exit_function,verbose=verbose,bar_freq=bar_freq)

    return modelPackage['loss_history']




def vectorizeStrings(strings,modelPackage,batch_size=100):
    modelPackage['model'].eval()

    chars = [stringToChars(s) for s in strings]
    vecs = []
    for i in range(0,len(chars),batch_size):
        batchChars = chars[i:i+batch_size]
        packed,sorted_chars = charIdsToPacked1Hot(batchChars)
        chars_to_id = {s:i for i,s in enumerate(sorted_chars)}
        if next(modelPackage['model'].parameters()).is_cuda:
            packed = packedToCuda(packed)
        sorted_vecs = modelPackage['model'](packed).data.cpu().numpy()
        selector = np.array([chars_to_id[s] for s in batchChars])
        batch_vecs = sorted_vecs[selector,:]
        vecs.append(batch_vecs)

    return np.vstack(vecs)


def findMatches(queryStrings,candidateStrings,modelPackage,n_matches=1):
    queryStrings = sorted(set(queryStrings))
    candidateStrings = sorted(set(candidateStrings))

    #Compute vector representations of strings
    vecs = [vectorizeStrings(strings,modelPackage) for strings in (queryStrings,candidateStrings)]

    #Use sklearn to efficiently find nearest neighbours
    nearestNeighbors = NearestNeighbors(n_neighbors=1,metric='euclidean')
    nearestNeighbors.fit(vecs[1])
    distances,matches = nearestNeighbors.kneighbors(vecs[0])

    matchPairs = [(i,j) for i,query_matches in enumerate(matches) for j in query_matches]
    pairDistances = np.array([d for query_distances in distances for d in query_distances])

    #Build results table
    stringPairs = [(queryStrings[i],candidateStrings[j]) for i,j in matchPairs]
    resultsDF = pd.DataFrame(stringPairs,columns=['query_string','candidate_string'])
    resultsDF['score'] = np.exp(-pairDistances**2)

    return resultsDF



def getTrainingDataFromPositiveMatches(positiveMatchesDF,modelPackage,sample=1000,take=50):

    #Evaluate model on matches
    sampleDF = positiveMatchesDF.sample(sample)
    matchesDF = findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage)
    matchesDF = pd.merge(matchesDF,positiveMatchesDF,how='left',on=['query_string','candidate_string'],indicator=True)
    matchesDF['match'] = (matchesDF['_merge'] == 'both').astype(float)
    matchesDF['abs_error'] = np.abs(matchesDF['match_prob'] - matchesDF['match'])

    #Make new training set from most incorrect results
    newTrainingDF = matchesDF.sort_values('abs_error').tail(take)
    newTrainingDF = newTrainingDF[['query_string','candidate_string','match']]

    return newTrainingDF,matchesDF


if __name__ == '__main__':

    #Train a base model

    namaDir = r'C:\Users\Brad\Google Drive\Research\Python3\nama'
    trainingDir = os.path.join(namaDir,'trainingData')
    modelDir = os.path.join(namaDir,'trainedModels')


    trainingFiles = [os.path.join(trainingDir,f) for f in os.listdir(trainingDir) if f.endswith('.csv')]
    trainingDF = pd.concat([pd.read_csv(f,encoding='utf8') for f in trainingFiles])

    trainingDF = trainingDF.sample(1000)

    modelPackage = newModel(cuda=True,d=500,lr=0.001,weight_decay=1e-8,recurrent_layers=1,bidirectional=True)

    #Repeat as desired:
    historyDF = trainModel(modelPackage,trainingDF,epochs=10,minibatch_size=20,resample_prob=0.1,verbose=True)
    plotLossHistory(historyDF,ylim=(0,0.5))

    #Set learning rate
    for g in modelPackage['optimizer'].param_groups:
        g['lr'] = 0.0001


    sampleDF = trainingDF.sample(100)

    matchesDF = findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage)

    matchesDF = pd.merge(matchesDF,sampleDF,'left',on=['query_string','candidate_string'])
    matchesDF['match'] = matchesDF['match'].fillna(0)

    sampleDF['match'].mean()
    matchesDF['match'].mean()

    matchesDF.groupby('match')['score'].plot.kde()

    matchesDF.sample(20).sort_values('score',ascending=False)


#
# packed,selector,match = next(generateMinibatches(trainingDF,cuda=True,base_size=2,random_samples=10))
#
#
# # W = packed
# # self = vectorModel(recurrent_layers=3,bidirectional=True).cuda()
# # h.size()
