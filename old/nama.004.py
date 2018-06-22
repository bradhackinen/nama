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
import networkx as nx
import regex as re


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
        trainingDF['{}_chars'.format(c)] = trainingDF['{}_string'.format(c)].apply(lambda s: b'<' stringToChars(s)[:max_len] + b'>')

    #Build componentMap
    componentMap = {}
    for c in 'query','candidate':
        componentMap.update({c:i for c,i in zip(trainingDF['{}_chars'.format(c)],trainingDF['component'])})


    for minibatchDF in dfChunks(trainingDF,base_size):
        batchPairs = {(c0,c1) for c0,c1 in zip(minibatchDF['query_chars'],minibatchDF['candidate_chars'])}
        batchPairs.update({(c1,c0) for c0,c1 in batchPairs})

        with torch.no_grad():
            packed,chars = charIdsToPacked1Hot((s for c in ('query','candidate') for s in minibatchDF['{}_chars'.format(c)]))

            selector = np.vstack(np.triu_indices(len(chars),k=1)).T

            inBatch = np.array([(chars[i],chars[j]) in batchPairs for i,j in selector])
            randPairs = np.random.uniform(0,1,size=selector.shape[0]) < resample_prob

            selector = selector[inBatch | randPairs,:]

            match = np.array([componentMap[chars[i]] == componentMap[chars[j]] for i,j in selector]).astype(float)

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

    match_prob =(-dist).exp()
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


def connectedStrings(strings,correctPairs=None,hash_function=None):
    if (correctPairs is None) and (hash_function is None):
        raise Exception('Must provide trainingDF or hash function')

    G = nx.Graph()
    G.add_nodes_from(strings)

    #Connect correct pairs
    if correctPairs:
        G.add_edges_from(correctPairs)

    #Connect strings with the same hash value (if hash function provided)
    if hash_function is not None:
        for s in strings:
            G.add_edge(s,hash_function(s))

    components = nx.connected_components(G)
    componentMap = {s:i for i,component in enumerate(components) for s in component}

    return componentMap


def trainModel(modelPackage,trainingDF,save_as=None,hash_function=None,minibatch_size=10,resample_ratio=5,epochs=10,lr=0.001,exit_function=None,verbose=True):

    trainingDF = trainingDF.copy()

    allStrings = set(trainingDF['query_string']) | set(trainingDF['candidate_string'])

    correctPairs = {(s0,s1) for i,s0,s1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m >= 0.5}
    correctPairs.update({(s1,s0) for s0,s1 in correctPairs})

    componentMap = connectedStrings(allStrings,correctPairs,hash_function)

    trainingDF['component'] = trainingDF['query_string'].apply(lambda s: componentMap[s])

    cuda = next(modelPackage['model'].parameters()).is_cuda

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

    resample_prob = resample_ratio / (b_schedule - 1)
    if resample_prob.max() > 1:
        raise Exception('Resample ratio not feasible for minibatch size {}'.format(b_schedule[np.argmax(resample_prob)]))

    # Train epochs
    for i in range(epochs):
        for g in modelPackage['optimizer'].param_groups:
            g['lr'] = lr_schedule[i]
        bar_freq = max(1,int((len(trainingDF)/b_schedule[i])/100))

        epochBatchIterator = (generateMinibatches(trainingDF,cuda=cuda,base_size=b_schedule[i],resample_prob=resample_prob[i]),)
        modelPackage['loss_history'] = trainWithHistory(lambda b: trainMinibatch(b,modelPackage),epochBatchIterator,modelPackage['loss_history'],
                                        exit_function=exit_function,verbose=verbose,bar_freq=bar_freq)

        if save_as is not None:
            if verbose: print('Saving model as {}'.format(save_as))
            saveModelPackage(modelPackage, os.path.join(modelDir,save_as))


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


def findDirectMatches(queryStrings,candidateStrings,trainingDF=None,hash_function=None):
    '''
    Finds strings that are matched because they are in the same conected component,
    where edges are defined by trainingDF pairs or a shared hash value.
    Optional hash function takes a string and produces a hash value.
    '''
    queryDF = pd.DataFrame(list(queryStrings),columns=['string'])
    candidateDF = pd.DataFrame(list(candidateStrings),columns=['string'])

    if trainingDF is not None:
        correctPairs = {(s0,s1) for i,s0,s1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m >= 0.5}
        correctPairs.update({(s1,s0) for s0,s1 in correctPairs})
    else:
        correctPairs = {}

    allStrings = set(queryStrings) | set(candidateStrings)
    componentMap = connectedStrings(allStrings,correctPairs,hash_function)

    for df in queryDF,candidateDF:
        df['component'] = df['string'].apply(lambda s: componentMap[s])

    queryDF = queryDF.rename(columns={'string':'query_string'})
    candidateDF = candidateDF.rename(columns={'string':'candidate_string'})

    matchesDF = pd.merge(queryDF,candidateDF,on='component')

    return matchesDF


# queryStrings,candidateStrings = sampleDF['query_string'],sampleDF['candidate_string']
def findFuzzyMatches(queryStrings,candidateStrings,modelPackage,incorrectPairs={},max_attempts=10):
    queryStrings = sorted(set(queryStrings))
    candidateStrings = sorted(set(candidateStrings))

    # Compute vector representations of strings
    vecs = [vectorizeStrings(strings,modelPackage) for strings in (queryStrings,candidateStrings)]

    # Use sklearn to efficiently find nearest neighbours
    nearestNeighbors = NearestNeighbors(n_neighbors=1,metric='euclidean')
    nearestNeighbors.fit(vecs[1])

    # Search for best match while discarding incorrect pairs
    # Re-try search with more neighbors for incorrect pairs
    matchesDF = pd.DataFrame()
    for attempt in range(max_attempts):

        distances,matches = nearestNeighbors.kneighbors(vecs[0],n_neighbors=1 + attempt)

        matchPairs = [(i,j) for i,query_matches in enumerate(matches) for j in query_matches]
        pairDistances = np.array([d for query_distances in distances for d in query_distances])

        #Build results table
        stringPairs = [(queryStrings[i],candidateStrings[j]) for i,j in matchPairs]

        df = pd.DataFrame(stringPairs,columns=['query_string','candidate_string'])
        df['score'] = np.exp(-pairDistances)
        df['attempt'] = attempt

        df = df[[(pair not in incorrectPairs) for pair in stringPairs]]

        matchesDF = matchesDF.append(df)

        # Reduce query vectors and strings to only the unmatched strings
        matched = set(matchesDF['query_string'])
        vecs[0] = vecs[0][[s not in matched for s in queryStrings],:]
        queryStrings = [s for s in queryStrings if s not in matched]

        if not queryStrings:
            break

    else:
        print('Warning: Reached max attempts ({}) while looking for a non-incorrect match'.format(max_attempts))

    return matchesDF#.drop('in_incorrect',axis=1)


def findMatches(queryStrings,candidateStrings,modelPackage,trainingDF=None,hash_function=None,n_matches=1):
    #Look for direct matches
    if (trainingDF is not None) or (hash_function is not None):
        matchesDF = findDirectMatches(queryStrings,candidateStrings,trainingDF,hash_function)
        matchesDF = matchesDF.drop('component',axis=1)
        matchesDF['score'] = 1

        matchedQueries = set(matchesDF['query_string'])
        queryStrings = [s for s in queryStrings if s not in matchedQueries]
    else:
        matchesDF = pd.DataFrame()
        queryStrings = list(queryStrings)

    #Look for fuzzy matches for remaining strings
    if len(queryStrings):
        if trainingDF is not None:
            incorrectPairs = {(c0,c1) for i,c0,c1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m < 0.5}
            incorrectPairs.update({(c1,c0) for c0,c1 in incorrectPairs})
        else:
            incorrectPairs = {}

        fuzzyMatchesDF = findFuzzyMatches(queryStrings,candidateStrings,modelPackage,incorrectPairs=incorrectPairs)

        matchesDF = matchesDF.append(fuzzyMatchesDF.drop('attempt',axis=1))

    return matchesDF.reset_index(drop=True).drop_duplicates()


def basicHash(s):
    '''
    A simple case and puctuation-insentive hash
    '''
    s = s.lower()
    s = re.sub(' & ',' and ',s)
    s = re.sub(r'[\s\.,:;/\'"\(\)]+',' ',s)
    s = s.strip()

    return s


def corpHash(s):
    '''
    Insensitive to case, puctation, and common corporation suffixes
    '''
    s = basicHash(s)
    s = re.sub('\s+(inc|ltd|ll[cp])$','',s)
    return s



if __name__ == '__main__':

    #Train a base model

    namaDir = r'C:\Users\Brad\Google Drive\Research\Python3\nama'
    trainingDir = os.path.join(namaDir,'trainingData')
    modelDir = os.path.join(namaDir,'trainedModels')

    trainingFiles = [os.path.join(trainingDir,f) for f in os.listdir(trainingDir) if f.endswith('.csv')]
    trainingDF = pd.concat([pd.read_csv(f,encoding='utf8') for f in trainingFiles])

    # trainingDF = trainingDF.sample(1000)

    modelPackage = newModel(cuda=True,d=750,weight_decay=1e-6,recurrent_layers=1,bidirectional=True)

    # modelPackage = loadModelPackage(os.path.join(modelDir,'allTrainingData_bi_d500.004.bin'))

    #Repeat as desired:
    historyDF = trainModel(modelPackage,trainingDF,hash_function=corpHash,epochs=20,minibatch_size=(10,200),resample_ratio=5,
                            save_as=os.path.join(modelDir,'allTrainingData_bi_d7500.001.bin'))
    plotLossHistory(historyDF)


    # plotLossHistory(modelPackage['loss_history'])

    # #Test matching code
    sampleDF = trainingDF[trainingDF['match'].isin({0,1})].sample(1000)
    #
    # findDirectMatches(sampleDF['query_string'],sampleDF['candidate_string'],trainingDF=sampleDF)
    # findDirectMatches(sampleDF['query_string'],sampleDF['candidate_string'],hash_function=basicHash)
    # findDirectMatches(sampleDF['query_string'],sampleDF['candidate_string'],trainingDF=sampleDF,hash_function=basicHash)
    #
    matchesDF = findFuzzyMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage)
    # matchesDF = findFuzzyMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage,
    #                 incorrectPairs = {('AXNES Aviation','Axnes Aviation AS'),('AXNES Aviation','YPF SA')})
    #
    # findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage)
    # findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage,trainingDF=sampleDF.sample(frac=0.05))
    # findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage,hash_function=basicHash)
    # findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage,trainingDF=sampleDF.sample(frac=0.05),hash_function=basicHash)
    #
    #
    matchesDF = pd.merge(matchesDF,sampleDF,'left',on=['query_string','candidate_string'])
    matchesDF['match'] = matchesDF['match'].fillna(0)

    matchesDF['match'].value_counts()
    matchesDF['match'].mean()
    matchesDF.groupby('match')['score'].mean()

    matchesDF.groupby('match')['score'].plot.kde()

    matchesDF.sample(20).sort_values('score',ascending=False)


# #
# # packed,selector,match = next(generateMinibatches(trainingDF,cuda=True,base_size=2,random_samples=10))
# #
# #
# # # W = packed
# # # self = vectorModel(recurrent_layers=3,bidirectional=True).cuda()
# # # h.size()
#
# trainingDF[trainingDF['query_string']=='Raytheon Company']
#
# componentMap['the boeing company']
#
# H = nx.Graph()
#
# H.add_edges_from([(s0,s1) for i,s0,s1 in trainingDF[trainingDF['component']==2457][['query_string','candidate_string']].itertuples()])
#
# nx.draw(H,with_labels=True,font_size=6)
#
# H['THE BOEING COMPANY']
#
# pd.DataFrame([(s0,s1,b) for (s0,s1),b in nx.edge_betweenness_centrality(H).items()],columns=['s0','s1','b']).sort_values('b',ascending=False)
