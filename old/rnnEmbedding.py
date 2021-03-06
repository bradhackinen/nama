import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from sklearn.neighbors import NearestNeighbors
import inspect
import matplotlib.pyplot as plt

from nama.utilities import *
from nama.defaults import *


def stringToChars(s,max_len=200):
    return b'<' + stringToAscii(s)[:max_len] + b'>'


def bytesToIds(b,clamp_range=None):
    byteIds = torch.from_numpy(np.frombuffer(b,dtype=np.uint8).copy())
    if clamp_range is not None:
        torch.clamp(byteIds,min=clamp_range[0],max=clamp_range[1],out=byteIds)
        byteIds = byteIds - clamp_range[0]

    return byteIds


def idsTo1Hot(x,max_id=127):
    w = torch.zeros(len(x),max_id+1)
    w.scatter_(1,x.long().view(-1,1),1.0)
    return w


def packedIdsTo1Hot(packedIds,max_id=127):
    packed1hot = idsTo1Hot(packedIds.data.data,max_id=max_id)
    return PackedSequence(packed1hot,packedIds.batch_sizes)


def bytesToPacked1Hot(byteStrings,clamp_range=None,presorted=False):
    if clamp_range is not None:
        max_id = clamp_range[1]-clamp_range[0]
    else:
        max_id = 127

    if not presorted:
        byteStrings = sorted(set(byteStrings),key=len,reverse=True)

    if not byteStrings[-1]:
        raise Exception('Cannot pack empty bytestring')

    ids = [bytesToIds(s,clamp_range=clamp_range) for s in byteStrings]
    packed = nn.utils.rnn.pack_sequence(ids)
    packed1hot = packedIdsTo1Hot(packed,max_id=max_id)

    return packed1hot,byteStrings


def packedTo(packedSequence,device):
    return PackedSequence(packedSequence.data.to(device),packedSequence.batch_sizes)


def charIdsToString(charIds):
    '''Only intended for debugging'''
    chars = ''.join(chr(i+32) for i in charIds)
    return chars



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



class RnnEmbeddingModel():
    def __init__(self,device='cpu',d=300,d_recurrent=None,recurrent_layers=1,bidirectional=False):
        self.args = locals()
        del self.args['self']

        if d_recurrent is None:
            self.d_recurrent = d
        else:
            self.d_recurrent = d_recurrent

        self.model = VectorModel(d_recurrent=d,d_out=d,recurrent_layers=recurrent_layers,bidirectional=bidirectional)

        self.model.to(torch.device(device))

        self.params = {'params':self.model.parameters()}

        self.optimizer = torch.optim.Adam([self.params])

        self.lossHistory = pd.DataFrame()


    def trainMinibatch(self,components,componentWeights,batch_size,within_weight=1):
        device = next(self.model.parameters()).device

        if not batch_size <= 0.5*len(components):
            raise Exception('Batch size must be smaller than half the number of components')

        h = {'batch_size':batch_size,'between_loss':0,'within_loss':0,'within_size':0}

        # Make a little utility vector that contains 0 and 1 for entering into the loss function
        target = torch.tensor([0,1]).float().to(device)

        self.optimizer.zero_grad()
        for pair in np.random.choice(components,size=(batch_size,2),p=componentWeights,replace=False):
            pairChars = [set(stringToChars(s) for s in component) for component in pair]

            with torch.no_grad():
                pairData = [bytesToPacked1Hot(chars,clamp_range=(31,126))[0] for chars in pairChars]

            pairData = [packedTo(packed,device) for packed in pairData]

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

                    withinLoss = within_weight*F.binary_cross_entropy(score,target[1]) / 2*float(batch_size)
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



    def train(self,matcher,epochs=10,epoch_size=100,minibatch_size=1,lr=0.001,weight_decay=1e-6,within_weight=1,save_as=None,verbose=True):
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

        bar_freq = epoch_size//20 + 1

        startingEpoch = self.lossHistory['epoch'].max() + 1 if len(self.lossHistory) else 0
        for i,epoch in enumerate(range(startingEpoch,startingEpoch + epochs)):
            if verbose: print('\nTraining epoch {}'.format(epoch))

            for g in self.optimizer.param_groups:
                g['lr'] = lr_schedule[i]
                g['weight_decay'] = weight_decay


            epochHistory = []
            for b in range(epoch_size):
                h = self.trainMinibatch(components,componentWeights,b_schedule[i],within_weight=within_weight)
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



    def vectorizeStrings(self,strings,batch_size=100,max_len=100):
        self.model.eval()
        device = next(self.model.parameters()).device

        chars = [stringToChars(s,max_len=max_len) for s in strings]
        vecs = []
        for i in range(0,len(chars),batch_size):
            batchChars = chars[i:i+batch_size]
            packed,sorted_chars = bytesToPacked1Hot(batchChars,clamp_range=(31,126))
            chars_to_id = {s:i for i,s in enumerate(sorted_chars)}
            packed = packedTo(packed,device)

            sorted_vecs = self.model(packed).data.cpu().numpy()
            selector = np.array([chars_to_id[s] for s in batchChars])
            batch_vecs = sorted_vecs[selector,:]

            vecs.append(batch_vecs)

        return np.vstack(vecs)


    def findSimilar(self,strings,min_score=0,n=10,batch_size=100,leaf_size=50,drop_duplicates=True):
        strings = sorted(set(strings))
        n = min(n,len(strings))

        vecs = self.vectorizeStrings(strings,batch_size=batch_size)

        nearestNeighbors = NearestNeighbors(n_neighbors=n,algorithm='ball_tree',leaf_size=leaf_size)
        nearestNeighbors.fit(vecs)

        distances,matches = nearestNeighbors.kneighbors(vecs)

        matchPairs = np.vstack([np.kron(np.arange(len(strings)),np.ones(n).astype(int)),matches.ravel()]).T

        if drop_duplicates:
            matchPairs = np.sort(np.array(matchPairs),axis=1)

        matchScores = np.exp(-distances.ravel()**2)


        matchDF = pd.DataFrame(matchPairs,columns=['string0','string1'])
        matchDF['score'] = matchScores

        matchDF = matchDF[matchDF['string0'] != matchDF['string1']].copy()

        if min_score > 0:
            matchDF = matchDF[matchDF['score'] >= min_score]

        if drop_duplicates:
            matchDF = matchDF.drop_duplicates(['string0','string1'])

        for c in 'string0','string1':
            matchDF[c] = matchDF[c].apply(lambda i: strings[i])

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


def loadRnnEmbeddingModel(filename,device='cpu'):
    state = torch.load(filename,map_location=torch.device(device))

    # For backwards compatibility, limit to current valid args
    validArgs = inspect.getfullargspec(RnnEmbeddingModel.__init__).args
    validArgs.remove('self')

    invalidArgs = {arg for arg in state['args'].keys() if arg not in validArgs}
    if invalidArgs:
        print('Warning: Loaded model includes invalid initalization args {}. They will be ignored.'.format(invalidArgs))

    state['args'] = {arg:value for arg,value in state['args'].items() if arg in validArgs}
    state['args']['device'] = device

    rnnEmbeddingModel = RnnEmbeddingModel(**state['args'])
    rnnEmbeddingModel.model.load_state_dict(state['model_state'])
    rnnEmbeddingModel.optimizer.load_state_dict(state['optimizer_state'])

    rnnEmbeddingModel.lossHistory = state['loss_history']

    return rnnEmbeddingModel



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
    rnnEmbeddingModel = rnnEmbeddingModel(device='cuda',d=100,d_recurrent=100,recurrent_layers=2,bidirectional=True)

    matcher.suggestMatches(rnnEmbeddingModel,min_score=0)


    profile.run('rnnEmbeddingModel.train(matcher,epochs=1,within_weight=0.1)',sort='tottime')

    profile.run('matcher.suggestMatches(rnnEmbeddingModel,min_score=0)',sort='tottime')



    df0 = matcher.suggestMatches(rnnEmbeddingModel,min_score=0,leaf_size=2)


    df0 = matcher.suggestMatches(rnnEmbeddingModel,min_score=0)
    df1 = matcher.suggestMatches(rnnEmbeddingModel,min_score=0,neighbor_batch_size=2)
    df2 = matcher.suggestMatches(rnnEmbeddingModel,min_score=0,neighbor_batch_size=3)


    # Initialize the matcher
    from nama.defaults import *
    trainingDF = pd.read_csv(os.path.join(trainingDir,'lobbyingClients_training.csv'))

    samplePairsDF = trainingDF.sample(100)
    sampleSinglesDF = trainingDF.sample(10000)
    strings = set(samplePairsDF[['candidate_string','query_string']].values.ravel()) \
            | set(sampleSinglesDF['query_string'])
    matcher = Matcher(strings)

    # Add some corpHash matches
    matcher.matchHash(nama.hashes.corpHash)

    matcher.simplify()

    matcher.matchesDF()

    rnnEmbeddingModel = rnnEmbeddingModel(device='cuda',d=20,d_recurrent=20,recurrent_layers=2,bidirectional=True)

    profile.run('rnnEmbeddingModel.train(matcher,epochs=1)',sort='tottime')


    profile.run('matcher.suggestMatches(rnnEmbeddingModel,min_score=0)',sort='tottime')


    df = matcher.suggestMatches(rnnEmbeddingModel,n=5)

    resultsDF = pd.DataFrame()
    for minibatch_size in 1,10:
        for lr in 1e-6,1e-7:
            print('\n\nminibatch_size={}'.format(minibatch_size))
            rnnEmbeddingModel = rnnEmbeddingModel(device='cuda',d=20,d_recurrent=20,recurrent_layers=2,bidirectional=True)

            # profile.run('rnnEmbeddingModel.train(matcher,epochs=10,epoch_size=1000//b,minibatch_size=b,lr=lr)',sort='tottime')
            historyDF = rnnEmbeddingModel.train(matcher,epochs=10,epoch_size=1000//minibatch_size,minibatch_size=minibatch_size,lr=lr)
            suggestedDF = matcher.suggestMatches(rnnEmbeddingModel,n=5,min_score=0)
            # print('Mean nearest neighbor score: {:0.3f}'.format(historyDF['score'].mean()))

            df = historyDF[['within_loss','between_loss','within_size']].tail(1000).mean().to_frame().T
            df['minibatch_size'] = minibatch_size
            df['lr'] = lr
            df['mean_nn_score'] = suggestedDF['score'].mean()

            resultsDF = resultsDF.append(df)


    # Test saving and loading across devices

    for device in ['cpu','cuda']:

        model = rnnEmbeddingModel(device=device)
        model.save(os.path.join(modelDir,'temp.bin'))

        loadRnnEmbeddingModel(os.path.join(modelDir,'temp.bin'))
        loadRnnEmbeddingModel(os.path.join(modelDir,'temp.bin'),device='cuda')

    # test compatibility with models saved from older version
    loadRnnEmbeddingModel(os.path.join(modelDir,'grantOrgsrnnEmbeddingModel.003.bin'))
