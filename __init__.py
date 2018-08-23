import os
import pandas as pd
import numpy as np
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


def generateMinibatches(trainingDF,cuda=False,max_len=100,base_size=10,resample_prob=0.5,shuffle=True):
    trainingDF = trainingDF[trainingDF['match'].notnull()].copy()

    if shuffle:
        trainingDF = trainingDF.sample(frac=1)

    for c in 'query','candidate':
        trainingDF['{}_chars'.format(c)] = trainingDF['{}_string'.format(c)].apply(stringToChars)

    #Build componentMap
    componentMap = {}
    for c in 'query','candidate':
        componentMap.update({c:i for c,i in zip(trainingDF['{}_chars'.format(c)],trainingDF['component'])})


    for minibatchDF in dfChunks(trainingDF,base_size):
        batchPairs = {(c0,c1) for c0,c1 in zip(minibatchDF['query_chars'],minibatchDF['candidate_chars'])}
        batchPairs.update({(c1,c0) for c0,c1 in batchPairs})

        with torch.no_grad():
            packed,chars = bytesToPacked1Hot((s for c in ('query','candidate') for s in minibatchDF['{}_chars'.format(c)]),clamp_range=(31,126))

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

    score =(-dist).exp()
    score = torch.clamp(score,min=0,max=1)

    loss = F.binary_cross_entropy(score,match,size_average=False)

    modelPackage['optimizer'].zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(modelPackage['model'].parameters(),5)

    modelPackage['optimizer'].step()

    return {'size':match.size(0),'loss':float(loss.data.cpu())}


class vectorModel(nn.Module):
    def __init__(self,d_in=96,d_recurrent=300,d_out=300,recurrent_layers=1,bidirectional=False):
        super().__init__()

        d_gru_out = d_recurrent*recurrent_layers*(1+int(bidirectional))

        self.char_embedding = nn.Linear(d_in,d_recurrent)
        self.embedding_dropout = nn.Dropout()
        self.gru = nn.GRU(d_recurrent,d_recurrent,recurrent_layers,bidirectional=bidirectional,batch_first=True,dropout=0.5 if recurrent_layers > 1 else 0)
        self.dropout = nn.Dropout()
        self.projection = nn.Linear(d_gru_out,d_out)
        # self.dropout2 = nn.Dropout()
        # self.projection2 = nn.Linear(d_gru_out,d_gru_out)

    def forward(self,W):
        X = PackedSequence(self.embedding_dropout(self.char_embedding(W.data)),W.batch_sizes)
        H,h = self.gru(X)
        #Concat layer and direction h vectors
        v = h.permute(1,0,2).contiguous().view(h.shape[1],-1)
        v = self.dropout(v)
        v = self.projection(v)
        # v = self.dropout2(v)
        # v = self.projection2(F.tanh(v)) + v
        return v


def newModel(cuda=False,d=300,d_recurrent=None,recurrent_layers=1,bidirectional=False,lr=1e-3,weight_decay=1e-6):
    modelPackage = {'args':locals(),'loss_history':pd.DataFrame()}
    if d_recurrent is None:
        d_recurrent = d

    modelPackage['model'] = vectorModel(d_recurrent=d,d_out=d,recurrent_layers=recurrent_layers,bidirectional=bidirectional)

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


def connectedStrings(strings,correct_pairs=None,incorrect_pairs=None,hash_function=None):
    if (correct_pairs is None) and (hash_function is None):
        raise Exception('Must provide correct pairs or hash function')

    G = nx.Graph()
    G.add_nodes_from(strings)

    # Connect correct pairs
    if correct_pairs:
        G.add_edges_from(correct_pairs)

    # Connect strings with the same hash value (if hash function provided)
    if hash_function is not None:
        for s in strings:
            G.add_edge(s,hash_function(s))

    # Disconnect any pairs that are forced to be 'incorrect'
    if incorrect_pairs:
        G.remove_edges_from(incorrect_pairs)

    components = nx.connected_components(G)
    componentMap = {s:i for i,component in enumerate(components) for s in component}

    return componentMap


def trainModel(modelPackage,trainingDF,testDF=None,save_as=None,hash_function=None,minibatch_size=10,resample_ratio=5,epochs=10,lr=0.001,exit_function=None,verbose=True):

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

        if verbose and testDF is not None:
            testResults = scoreTestDF(modelPackage,testDF,hash_function=hash_function)
            print('Test loss: {:.3f} (Best accuracy of {:.3f} with threshold {:.3f})'.format(testResults['matches']['loss'].mean(),testResults['optimal_accuracy'],testResults['optimal_threshold']))

        if save_as is not None:
            if verbose: print('Saving model as {}'.format(save_as))
            saveModelPackage(modelPackage, os.path.join(modelDir,save_as))

    return modelPackage['loss_history']



def vectorizeStrings(strings,modelPackage,batch_size=100,max_len=200):
    modelPackage['model'].eval()

    chars = [stringToChars(s,max_len=max_len) for s in strings]
    vecs = []
    for i in range(0,len(chars),batch_size):
        batchChars = chars[i:i+batch_size]
        packed,sorted_chars = bytesToPacked1Hot(batchChars,clamp_range=(31,126))
        chars_to_id = {s:i for i,s in enumerate(sorted_chars)}
        if next(modelPackage['model'].parameters()).is_cuda:
            packed = packedToCuda(packed)
        sorted_vecs = modelPackage['model'](packed).data.cpu().numpy()
        selector = np.array([chars_to_id[s] for s in batchChars])
        batch_vecs = sorted_vecs[selector,:]
        vecs.append(batch_vecs)

    return np.vstack(vecs)


def scorePairs(stringPairs,modelPackage,batch_size=100):
    vecs = [torch.from_numpy(vectorizeStrings(strings,modelPackage,batch_size=batch_size)) for strings in zip(*stringPairs)]

    dist = F.pairwise_distance(vecs[0],vecs[1])

    score =(-dist).exp()
    score = torch.clamp(score,min=0,max=1)

    return score.numpy()



def findDirectMatches(query_strings,candidate_strings,trainingDF=None,hash_function=None,min_score=0.5):
    '''
    Finds strings that are matched because they are in the same conected component,
    where edges are defined by trainingDF pairs or a shared hash value.
    Optional hash function takes a string and produces a hash value.
    '''
    query_strings = list(set(query_strings))
    candidate_strings = list(set(candidate_strings))

    queryDF = pd.DataFrame(query_strings,columns=['string'])
    candidateDF = pd.DataFrame(candidate_strings,columns=['string'])

    if trainingDF is not None:
        correctPairs = {(s0,s1) for i,s0,s1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m >= min_score}
        incorrectPairs = {(s0,s1) for i,s0,s1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m < min_score}
    else:
        correctPairs = set()
        incorrectPairs = set()

    allStrings = set(query_strings) | set(candidate_strings)
    componentMap = connectedStrings(allStrings,correct_pairs=correctPairs,incorrect_pairs=incorrectPairs,hash_function=hash_function)

    for df in queryDF,candidateDF:
        df['component'] = df['string'].apply(lambda s: componentMap[s])

    queryDF = queryDF.rename(columns={'string':'query_string'})
    candidateDF = candidateDF.rename(columns={'string':'candidate_string'})

    matchesDF = pd.merge(queryDF,candidateDF,on='component')

    return matchesDF



def findFuzzyMatches(query_strings,candidate_strings,modelPackage,incorrect_pairs={},best_only=True,min_score=0.95,max_attempts=10,batch_size=100):
    query_strings = set(query_strings)
    candidate_strings = set(candidate_strings)

    strings = list(query_strings | candidate_strings)
    string_ids = {s:i for i,s in enumerate(strings)}

    query_strings = list(query_strings)
    candidate_strings = list(candidate_strings)

    query_selector = np.array([string_ids[s] for s in query_strings])
    candidate_selector = np.array([string_ids[s] for s in candidate_strings])

    # Compute vector representations of strings
    vecs = vectorizeStrings(strings,modelPackage,batch_size=batch_size)

    # Search for best match while discarding incorrect pairs
    # Re-try search with more neighbors for incorrect pairs
    matchesDF = pd.DataFrame()
    for attempt in range(max_attempts):
        # Use sklearn to efficiently find nearest neighbours
        nearestNeighbors = NearestNeighbors(n_neighbors=1,radius=-np.log(max(min_score,1e-8)),metric='euclidean')
        nearestNeighbors.fit(vecs[candidate_selector])

        if best_only:
            distances,matches = nearestNeighbors.kneighbors(vecs[query_selector],n_neighbors=1 + attempt)
        else:
            distances,matches = nearestNeighbors.radius_neighbors(vecs[query_selector])

        matchPairs = [(i,j) for i,query_matches in enumerate(matches) for j in query_matches]
        pairDistances = np.array([d for query_distances in distances for d in query_distances])

        #Build results table
        stringPairs = [(query_strings[i],candidate_strings[j]) for i,j in matchPairs]

        df = pd.DataFrame(stringPairs,columns=['query_string','candidate_string'])
        df['score'] = np.exp(-pairDistances)
        df['attempt'] = attempt

        df = df[[(pair not in incorrect_pairs) for pair in stringPairs]]

        matchesDF = matchesDF.append(df)

        if not best_only:
            break

        # Reduce query vectors and strings to only the unmatched strings
        query_strings = sorted(set(query_strings) - set(matchesDF['query_string']))
        query_selector = np.array([string_ids[s] for s in query_strings])

        if not len(query_selector):
            break

    else:
        print('Warning: Reached max attempts ({}) while looking for a non-incorrect match'.format(max_attempts))

    if min_score:
        matchesDF = matchesDF[matchesDF['score'] >= min_score].copy()

    return matchesDF



def findMatches(queryStrings,candidateStrings,modelPackage,trainingDF=None,hash_function=None,n_matches=1,batch_size=100,min_score=0.5):
    #Look for direct matches
    if (trainingDF is not None) or (hash_function is not None):
        matchesDF = findDirectMatches(queryStrings,candidateStrings,trainingDF,hash_function,min_score=min_score)
        matchesDF = matchesDF.drop('component',axis=1)
        matchesDF['score'] = 1

        matchedQueries = set(matchesDF['query_string'])
        queryStrings = [s for s in queryStrings if s not in matchedQueries]
    else:
        matchesDF = pd.DataFrame()
        queryStrings = list(queryStrings)

    #Look for fuzzy matches for remaining strings
    if len(queryStrings) and min_score < 1:
        if trainingDF is not None:
            incorrectPairs = {(c0,c1) for i,c0,c1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m < 0.5}
            incorrectPairs.update({(c1,c0) for c0,c1 in incorrectPairs})
        else:
            incorrectPairs = {}

        fuzzyMatchesDF = findFuzzyMatches(queryStrings,candidateStrings,modelPackage,incorrect_pairs=incorrectPairs,batch_size=batch_size,min_score=min_score)

        matchesDF = matchesDF.append(fuzzyMatchesDF.drop('attempt',axis=1))

    matchesDF = matchesDF.groupby('query_string').first().reset_index()
    matchesDF = matchesDF.sort_values('score',ascending=False)

    return matchesDF.reset_index(drop=True).drop_duplicates()



def scoreTestDF(modelPackage,testDF,hash_function=None):
    # Compute match scores (note that false negatives will not be included - need to correct for this later)
    matchesDF = findFuzzyMatches(testDF['query_string'],testDF['candidate_string'],modelPackage,min_score=0)

    # Compute correct match values (using hash function if provided)
    correctPairs = {(s0,s1) for i,s0,s1,m in testDF[['query_string','candidate_string','match']].itertuples() if m >= 0.5}
    correctPairs.update({(s1,s0) for s0,s1 in correctPairs})

    allStrings = set(testDF['query_string']) | set(testDF['candidate_string'])
    componentMap = connectedStrings(allStrings,correctPairs,hash_function)

    matchesDF['match'] = [float(componentMap[s0]==componentMap[s1]) for i,s0,s1 in matchesDF[['query_string','candidate_string']].itertuples()]

    #Compute test loss
    matchesDF['loss'] = F.binary_cross_entropy(torch.from_numpy(matchesDF['score'].values).float(),torch.from_numpy(matchesDF['match'].values).float(),reduce=False).numpy()

    # Compute accuracy as a function of score threshold
    accuracyDF = matchesDF[['match','score']].copy()
    accuracyDF = accuracyDF.sort_values('score')

    accuracyDF['true_positives'] = accuracyDF.loc[::-1,'match'].cumsum()[::-1]
    accuracyDF['true_negatives'] = (1 - accuracyDF['match']).cumsum()

    accuracyDF['correct'] = accuracyDF['true_positives'] + accuracyDF['true_negatives']
    accuracyDF['accuracy'] = accuracyDF['correct'] / len(testDF)

    optimalThreshold = accuracyDF['score'][accuracyDF['accuracy'].idxmax()]
    optimalAccuracy = accuracyDF['accuracy'].max()

    return {'matches':matchesDF,'accuracies':accuracyDF,'optimal_accuracy':optimalAccuracy,'optimal_threshold':optimalThreshold}



def findClusters(strings,modelPackage,min_score=0.99,trainingDF=None,hash_function=None,as_df=True):
    strings = sorted(set(strings))

    correctPairs = set()
    incorrectPairs = set()

    if trainingDF is not None:
        correctPairs.update({(s0,s1) for i,s0,s1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m >= min_score})
        incorrectPairs.update({(s0,s1) for i,s0,s1,m in trainingDF[['query_string','candidate_string','match']].itertuples() if m < min_score})

    if min_score < 1:
        matchesDF = findFuzzyMatches(strings,strings,modelPackage,best_only=False,min_score=min_score)
        correctPairs.update({(s0,s1) for i,s0,s1 in matchesDF[['query_string','candidate_string']].itertuples()})

    clusterMap = connectedStrings(strings,correct_pairs=correctPairs,incorrect_pairs=incorrectPairs,hash_function=hash_function)

    if as_df:
        df = pd.DataFrame(strings,columns=['string'])
        df['cluster'] = df['string'].apply(lambda s: clusterMap[s])
        return df
    else:
        return clusterMap



def merge(queryDF,candidateDF,modelPackage,how='inner',on=None,left_on=None,right_on=None,min_score=0.5,**search_args):
    if on is not None:
        left_on = on
        right_on = on

    if left_on is None or right_on is None:
        raise('Must provide column to merge on')

    matchesDF = findMatches(queryDF[left_on],candidateDF[right_on],modelPackage,**search_args)
    matchesDF = matchesDF[matchesDF['score']>min_score]

    matchesDF = pd.merge(queryDF,matchesDF,left_on=left_on,right_on='query_string',how=how)
    matchesDF = pd.merge(matchesDF,candidateDF,left_on='candidate_string',right_on=right_on,how=how)

    matchesDF = matchesDF.drop(['query_string','candidate_string'],axis=1)

    matchesDF = matchesDF.sort_values('score',ascending=False)

    return matchesDF



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
    A hash function for corporate subsidiaries
    Insensitive to
        -case & puctation
        -'the' prefix
        -common corporation suffixes, including 'holding co'
    '''
    s = basicHash(s)
    if s.startswith('the '):
        s = s[4:]
    s = re.sub('( (holding co|inc|ltd|ll?[cp]|co(rp)?|s[ae]|plc))+$','',s,count=1)

    return s



def loadTrainingData(filename,encoding='utf8',force_binary_match=True):
    df = pd.read_csv(filename,encoding=encoding)
    df = df[['query_string','candidate_string','match']]
    df = df[df['match'].notnull()]
    if force_binary_match:
        df['match'] = (df['match'] > 0.5).astype(float)

    return df





if __name__ == '__main__':

    #Train a base model

    namaDir = r'C:\Users\Brad\Google Drive\Research\Python3\nama'
    trainingDir = os.path.join(namaDir,'trainingData')
    modelDir = os.path.join(namaDir,'trainedModels')

    trainingFiles = [os.path.join(trainingDir,f) for f in os.listdir(trainingDir) if f.endswith('.csv')]
    trainingDF = pd.concat([loadTrainingData(f) for f in trainingFiles]).drop_duplicates()

    # trainingDF = trainingDF.sample(frac=1).reset_index(drop=True)

    testDF = trainingDF.sample(n=1000)
    trainDF = trainingDF[~trainingDF.index.get_level_values(0).isin(testDF.index.get_level_values(0))]

    # trainingDF = trainingDF.sample(1000)

    # modelPackage = newModel(cuda=True,d=300,d_recurrent=200,weight_decay=1e-6,recurrent_layers=3,bidirectional=True)

    modelPackage = loadModelPackage(os.path.join(modelDir,'allTrainingData_3bi200_to_300.003.bin'))

    #Repeat as desired:
    historyDF = trainModel(modelPackage,trainDF,testDF=testDF,hash_function=corpHash,epochs=5,minibatch_size=(11,50),resample_ratio=10,
                            save_as=os.path.join(modelDir,'allTrainingData_3bi200_to_300.003.bin'))
    plotLossHistory(historyDF)

    df = findFuzzyMatches(testDF['query_string'],testDF['candidate_string'],modelPackage,best_only=False,min_score=0.95)

    testResults = scoreTestDF(modelPackage,testDF,corpHash)

    testResults['matches'].groupby('match')['score'].plot.kde()

    testResults['matches'].sample(50).sort_values('loss',ascending=False)


    #Review training data
    review = scoreTestDF(modelPackage,trainingDF,corpHash)
    df = review['matches'].sort_values('loss',ascending=False).head(1000)
    df[['query_string','candidate_string','match']].to_csv(os.path.join(namaDir,'trainingData','corrections','corrections.001.csv'),index=False,encoding='mbcs')


    # # Test individual functions
    # df = findFuzzyMatches(testDF['query_string'],testDF['candidate_string'],modelPackage,best_only=False,min_score=0.5)
    # df = findDirectMatches(testDF['query_string'],testDF['candidate_string'],trainingDF=trainingDF,hash_function=corpHash,min_score=0.5)
    # df = findMatches(testDF['query_string'],testDF['candidate_string'],modelPackage,trainingDF=trainingDF,hash_function=corpHash,min_score=0.5)
    # df = findMatches(testDF['query_string'],testDF['candidate_string'],modelPackage,trainingDF=trainingDF,hash_function=corpHash,min_score=1)
    # df = findClusters(testDF['query_string'],modelPackage,trainingDF=trainingDF,hash_function=corpHash,min_score=0.5)
    #
    # import cProfile
    # cProfile.run("findClusters(trainingDF['query_string'].sample(10000),modelPackage,trainingDF=trainingDF,hash_function=corpHash,min_score=1)",sort='cumtime')
