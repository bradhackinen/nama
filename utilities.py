import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import math
from unidecode import unidecode



def stringToAscii(s):
    s = unidecode(s)
    s = s.encode('ascii')
    return s


def bytesToIds(b,clamp_range=None):
    '''
    For
    '''
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


def packedToCuda(packedSequence):
    return PackedSequence(packedSequence.data.cuda(),packedSequence.batch_sizes)


def charIdsToString(charIds):
    '''Only intended for debugging'''
    chars = ''.join(chr(i+32) for i in charIds)
    return chars


def trainWithHistory(batch_train_fuction,epoch_batch_iterator,historyDF=None,exit_function=None,verbose=True,bar_freq=1):
    '''
    Tracks training history for a training function exucuted over a nested
    epoch-batch iterator
    '''

    if historyDF is None:
        historyDF = pd.DataFrame()

    try:
        start_epoch = historyDF['epoch'].max() + 1
    except:
        start_epoch = 0

    for epoch,batch_iterator in enumerate(epoch_batch_iterator,start_epoch):
        epochHistory = []
        if verbose: print('\nTraining epoch {}'.format(epoch))
        for batch,batchData in enumerate(batch_iterator):

            batchHistory = batch_train_fuction(batchData)

            batchHistory['batch'] = batch
            epochHistory.append(batchHistory)

            if verbose and not (batch + 1) % bar_freq: print('|',end='')

        epochHistoryDF = pd.DataFrame(epochHistory)

        epochHistoryDF['epoch'] = epoch

        historyDF = historyDF.append(epochHistoryDF)

        if verbose: print('\nMean loss: {}'.format(epochHistoryDF['loss'].sum()/epochHistoryDF['size'].sum()))

        if not exit_function is None:
            if exit_function(historyDF):
                break

    return historyDF



def plotLossHistory(historyDF,ylim=None,scatter_alpha=0.25,max_points=10000):
    historyDF = historyDF.copy()
    historyDF['x'] = historyDF['size'].cumsum()
    if len(historyDF) > max_points:
        historyDF = historyDF.sample(max_points).sort_values('x')
    bw = len(historyDF)/20
    historyDF['smoothed_loss'] = (historyDF['loss']/historyDF['size']).rolling(window=math.ceil(bw*4),center=True,min_periods=1,win_type='gaussian').mean(std=bw)

    plt.scatter(x=historyDF['x'],y=historyDF['loss']/historyDF['size'],s=3,alpha=scatter_alpha)
    plt.plot(historyDF['x'],historyDF['smoothed_loss'])
    if ylim:
        plt.gca().set_ylim(*ylim)
    plt.show()





def dfChunks(df,chunk_size):
    for i in range(0,len(df),chunk_size):
        yield df[i:i+chunk_size]
