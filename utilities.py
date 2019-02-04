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


def packedTo(packedSequence,device):
    return PackedSequence(packedSequence.data.to(device),packedSequence.batch_sizes)


def charIdsToString(charIds):
    '''Only intended for debugging'''
    chars = ''.join(chr(i+32) for i in charIds)
    return chars


def dfChunks(df,chunk_size):
    for i in range(0,len(df),chunk_size):
        yield df[i:i+chunk_size]
