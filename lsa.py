import os
import pandas as pd
import numpy as np
from sparsesvd import sparsesvd

from nama.utilities import *
from nama.defaults import *
from nama.tokenizers import *



class LSAModel():

    def __init__(self,strings,hasher=None,tokenizer=lambda s: ngrams(s,2),k=50,weighting='norm',no_below=1):
        self.bow = BOW()
        self.hasher = hasher
        self.tokenizer = tokenizer
        self.k = k
        self.weighting = weighting
        self.no_below = no_below

        C = self.bow.fit((self.tokenizer(string) for string in strings),no_below=self.no_below)

        if weighting:
            if weighting == 'norm':
                self.w = 1/np.sqrt(C.power(2).sum(axis=1))

            elif weighting == 'idf':
                self.w = 1/(np.log(1 + (C > 0).sum(axis=1)))

            elif weighting == 'entropy':

                P = C.multiply(1/C.sum(axis=1))
                S = P.copy()
                S.data = P.data*np.log(P.data)

                self.w = 1 + (S.sum(axis=1) / np.log(C.shape[1]))

            else:
                raise Exception('Unknown weighting type:',weighting)
        else:
            self.w = np.ones((len(self.bow.tokensDF),1))

        V,s,_ = sparsesvd(C.multiply(self.w).tocsc(),k=k)
        self.V = V / s[:,np.newaxis]


    def vectorizeStrings(self,strings,epsilon=1e-9):
        C = self.bow.countMatrix((self.tokenizer(string) for string in strings))
        vecs = C.T.dot(self.V.T)

        return vecs



if __name__ == '__main__':

    # Test code
    import nama
    from nama.matcher import Matcher
    import nama.similarity as similarity
    import cProfile as profile
    from nama.hashes import corpHash

    # Initialize the matcher
    matcher = Matcher(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO','ABC Inc.','XYZ Co.','ABC Inc.','XYZ Co.'])

    # Add some corpHash matches
    matcher.matchHash(nama.hashes.corpHash)

    # Initalize a new, untrained similarity model
    tokenizer = lambda s: ngrams(corpHash(s),3)
    lsaModel = LSAModel(matcher.strings(),tokenizer=tokenizer,k=100)
    lsaModel.vectorizeStrings(matcher.strings())
    matchDF = similarity.findNearestMatches(matcher.strings(),lsaModel)
    matchDF = similarity.scoreSimilarity(matchDF,matcher,show_plot=True)


    # matcher.suggestMatches(lsaModel,min_score=0)
