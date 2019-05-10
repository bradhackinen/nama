import pandas as pd
import numpy as np
from collections import Counter
import scipy.sparse as sparse
from unidecode import unidecode


def stringToAscii(s):
    s = unidecode(s)
    s = s.encode('ascii')
    return s



def dfChunks(df,chunk_size):
    for i in range(0,len(df),chunk_size):
        yield df[i:i+chunk_size]




class BOW():
    def fit(self,docs,no_below=2,returnCountMatrix=True):
        occurrences,J = self.occurrences(docs)

        # Prepare token vocabulary information
        docCounts = Counter(t for (i,t) in occurrences.keys())
        self.tokensDF = pd.DataFrame([(t,c) for t,c in docCounts.items() if c>=no_below],columns=['token','n_docs'])
        self.tokensDF = self.tokensDF.sort_values('n_docs',ascending=False).reset_index(drop=True)

        self.tokenid = {t:i for i,t in enumerate(self.tokensDF['token'])}

        # Optionally compute and return countMatrix (saves second pass of counting occurrences)
        if returnCountMatrix:
            return self.occurrencesToCountMatrix(occurrences,J)

    def countMatrix(self,docs):
        C = self.occurrencesToCountMatrix(*self.occurrences(docs))

        return C

    def frequencyMatrix(self,docs):
        C = self.countMatrix(docs)
        F = C.multiply(1/C.sum(axis=0))

        return F

    def occurrencesToCountMatrix(self,occurrences,J=None):
        C = np.array([(self.tokenid[t],j,c) for (j,t),c in occurrences.items() if t in self.tokenid])

        if J is None:
            J = C[:,1].max()+1

        C = sparse.coo_matrix((C[:,2],(C[:,0],C[:,1])),shape=(len(self.tokensDF),J)).tocsc()

        return C

    def occurrences(self,docs):
        occurrences = Counter()
        for j,doc in enumerate(docs):
            occurrences.update((j,t) for t in doc)

        return occurrences,j+1
