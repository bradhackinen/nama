import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt

from nama.similarity import loadModelPackage,findFuzzyMatches
from nama.defaults import defaultSimilarityModel
from nama.hashes import *


class Matcher():
    def __init__(self,strings=None,similarityModel=defaultSimilarityModel,cuda=False):
        self.G = nx.Graph()
        self.stringCounts = Counter()
        self.cuda = cuda

        if strings:
            self.addStrings(strings)

        if similarityModel:
            self.loadSimilarityModel(similarityModel)
        else:
            self.similarityModel = None

    def loadSimilarityModel(self,filename=defaultSimilarityModel):
        self.similarityModel = loadModelPackage(filename,cuda=self.cuda)

    def addStrings(self,strings):
        self.G.add_nodes_from(strings)
        self.stringCounts.update(strings)

    def removeStrings(self,strings):
        self.G.remove_nodes_from(strings)
        for s in strings:
            del self.stringCounts[s]

    def connect(self,string0,string1,score=1,source='manual'):
        G.add_edge(string0,string1,score=score,method=method)

    def disconnect(self,string0,string1):
        G.remove_edge(string0,string1)

    def addConnections(self,pairs,scores,source):
        for (s0,s1),score in zip(pairs,scores):
            # Skip new connection if score lower than or equal to existing connection
            if self.G.has_edge(s0,s1) and self.G[s0][s1]['score'] >= score:
                continue
            self.G.add_edge(s0,s1,score=score,source=source)

    def removeConnections(self,pairs):
        self.G.remove_edges_from(pairs)

    def applyMatchDF(self,matchDF,score=1,source='manual'):
        matchDF = matchDF.copy()
        if 'score' not in matchDF.columns:
            matchDF['score'] = score
        if 'source' not in matchDF.columns:
            matchDF['source'] = source

        correctDF = matchDF[matchesDF['match']][['string0','string1','score','source']]
        incorrectDF = matchDF[~matchesDF['match']][['string0','string1']]

        self.addConnections(zip(correctDF['string0'],correctDF['string1']),correctDF['score'],source=source)
        self.removeConnections(zip(correctDF['string0'],correctDF['string1']))

        # self.G.add_edges_from((s0,s1,{'score':score,'source':source}) for i,s0,s1,score,source in correctDF.itertuples())
        # self.G.remove_edges_from((s0,s1) for i,s0,s1 in incorrectDF.itertuples())

    def applyMatchCSV(self,filename,encoding='utf8'):
        matchDF = pd.read_csv(filename,encoding=encoding)
        matchDF['match'] = matchDF['match'] > 0.5
        matchDF['source'] = filename
        matchDF['source'] = matchDF['source'] + ' line: ' + (matchDF.index.get_level_values(0) + -1).astype(str)

    def connectHash(self,hash_function=basicHash,score=1):
        pairs = [(s,hash_function(s)) for s in list(self.G.nodes())]
        scores = [score]*len(pairs)
        self.addConnections(pairs=pairs,scores=scores,source=hash_function.__name__)
        # for s in list(self.G.nodes()):
        #     self.G.add_edge(s,hash_function(s),score=score,source=hash_function.__name__)

    def connectSimilar(self,min_score=0.9,batch_size=100):
        if self.similarityModel is None:
            raise Exception('No similarity model loaded')

        matchDF = findFuzzyMatches(self.stringCounts.keys(),self.similarityModel,min_score=min_score,batch_size=batch_size)

        self.addConnections(zip(matchDF['string0'],matchDF['string1']),matchDF['score'],source='similarity')

        # self.G.add_edges_from((s0,s1,{'score':score,'source':'similarity'}) for i,s0,s1,score in matchDF.itertuples())

    def componentMap(self):
        components = nx.connected_components(self.G)
        return {s:i for i,component in enumerate(components) for s in component}

    def connectionsDF(self):
        df = pd.concat([pd.DataFrame(list(self.G.edges()),columns=['string0','string1']),
                        pd.DataFrame([d for s0,s1,d in self.G.edges(data=True)])],axis=1)
        return df

    def clustersDF(self):
        componentMap = self.componentMap()
        pd.DataFrame([(s,i) for s,i in componentMap.items()],columns=['string','component'])

    def merge(self,leftDF,rightDF,how='inner',on=None,left_on=None,right_on=None,component_column_name='component'):

        if on is not None:
            left_on = on
            right_on = on

        if left_on is None or right_on is None:
            raise Exception('Must provide column to merge on')

        componentMap = self.componentMap()

        leftDF = leftDF.copy()
        rightDF = rightDF.copy()

        leftDF[component_column_name] = leftDF[left_on].apply(lambda s: componentMap.get(s,np.nan))
        rightDF[component_column_name] = rightDF[right_on].apply(lambda s: componentMap.get(s,np.nan))

        if how in ['inner','right']:
            leftDF = leftDF[leftDF[component_column_name].notnull()]

        if how in ['inner','left']:
            leftDF = leftDF[leftDF[component_column_name].notnull()]

        matchesDF = pd.merge(leftDF,rightDF,on=component_column_name,how=how)

        return matchesDF



if __name__ == '__main__':

    import pandas as pd
    import nama
    from nama.hashes import *


    df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
    df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])

    # Initialize the matcher
    matcher = Matcher()

    # Add the strings we want to match to the match graph
    matcher.addStrings(df1['name'])
    matcher.addStrings(df2['name'])

    # At this point we can merge on exact matches, but there isn't much point (equivalent to pandas merge function)
    matcher.merge(df1,df2,on='name')

    # Connect strings if they share a hash string
    # (corphash removes common prefixes and suffixes (the, inc, co, etc) and makes everything lower-case)
    matcher.connectHash(corpHash)

    # Now merge will find all the matches we want except  'ABC Inc.' <--> 'A.B.C. INCORPORATED'
    matcher.merge(df1,df2,on='name')

    # Use fuzzy matching to find likely misses (GPU accelerated with cuda=True)
    matcher.connectSimilar(min_score=0)

    # Review fuzzy matches
    connectionsDF = matcher.connectionsDF()

    # Add manual match
    matcher.connect('ABC Inc.','A.B.C. INCORPORATED')

    # Drop other fuzzy matches from the graph
    matcher.disconnectBySimilarity(min_score=1)

    # Final merge, ignoring fuzzy matches
    matcher.merge(df1,df2,on='name')


    # We can also cluster names and assign ids to each
    clusterDF = matcher.clustersDF()





self = matcher

pos = nx.spring_layout(self.G,weight='similarity',k=0.75)

stringNodes = self.stringCounts.keys()
hashNodes = [s for s in self.G.nodes() if s not in self.stringCounts]

nx.draw_networkx_edges(nx.subgraph(self.G,hashNodes),pos=pos)
nx.draw_networkx_nodes(nx.subgraph(self.G,hashNodes),node_color='w',pos=pos)
nx.draw_networkx_labels(nx.subgraph(self.G,stringNodes),font_color='k',pos=pos)
nx.draw_networkx_labels(nx.subgraph(self.G,hashNodes),font_color='r',pos=pos)

ax = plt.gca()
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)

# nx.get_edge_attributes(self.G,'source')
#
# nx.draw_networkx(nx.subgraph(self.G,self.stringCounts.keys()),edge_color='r',pos=pos)
#
# nx.draw_networkx(self.G,pos=pos,node_color='w')
#
# self.connectionsDF()

# list(matcher.G.nodes())
# matcher.G.edges(data=True)
