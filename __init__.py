import os
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from nama.similarity import loadModelPackage,findFuzzyMatches
from nama.defaults import defaultSimilarityModel
from nama.hashes import *


class Matcher():
    def __init__(self,strings=None,similarityModel=defaultSimilarityModel,cuda=False):
        self.G = nx.Graph()
        self.counts = Counter()
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
        self.counts.update(strings)
        self.G.add_nodes_from(((s,{'count':self.counts[s]}) for s in strings))

    def removeStrings(self,strings):
        self.G.remove_nodes_from(strings)
        for s in strings:
            del self.counts[s]

    def addMatch(self,string0,string1,score=1,source='manual'):
        self.G.add_edge(string0,string1,score=score,source=source)

    def removeMatch(self,string0,string1):
        self.G.remove_edge(string0,string1)

    def addMatches(self,pairs,scores,source):
        for (s0,s1),score in zip(pairs,scores):
            if s0 != s1:
                if self.G.has_edge(s0,s1) and self.G[s0][s1]['score'] >= score:
                    # Skip new connection if score lower than or equal to existing connection
                    continue
                self.G.add_edge(s0,s1,score=score,source=source)

    def removeMatches(self,pairs):
        self.G.remove_edges_from(pairs)

    def filterMatches(self,filter_function):
        for s0,s1,d in list(self.G.edges(data=True)):
            d = d.copy()
            d['string0'] = s0
            d['string1'] = s1
            if not filter_function(d):
                self.G.remove_edge(s0,s1)

    def removeUnused(self):
        '''
        Removes any nodes that do not connect counted strings
        '''
        while True:
            # Repeatedly find and remove uncounted leaf nodes
            unused = [s for s in self.G.nodes() if (s not in self.counts) and (self.G.degree[s] <= 1)]
            for s in unused:
                self.G.remove_node(s)

            if not unused:
                # Break out of loop when there are no more nodes to remove
                break


    def applyMatchDF(self,matchDF,source='matchDF'):
        self.addMatches(zip(matchDF['string0'],matchDF['string1']),matchDF['score'],source=source,remove_unused=remove_unused)

        nonMatchesDF = matchDF[matchDF['score']==0]
        self.removeMatches(zip(nonMatchesDF['string0'],nonMatchesDF['string1']))

    # def applyMatchCSV(self,filename,encoding='utf8'):
    #     matchDF = pd.read_csv(filename,encoding=encoding)
    #     matchDF['line'] = matchDF.index.get_level_values(0) + 1
    #     # matchDF['source'] = matchDF['source'] + ' line: ' + (matchDF.index.get_level_values(0) + -1).astype(str)
    #
    #     self.applyMatchDF(matchDF,source=filename)

    def matchHash(self,hash_function=basicHash,score=1,min_string_count=1):
        # df = pd.DataFrame(list(self.counts.keys()),columns=['string0'])
        # df['string1'] = df['string0'].apply(hash_function)
        #
        # if drop_unused:
        #     # Only add hash strings if they connect two other strings
        #     df = df[df.groupby('string1')['string0'].transform(len)>1]
        #     df['score'] = 1
        #
        # self.applyMatchDF(df,source=hash_function.__name__)

        pairs = [(s,hash_function(s)) for s in self.G.nodes() if self.counts[s] >= min_count]
        scores = [score]*len(pairs)
        self.addMatches(pairs=pairs,scores=scores,source=hash_function.__name__)

    def matchSimilar(self,min_score=0.9,batch_size=100,min_string_count=1):
        if self.similarityModel is None:
            raise Exception('No similarity model loaded')

        matchDF = findFuzzyMatches((s for s in self.G.nodes() if self.counts[s] >= min_count),
                            self.similarityModel,min_score=min_score,batch_size=batch_size)

        self.addMatches(zip(matchDF['string0'],matchDF['string1']),matchDF['score'],source='similarity')

    def componentMap(self):
        components = nx.connected_components(self.G)
        return {s:i for i,component in enumerate(components) for s in component}

    def matches(self,string=None):
        if string is None:
            return self.G
        else:
            return self.G.subgraph(nx.node_connected_component(self.G,string))

    def matchesDF(self,string=None):
        G = self.matches(string)
        df = pd.concat([pd.DataFrame(list(G.edges()),columns=['string0','string1']),
                        pd.DataFrame([d for s0,s1,d in G.edges(data=True)])],axis=1)
        return df

    def componentsDF(self):
        componentMap = self.componentMap()

        return pd.DataFrame([(s,i) for s,i in componentMap.items() if s in self.counts],columns=['string','component'])

    def componentSummaryDF(self,sort_by='count',ascending=False):
        df = matcher.componentsDF()
        df['count'] = df['string'].apply(lambda s: matcher.counts[s])
        df = componentsDF.sort_values(['component','count'],ascending=[True,False])
        df['unique'] = 1
        df = df.groupby('component').agg({'string':'first','count':'sum','unique':'sum'})

        if sort_by is not None:
            df = df.sort_values(sort_by,ascending=ascending)

        return df

    def bridgeImpacts(self,string=None):
        G = self.matches(string)
        impacts = {}
        for component in nx.connected_components(G):

            if len(component) == 1:
                continue

            elif len(component) == 2:
                s0,s1 = component
                impacts[(s0,s1)] = self.counts[s0]*self.counts[s1]

            else:
                G_c = G.subgraph(component)

                for s0,s1 in nx.algorithms.bridges(G_c):
                    G_b = G_c.copy()
                    G_b.remove_edge(s0,s1)

                    bridgedComponents = list(nx.connected_components(G_b))
                    assert len(bridgedComponents) == 2

                    counts = [sum(self.counts[s] for s in c) for c in bridgedComponents]
                    impact = counts[0]*counts[1]

                    impacts[(s0,s1)] = impact

        return impacts

    def bridgeImpactsDF(self,string=None):
        impacts = self.bridgeImpacts(string=string)
        df = pd.DataFrame([(s0,s1,impact) for (s0,s1),impact in impacts.items()],columns=['string0','string1','impact'])

        df = pd.merge(df,self.matchesDF(string=string))

        df = df.sort_values('impact',ascending=False)


        return df

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

        return pd.merge(leftDF,rightDF,on=component_column_name,how=how)

    def plotMatches(self,string=None,ax=None,cmap='tab10'):
        G = self.matches(string)

        if string is None:
            pos = nx.spring_layout(G,weight='score',k=0.75,iterations=50)
        else:
            pos = nx.kamada_kawai_layout(G)

        stringNodes = self.counts.keys()
        hashNodes = [s for s in G.nodes() if s not in self.counts]
        sources = sorted(set(nx.get_edge_attributes(G,'source').values()))

        if ax is None:
            fig, ax = plt.subplots()
        cmap = plt.get_cmap(cmap)
        for i,source in enumerate(sources):
            sourceEdges = [(s0,s1,d) for s0,s1,d in G.edges(data=True) if d['source']==source]
            coordinates = [[pos[s0],pos[s1]] for s0,s1,d in sourceEdges]
            alphas = [d['score'] for s0,s1,d in sourceEdges]

            color = cmap(i)[:3]
            rgba = [color+(d['score'],) for s0,s1,d in sourceEdges]

            if source == 'similarity':
                linestyles = ':'
            else:
                linestyles = 'solid'
            lc = mc.LineCollection(coordinates,label=source,color=rgba,linestyles=linestyles,zorder=0)

            ax.add_collection(lc)

            edgeLabels = {(s0,s1):'{:.2f}'.format(d['score']) for s0,s1,d in sourceEdges if d['score']<1}
            nx.draw_networkx_edge_labels(G,edge_labels=edgeLabels,font_color=color,pos=pos,bbox={'color':'w','linewidth':1})#,zorder=100)

        nx.draw_networkx_nodes(G,node_color='w',pos=pos)

        nx.draw_networkx_labels(nx.subgraph(G,stringNodes),font_color='k',pos=pos)
        nx.draw_networkx_labels(nx.subgraph(G,hashNodes),font_color='#888888',pos=pos)

        plt.legend()

        ax.axis('off')
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)

        return ax




if __name__ == '__main__':

    # Run demo code

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

    # Match strings if they share a hash string
    # (corphash removes common prefixes and suffixes (the, inc, co, etc) and makes everything lower-case)
    matcher.matchHash(corpHash)

    # Now merge will find all the matches we want except  'ABC Inc.' <--> 'A.B.C. INCORPORATED'
    matcher.merge(df1,df2,on='name')

    # Use fuzzy matching to find likely misses (GPU accelerated with cuda=True)
    matcher.matchSimilar(min_score=0.5)

    # Review fuzzy matches
    matcher.matchesDF()

    # Add manual matches
    matcher.addMatch('ABC Inc.','A.B.C. INCORPORATED')
    matcher.addMatch('XYZ Co.','X Y Z CO')

    # Drop remaining fuzzy matches from the graph
    matcher.filterMatches(lambda m: m['source'] != 'similarity')

    # Final merge
    matcher.merge(df1,df2,on='name')

    # We can also cluster names by connected component and assign ids to each
    matcher.componentsDF()

    # matcher.plotMatches()

    matcher.bridgeImpactsDF()


    matcher.plotMatches()

    matcher.addMatch('xyz','123')
    matcher.addMatch('456','123')

    matcher.removeUnused()
