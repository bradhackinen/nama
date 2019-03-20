import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def findNearestMatches(strings,similarityModel,n=10,drop_duplicates=True,normalize=False,drop_zero_vecs=True,**nearestNeighborsArgs):
    strings = np.array(sorted(set(strings)))
    n = min(n,len(strings))

    vecs = similarityModel.vectorizeStrings(strings)

    # Optionally drop strings with zero-vectors
    if drop_zero_vecs:
        nonzero = (vecs != 0).max(axis=1)
        strings = strings[nonzero]
        vecs = vecs[nonzero,:]

    if normalize:
        vecs = vecs / np.sqrt((vecs**2).sum(axis=1))[:,np.newaxis]

    nearestNeighbors = NearestNeighbors(n_neighbors=n,**nearestNeighborsArgs)
    nearestNeighbors.fit(vecs)

    distances,matches = nearestNeighbors.kneighbors(vecs)

    matchPairs = np.vstack([np.kron(np.arange(len(strings)),np.ones(n).astype(int)),matches.ravel()]).T

    if drop_duplicates:
        matchPairs = np.sort(np.array(matchPairs),axis=1)

    matchDF = pd.DataFrame(matchPairs,columns=['string0','string1'])
    matchDF['score'] = np.exp(-distances.ravel())

    matchDF = matchDF[matchDF['string0'] != matchDF['string1']].copy()

    if drop_duplicates:
        matchDF = matchDF.drop_duplicates(['string0','string1'])

    for i in [0,1]:
        matchDF['string{}'.format(i)] = matchDF['string{}'.format(i)].apply(lambda s: strings[s])

    matchDF = matchDF.sort_values('score',ascending=False).reset_index(drop=True)

    return matchDF


def scoreCurve(d,a=1,b=1):
    return np.exp(-(np.abs(a)*d + np.abs(b)*(d**2)))


def calibrateScores(matchDF,matcher,max_sample=10000,show_plot=False,plot_res=100):

    componentMap = matcher.componentMap()
    for i in [0,1]:
        matchDF['component{}'.format(i)] = matchDF['string{}'.format(i)].apply(lambda s: componentMap[s])

    matchDF['within_component'] = matchDF['component0'] == matchDF['component1']

    sampleDF = matchDF[matchDF['score']<1]
    if len(sampleDF) > max_sample:
        sampleDF = sampleDF.sample(max_sample)

    if sampleDF['within_component'].std() == 0:
        raise Exception('Warning: Need both within and between-component matches with imperfect scores to calibrate.')

    sampleDF = sampleDF.sort_values('score')

    try:
        (a,b),cov = curve_fit(scoreCurve,sampleDF['distance'].values,sampleDF['within_component'].values)
    except:
        b=1
        a,cov = curve_fit(lambda d,a: scoreCurve(d,a,b),sampleDF['distance'].values,sampleDF['within_component'].values)

    if show_plot:
        plt.scatter(x='distance',y='within_component',data=matchDF)
        x = np.linspace(0,matchDF['distance'].max(),plot_res)
        plt.plot(x,scoreCurve(x,a,b))

    matchDF['score'] = scoreCurve(matchDF['distance'],a,b)

    return matchDF



# def scorePlot()
#
#
# import seaborn as sb
# import matplotlib.pyplot as plt
#
# plt.scatter(x='distance',y='within_component',data=matchDF)
# plt.plot(x='distance',y='score',data=matchDF)
#
# plt.plot(matchDF['distance'],matchDF['score'])
#
# plt.plot(np.linspace(0,4,100),scoreCurve(np.linspace(0,4,100),0.5,0.5))
# plt.plot(np.linspace(0,4,100),scoreCurve(np.linspace(0,4,100),0,1))
# plt.plot(np.linspace(0,4,100),scoreCurve(np.linspace(0,4,100),1,0))
