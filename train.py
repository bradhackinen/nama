import os
import pandas as pd
from nama.nama import newModel,trainModel,findMatches,saveModelPackage,loadModelPackage
from torchUtilities import *
from torch import save

namaDir = r'C:\Users\Brad\Google Drive\Research\Python3\nama'
trainingDir = os.path.join(namaDir,'trainingData')
modelDir = os.path.join(namaDir,'trainedModels')


trainingFiles = [os.path.join(trainingDir,f) for f in os.listdir(trainingDir) if f.endswith('.csv')]
trainingDF = pd.concat([pd.read_csv(f,encoding='utf8') for f in trainingFiles])

sampleDF = trainingDF.sample(1000)


modelPackage = newModel(d=300,lr=0.001,weight_decay=1e-8,recurrent_layers=1,bidirectional=True,cuda=True)

# modelPackage = loadModelPackage(os.path.join(modelDir,'allTrainingData_2lbi_d200.bin'))

#Set learning rate
for g in modelPackage['optimizer'].param_groups:
    g['lr'] = 0.0001


#Repeat as desired:
historyDF = trainModel(modelPackage,sampleDF,epochs=20,minibatch_size=20,resample_prob=0.1,verbose=True)
plotLossHistory(historyDF),ylim=(0,0.4))



saveModelPackage(modelPackage, os.path.join(modelDir,'allTrainingData_2lbi_d200.bin'))


sampleDF = trainingDF.sample(1000)
matchesDF = findMatches(sampleDF['query_string'],sampleDF['candidate_string'],modelPackage)

matchesDF.sample(20).sort_values('score',ascending=False)
