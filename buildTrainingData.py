import os
import pandas as pd
import numpy as np


trainingDir = r'C:\Users\Brad\Google Drive\Research\Python3\nama\trainingData'


#Convert old nama training sets-------------------------------------------------
def convertOldTrainingSet(oldFile,newFile):
    trainingDF = pd.read_csv(oldFile,encoding='mbcs')
    trainingDF = trainingDF.rename(columns={'string_0':'query_string','string_1':'candidate_string'})
    trainingDF = trainingDF[trainingDF['query_string']!=trainingDF['candidate_string']]

    trainingDF.to_csv(newFile,index=False,encoding='utf8')


convertOldTrainingSet(r'C:\Users\Brad\Google Drive\Research\US Clean Power Plan\Python\Data Processing\namaTrainingSet_capIQ.csv',
                        os.path.join(trainingDir,'energyIndustryCapIQ_training.csv'))

convertOldTrainingSet(r'C:\Users\Brad\Google Drive\Research\US Clean Power Plan\Python\Data Processing\namaTrainingSet_capIQ_comments.csv',
                        os.path.join(trainingDir,'energyIndustryCommentsCapIQ_training.csv'))

convertOldTrainingSet(r'C:\Users\Brad\Google Drive\Research\RegulatoryComments\CRSPNamaTrainingSet.csv',
                        os.path.join(trainingDir,'commentsCRSP_training.csv'))


#Build training set from lobbying data------------------------------------------
#Note: Will contain only positive matches
from openSecrets import lobbying
from vectorizedMinHash import fastNGramHashes
filingsDF = lobbying.loadDF('filings')

clientsDF = filingsDF[['client','client_raw']].drop_duplicates()
# del filingsDF


# clientsDF['client_raw_cleaned'] = clientsDF['client_raw'].str.replace(r'\(?(f/?k/?a|formerly known as).+','',case=False)
# clientsDF['client_raw_cleaned'] = clientsDF['client_raw_cleaned'].str.replace(r'.+(on behalf of|\(for)\s+','',case=False)
# clientsDF['client_raw_cleaned'] = clientsDF['client_raw_cleaned'].str.replace(r'\(.*\)','',case=False)
# clientsDF['client_raw_cleaned'] = clientsDF['client_raw_cleaned'].str.replace(r'[\(\)]','',case=False)
# clientsDF['client_raw_cleaned'] = clientsDF['client_raw_cleaned'].str.strip()




clientsDF = clientsDF.dropna(axis=0)

clientsDF = clientsDF[~clientsDF['client_raw'].str.contains(r'([/\(\)\[\]\{\}]|f/?k/?a|formerly|on behalf of|\(for|doing business as|d/?b/?a)',case=False)]
clientsDF['client_raw'] = clientsDF['client_raw'].str.strip()

clientsDF[clientsDF[]]

for c in '','_raw':
    clientsDF['client{}_ngrams'.format(c)] = clientsDF['client{}'.format(c)].apply(lambda s: set(fastNGramHashes(s.lower().encode('utf8'),n=2)))

clientsDF['jaccard'] = [len(s0 & s1)/len(s0 | s1) for i,s0,s1 in clientsDF[['client_ngrams','client_raw_ngrams']].itertuples()]

for c in '','_raw':
    clientsDF['client{}_acronym'.format(c)] = clientsDF['client{}'.format(c)].str.title()
    clientsDF['client{}_acronym'.format(c)] = clientsDF['client{}_acronym'.format(c)].str.replace(r'[^A-Z]','')

clientsDF['acronym_intersection'] = [(a0 in a1) or (a1 in a0) for i,a0,a1 in clientsDF[['client_acronym','client_raw_acronym']].itertuples()]

# Select high quality matches only
trainingDF = clientsDF[clientsDF['acronym_intersection'] | (clientsDF['jaccard'] > 0.1)]

trainingDF = trainingDF[['client','client_raw']]
trainingDF = trainingDF.rename(columns={'client':'candidate_string','client_raw':'query_string'})
trainingDF = trainingDF.replace('',np.nan)
trainingDF = trainingDF.dropna(axis=0)
trainingDF = trainingDF[trainingDF['query_string']!=trainingDF['candidate_string']]
trainingDF['match'] = 1



trainingDF.to_csv(os.path.join(trainingDir,'lobbyingClients_training.csv'),index=False,encoding='utf8')



#Build training data from compustat company name and legal name fields
compustatDF = pd.read_csv(r'E:\Data\WRDS\compustat_annualSummary_2000-2018.csv')
trainingDF = compustatDF[compustatDF['conm']!=compustatDF['conml']][['conm','conml']].drop_duplicates()
trainingDF.columns = ['query_string','candidate_string']
trainingDF['match'] = 1


trainingDF.to_csv(os.path.join(trainingDir,'compustatLegalName_training.csv'),index=False,encoding='utf8')




#Build training data from simple modifications
trainingFiles = [os.path.join(trainingDir,f) for f in os.listdir(trainingDir) if f.endswith('.csv')]
trainingDF = pd.concat([pd.read_csv(f,encoding='utf8') for f in trainingFiles])


#Create training data with simple puctuation modifications
punctuationDF = pd.DataFrame(list(set(trainingDF['query_string']) | set(trainingDF['candidate_string'])),columns=['query_string'])
punctuationDF['candidate_string'] = punctuationDF['query_string']
punctuationDF['candidate_string'] = punctuationDF['candidate_string'].str.replace(r'[\.,]',' ')
punctuationDF['candidate_string'] = punctuationDF['candidate_string'].str.replace(r'\s+',' ')
punctuationDF['candidate_string'] = punctuationDF['candidate_string'].str.strip()
punctuationDF = punctuationDF[punctuationDF['candidate_string'] != punctuationDF['query_string']]
punctuationDF['match'] = 1

punctuationDF.sample(10000).to_csv(os.path.join(trainingDir,'puctuation_training.csv'),index=False,encoding='utf8')


#Create training data with simple puctuation modifications
capitalizationDF = pd.DataFrame(list(set(trainingDF['query_string']) | set(trainingDF['candidate_string'])),columns=['query_string'])
capitalizationDF['candidate_string'] = capitalizationDF['query_string']
capitalizationDF['candidate_string'] = capitalizationDF['candidate_string'].str.upper()
capitalizationDF = capitalizationDF[capitalizationDF['candidate_string'] != capitalizationDF['query_string']]
capitalizationDF['match'] = 1

capitalizationDF.sample(10000).to_csv(os.path.join(trainingDir,'capitalization_training.csv'),index=False,encoding='utf8')


#Create training data with 'The' prefix removed
thePrefixDF = pd.DataFrame(list(set(trainingDF['query_string']) | set(trainingDF['candidate_string'])),columns=['query_string'])
thePrefixDF['candidate_string'] = thePrefixDF['query_string']
thePrefixDF['candidate_string'] = thePrefixDF['candidate_string'].str.replace(r'^the ','',case=False)
thePrefixDF = thePrefixDF[thePrefixDF['candidate_string'] != thePrefixDF['query_string']]
thePrefixDF['match'] = 1

thePrefixDF.to_csv(os.path.join(trainingDir,'the_training.csv'),index=False,encoding='utf8')
