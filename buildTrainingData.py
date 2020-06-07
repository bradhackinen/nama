'''
This script builds training data from lobbying filings that have been manually
cleaned by the Center for Responsive Politics. The source data is available from
www.opensecrets.org.

The openSecrets data loader used in this script is available here:
https://github.com/bradhackinen/openSecrets


NOTE: As currently written, the script generates train, validate and test files
that contain non-overlapping sets of string pairs. BUT, the same individual
string can appear in all three files if it is part of multiple pairs. This
occurs with high frequency for the cleaned version of some organization names.

In the future it might be useful to distinguish between testing few-shot and
zero-shot learning, based on whether the one of the strings in a validation or
test pair ever appears in the training data.
'''

from pathlib import Path
import pandas as pd
import numpy as np

import openSecrets

from nama import trainingDir


import re
from random import sample


# Load cleaned and raw client strings from lobbying data
filingsDF = openSecrets.load('lobbying.filings')

clientsDF = filingsDF[['client','client_raw']].drop_duplicates().dropna()
clientsDF.columns = ['string0','string1']

registrantsDF = filingsDF[['registrant','registrant_raw']].drop_duplicates().dropna()
registrantsDF.columns = ['string0','string1']

pairsDF = pd.concat([clientsDF,registrantsDF])


# Standardize whitespace (precautionary)
for c in ['string0','string1']:
    pairsDF[c] = [re.sub(r'\s+',' ',s.strip()) for s in pairsDF[c]]

# Drop cases where the clean and raw strings are identical (not useful for training)
pairsDF = pairsDF[pairsDF['string0']!=pairsDF['string1']]

# Drop raw strings that look like they contain multiple names
pairsDF = pairsDF[~pairsDF['string1'].str.contains(r'([/\(\)\[\]\{\}]|f[/\.]?k[/\.]?a|formerly|on behalf of|\(for|doing business as|d/?b/?a)',case=False)]

pairsDF.to_csv(trainingDir/'lobbyingOrgs.csv',index=False)



'''
Split out train, validate, and test sets

'''
# Shuffle
pairsDF = pairsDF.sample(frac=1).reset_index(drop=True)

validateDF = pairsDF.loc[:9999,:]
testDF = pairsDF.loc[10000:19999,:]
trainDF = pairsDF.loc[20000:]

validateDF.to_csv(trainingDir/'validate.csv',index=False)
testDF.to_csv(trainingDir/'test.csv',index=False)
trainDF.to_csv(trainingDir/'train.csv',index=False)


'''
Build augmented training data by mutating strings from the training set
'''

def removeRandomChar(s):
    if len(s) >= 2:
        i = np.random.randint(len(s))
        s = s[:i] + s[i+1:]
    return s

def replicateRandomChar(s):
    if len(s) >= 2:
        i = np.random.randint(len(s))
        j = np.random.randint(len(s))
        s = s[:i] + s[j] + s[i:]
    return s

def stripThe(s):
    s = re.sub(r'(^the )|(, the$)','',s,flags=re.IGNORECASE)
    return s

def swapAnd(s):
    if '&' in s:
        s = s.replace(' & ',' and ')
    else:
        s = re.sub(' and ',' & ',s,flags=re.IGNORECASE)
    return s

def stripLegal(s):
    s = re.sub(',?( (group|holding(s)?( co)?|inc(orporated)?|ltd|l\.?l?\.?[cp]|co(rp(oration)?|mpany)?|s\.?[ae]|p\.?l\.?c)[,\.]*)$','',s,count=1,flags=re.IGNORECASE)
    return s

def toAcronym(s):
    s = stripThe(s)
    s = stripLegal(s)
    s = re.sub(' and ',' ',s,flags=re.IGNORECASE)

    tokens = s.split()
    if len(tokens) > 1:
        return ''.join(t[0].upper() for t in tokens if t.lower() not in ['of','the','for'])
    else:
        return s.upper()

def truncateWords(s):
    tokens = []
    for t in s.split():
        if len(t) >= 6:
            tokens.append(t[:np.random.randint(2,len(t)-2)])
        else:
            tokens.append(t)
    s = ' '.join(tokens)
    return s


mutations = [
            stripLegal,
            lambda s: stripLegal(stripLegal(s)),
            stripThe,
            swapAnd,
            removeRandomChar,
            replicateRandomChar,
            toAcronym,
            truncateWords,
            lambda s: ' '.join(s.split()[:-1]),
            lambda s: re.sub(r',.*','',s,flags=re.IGNORECASE),
            lambda s: re.sub(r'[.,:;]','',s),
            lambda s: re.sub(r'[.,:;\-]',' ',s).strip(),
            lambda s: re.sub(r'(?<=\w)[aeiouy]','',s,flags=re.IGNORECASE),
            lambda s: s.title(),
            lambda s: s.upper()
            ]

def randomMutation(s,n=1):
    for m in sample(mutations,n):
        s = m(s)
    return s


orgStrings = set(trainDF['string0']) | set(trainDF['string1'])


n_samples = 3
df = pd.DataFrame([(s,randomMutation(s,1)) for s in orgStrings for i in range(n_samples)],columns=['string0','string1'])

df = df[df['string0'] != df['string1']]
df = df[df['string1'].str.len() >=2]

df = pd.concat([trainDF,df])

df = df.drop_duplicates()
df = df.sample(frac=1)

df.to_csv(trainingDir/f'train_augmented.csv',index=False)
