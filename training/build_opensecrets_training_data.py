'''
This script builds matchess from names that have been manually
cleaned by the Center for Responsive Politics. The source data is available from
www.opensecrets.org.

The opensecrets data loader module used in this script is available here:
https://github.com/bradhackinen/opensecrets
'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter

import opensecrets

import nama
from nama.utils import simplify
from nama.scoring import split_on_groups

data_dir = Path(os.environ['NAMA_DATA'])



def clean_name(s):
    # Standardise whitespace
    s = re.sub(r'\s+',' ',s.strip())

    # Take client part of names containing "on behalf of" or "obo"
    m = re.search(r'([ \(]on behalf of|obo) ([^\)]+)',s,flags=re.IGNORECASE)
    if m:
        s = m.group(2).strip()

    # Drop raw strings that look like they contain multiple names
    # (sometimes "on behalf of" appears multiple times - drop these cases also)
    if re.search(r'([/\(\)\[\]\{\}]|f[/\.]?k[/\.]?a|formerly|\(for|doing business as|d/?b/?a)|on behalf of| OBO |in affiliation with',s,flags=re.IGNORECASE):
        return np.nan

    return s


# Load cleaned and raw client strings from lobbying data
print('Loading filings data')
filings_df = opensecrets.load_df('lobbying.filings')

print('Getting string pairs')
raw_df = (pd.concat([
            filings_df[['client_raw','client']]
                .dropna()
                .rename(columns={'client_raw':'raw','client':'clean'}),
            filings_df[['client']]
                .dropna()
                .rename(columns={'client':'clean'})
                .assign(raw=lambda df: df['clean']),
            filings_df[['registrant_raw','registrant']]
                .dropna()
                .rename(columns={'registrant_raw':'raw','registrant':'clean'}),        
            filings_df[['registrant']]
                .dropna()
                .rename(columns={'registrant':'clean'})
                .assign(raw=lambda df: df['clean']),
            ])
            .assign(count=1)
            .groupby(['raw','clean'])[['count']].sum()
            .reset_index()
            .assign(raw=lambda df: df['raw'].apply(clean_name))
            .dropna())

# Some raw names are associated with multiple different clean names (reason is unclear)
# In these cases, keep only the most common clean name associated with each raw name.
raw_df = (raw_df
            .sort_values('count',ascending=False)
            .drop_duplicates(subset=['raw'],keep='first'))

# Add uppercase strings
upper_df = raw_df.assign(raw = lambda df: df['raw'].str.upper())

raw_df = (pd.concat([raw_df,upper_df])
          .drop_duplicates(subset=['raw'],keep='first'))

print('Compiling match data')
# Build match data
matches = nama.from_df(raw_df,
                    string_column='raw',
                    group_column='clean',
                    count_column='count')

# Unite by simplify - want to make sure we aren't missing easy matches
matches = matches.unite(simplify)


# Save files
matches.to_csv(data_dir/'training_data'/'opensecrets_all_matches.csv')

train,test = split_on_groups(matches,0.8,seed=1)

train.to_csv(data_dir/'training_data'/'opensecrets_train.csv')
test.to_csv(data_dir/'training_data'/'opensecrets_test.csv')