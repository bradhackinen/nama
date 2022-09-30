'''
This script builds matchers from names that have been manually
cleaned by the Center for Responsive Politics. The source data is available from
www.opensecrets.org.

The opensecrets data loader module used in this script is available here:
https://github.com/bradhackinen/opensecrets
'''

from pathlib import Path
import pandas as pd
import re
# from unidecode import unidecode
from collections import Counter

import nama
from nama import root_dir

import opensecrets


def clean_name(s):
    # Standardise whitespace
    s = re.sub(r'\s+',' ',s.strip())

    # Take client part of names containing "on behalf of" or "obo"
    m = re.search(r'([ \(]on behalf of|obo) ([^\)]+)',s,flags=re.IGNORECASE)
    if m:
        s = m.group(2).strip()

    # Drop raw strings that look like they contain multiple names
    # (sometimes "on behalf of" appears multiple times - drop these cases also)
    if re.search(r'([/\(\)\[\]\{\}]|f[/\.]?k[/\.]?a|formerly|\(for|doing business as|d/?b/?a)|on behalf of|in affiliation with',s,flags=re.IGNORECASE):
        return ''

    return s


def get_training_matcher(df,raw_col,clean_col):
    df = df[[raw_col,clean_col]].dropna()

    for c in raw_col,clean_col:
        df[c] = df[c].apply(clean_name)
        df = df[df[c].str.len() > 1]

    # Collapse to counts
    counts = Counter((s0,s1) for s0,s1 in df[[raw_col,clean_col]].values)
    counts.update((s,s) for s in df[clean_col])

    """
    Some raw names are associated with multiple different clean names (reason is unclear)
    In these cases, keep only the most common clean name associated with each raw name.
    """
    df = pd.DataFrame(((s0,s1,c) for (s0,s1),c in counts.most_common()),columns=['string0','string1','count']) \
            .groupby('string0') \
            .first() \
            .reset_index()

    matcher = nama.from_df(df,string_column='string0',group_column='string1',count_column='count')

    return matcher


# Load cleaned and raw client strings from lobbying data
print('Loading filings data')
filings_df = opensecrets.load_df('lobbying.filings')

for c in 'client','registrant':
    print(f'Building {c} matcher')
    matcher = get_training_matcher(filings_df,f'{c}_raw',c)
    save_name = Path(root_dir)/'training'/'data'/f'opensecrets_{c}s.csv'
    print(f'Saving {c} matcher as {save_name}')
    matcher.to_csv(save_name)

# s = 'the Washington Group on behalf of SoundExchange'
c = 'client_raw'

filings_df[filings_df[c].astype(str).str.contains('Piramal')].T


df = filings_df[filings_df[c].astype(str).str.contains('Piramal')][[c]] \
    .value_counts() \
    .to_frame('count') \
    .reset_index()

df['clean'] = df['client_raw'].apply(clean_name)


# # load donation org strings
# print('Loading donor employer data')
# donors_df = opensecrets.load_df('campaign_finance.individual',fields=['org_name','employer'])
#
# print(f'Building donor employer matcher')
# matcher = get_training_matcher(filings_df,'employer','org_name')
# save_name = Path(root_dir)/'training'/'data'/f'opensecrets_employers.csv'
# print(f'Saving donor employer matcher as {save_name}')
# matcher.to_csv(save_name)
#
#
# df = matcher.to_df()
#
# df2 = df[df['group'] != '!BEW LO 58']
# df3 = df2[df2['group'] != '!ST AMERICAN']
# df4 = df3[df3['count']>1]
#
# df[df['group'].str.startswith('{')]['group'].value_counts()
# df[df['group'].str.startswith('#')]['group'].value_counts()
#
# df3.head(50)
#
#
# df3.sample(50)
#
# df4.sample(50)
#
#
clients_df = pd.read_csv(Path(root_dir)/'training'/'data'/'opensecrets_clients.csv')
#
# clients_df[clients_df['string'].str.contains('on behalf of',case=False)].sample(50)
