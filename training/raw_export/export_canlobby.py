import os
from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode

import nama

data_dir = Path(os.environ['NAMA_DATA'])

'''
The manual matching was done with lower-case strings, so we need to clean up
the manual data and then merge it with the raw strings to build the match data.
'''

# Load and clean the manual matches
manual_df = pd.read_csv(Path(data_dir)/'training_data'/'raw'/'canlobby'/'ClientNames_lower.csv',encoding='iso-8859-1')
manual_df = manual_df.rename(columns={'name':'string_lower'})
manual_df = manual_df[['string_lower','updated_name']]
manual_df = manual_df.dropna()

for c in 'string_lower','updated_name':
    manual_df[c] = manual_df[c].apply(unidecode)
    manual_df[c] = manual_df[c].apply(lambda s: re.sub(r'\s+',' ',s))
    manual_df[c] = manual_df[c].str.strip()

manual_df = manual_df.drop_duplicates()


# Merge with the raw strings from the communications files
communications_df = pd.read_csv(Path(data_dir)/'training_data'/'raw'/'canlobby'/'Communication_PrimaryExport.csv')

communications_df = communications_df[['EN_CLIENT_ORG_CORP_NM_AN']] \
            .rename(columns={'EN_CLIENT_ORG_CORP_NM_AN':'string'})
communications_df['string'] = communications_df['string'].str.strip()


# Combine the raw strings with the manual matches to created cleaned match data
cleaned_df = communications_df['string'].value_counts().to_frame('count') \
            .reset_index() \
            .rename(columns={'index':'string'})

cleaned_df['string_lower'] = cleaned_df['string'].str.lower() \
                        .apply(unidecode) \
                        .apply(lambda s: re.sub(r'\s+',' ',s)) \
                        .str.strip()

cleaned_df = (pd.merge(cleaned_df,manual_df,'left',on='string_lower')
                .rename(columns={'updated_name':'group'}))

# Fill singletons with string
cleaned_df.loc[cleaned_df['group'].isnull(),'group'] = cleaned_df.loc[cleaned_df['group'].isnull(),'string']


cleaned_df = cleaned_df[['string','count','group']]

cleaned_df = cleaned_df.dropna()


# convert to match data format
matches = nama.from_df(cleaned_df,
                    string_column='string',
                    group_column='group',
                    count_column='count')


matches.to_csv(Path(data_dir)/'training_data'/'processed'/'canlobby_raw_matches.csv')