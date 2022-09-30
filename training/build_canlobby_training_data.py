from pathlib import Path
import pandas as pd
import re
from unidecode import unidecode

from nama.config import data_dir


sorena_df = pd.read_csv(Path(root_dir)/'training_data'/'ClientNames_lower.csv',encoding='iso-8859-1')
sorena_df = sorena_df.rename(columns={'name':'string_lower'})
sorena_df = sorena_df[['string_lower','updated_name']]
sorena_df = sorena_df.dropna()

for c in 'string_lower','updated_name':
    sorena_df[c] = sorena_df[c].apply(unidecode)
    sorena_df[c] = sorena_df[c].apply(lambda s: re.sub(r'\s+',' ',s))
    sorena_df[c] = sorena_df[c].str.strip()

sorena_df = sorena_df.drop_duplicates()



com_df = pd.read_csv(Path(root_dir)/'training_data'/'Communication_PrimaryExport.csv')

com_df = com_df[['EN_CLIENT_ORG_CORP_NM_AN']] \
            .rename(columns={'EN_CLIENT_ORG_CORP_NM_AN':'string'})
com_df['string'] = com_df['string'].str.strip()


df = com_df['string'].value_counts().to_frame('count') \
            .reset_index() \
            .rename(columns={'index':'string'})

df['string_lower'] = df['string'].str.lower() \
                        .apply(unidecode) \
                        .apply(lambda s: re.sub(r'\s+',' ',s)) \
                        .str.strip()

df = pd.merge(df,sorena_df,'left',on='string_lower')

df['group'] = df['updated_name']

df.loc[df['group'].isnull(),'group'] = df.loc[df['group'].isnull(),'string_lower']


df = df[['string','count','group']]

df = df.dropna()

df.to_csv(Path(root_dir)/'demo'/'canlobby_clients_manual.csv')
