import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

corpwatch_dir = Path('/media/hd2/Data/corpwatch/corpwatch_api_tables_csv')


names_df = (pd.read_csv(corpwatch_dir/'company_names.csv',
                        usecols=['company_name','cw_id'],
                        sep='\t',na_values='NULL')
                .drop_duplicates()
                .assign(
                    n_ids = lambda df:df.groupby('company_name')['cw_id'].transform('count'),
                    n_names = lambda df:df.groupby('cw_id')['company_name'].transform('count')))

names_df.sort_values('cw_id').query('n_names > 1')


names_df.loc[names_df['cw_id'] == names_df['cw_id'].sample(1).values[0]]




df = (pd.read_csv(corpwatch_dir/'company_names.csv',
                        sep='\t',na_values='NULL'))


df['source'].value_counts()