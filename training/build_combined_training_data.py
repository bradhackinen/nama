from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
# from nama.strings import simplify
data_dir = Path('/home/brad/Dropbox/Data/nama')

canlobby = nama.read_csv(data_dir/'training_data'/'canlobby_train.csv')
opensecrets = nama.read_csv(data_dir/'training_data'/'opensecrets_train.csv')

gold = canlobby + opensecrets


gold.to_csv(data_dir/'training_data'/'combined_train.csv')


# HACK: Remove compound strings before saving matcher

import re
compound_pattern = r'(f[/\.]?k[/\.]?a|formerly|\(for |doing business as |d/?b/?a )|on behalf of| OBO |in affiliation with'
compound_strings = [s for s in gold.strings() if re.search(compound_pattern,s,flags=re.IGNORECASE)]


gold = gold.drop(compound_strings)


gold.to_csv(data_dir/'training_data'/'combined_train_no_compound.csv')



opensecrets.matches()
# gold_df = gold.to_df()


# upper_df = gold_df.copy() \
#             .assign(string=lambda df: df['string'].str.upper())

# simple_df = gold_df.copy() \
#             .assign(string=lambda df: [simplify(s) for s in df['string']])

# simple_upper_df = simple_df.copy() \
#             .assign(string=lambda df: df['string'].str.upper())

# upper_train_df = pd.concat([upper_df,simple_upper_df]) \
#                 .drop_duplicates(subset=['string'])

# upper_train_df.to_csv(data_dir/'training_data'/'augmented_train_upper_case.csv')


# mixed_train_df = pd.concat([gold_df,upper_df,simple_df,simple_upper_df]) \
#                 .drop_duplicates(subset=['string'])

# mixed_train_df.to_csv(data_dir/'training_data'/'augmented_train_mixed_case.csv')


# gold_mixed = nama.read_csv(data_dir/'training_data'/'combined_train.csv')

# gold_mixed.matches('Microsoft')


# gold_upper = nama.read_csv(data_dir/'training_data'/'combined_train_upper_case.csv')

# gold_upper.matches('MICROSOFT')
