from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, confusion_df, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity

from regcomments.config import data_dir


org_strings_df = pd.read_csv(Path(data_dir)/'org_extraction'/'extracted_org_strings.csv')

org_strings_df = org_strings_df[org_strings_df['org_strings'].notnull()]

org_strings_df.sample(10)


print(org_strings_df.loc[8557265,'source_string'])

print(org_strings_df.loc[6059162,'source_string'])

# gold = nama.read_csv(data_dir/'training_data'/'canlobby_client_names.csv')
#
# groups = sorted(gold.groups.keys())
#
# all_train,test = split_on_groups(gold,0.8,seed=1)
#
# raw_train = nama.Matcher(all_train.strings())
#
#
# sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
# sim.to('cuda:1')
# history_df = sim.train(all_train,use_counts=False,score_lr=1e-6,transformer_lr=1e-6,projection_lr=1e-6,val_seed=1,augment=False,batch_size=16,max_epochs=20)
#
# sim.test(test,gold,use_counts=False)
#
#
# del sim
#
#
#
#
# aug_train = augment_matcher(raw_train,n=3)
#
#
# sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
# sim.to('cuda:1')
# history_df = sim.train(aug_train,use_counts=False,score_lr=1e-6,transformer_lr=1e-6,projection_lr=1e-4,val_seed=1,augment=False,batch_size=16,max_epochs=20)
#
#
# sim.test(test,gold,use_counts=False)
#
#
#
#
# sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
# sim.to('cuda:1')
# history_df = sim.train(aug_train,use_counts=False,score_lr=1e-5,transformer_lr=0,val_seed=1,augment=False,batch_size=16,max_epochs=10)
#
# sim.test(test,gold,use_counts=False)
