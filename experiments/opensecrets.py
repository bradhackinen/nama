import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, confusion_df, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity
from nama.strings import simplify_corp

gold = nama.read_csv(nama.root_dir/'training'/'data'/'opensecrets_client_names.csv')

train = nama.Matcher(gold.strings()).unite(simplify_corp)

sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
sim.to('cuda:1')
history_df = sim.train(train,
                        use_counts=False,
                        augment=False,
                        transformer_lr=1e-5,
                        score_lr=1,
                        max_epochs=1,
                        batch_size=64,
                        verbose=True,
                        restore_best=True,
                        early_stopping=True,
                        max_grad_norm=1
                       )

pred = sim.predict(gold.strings())


print('training score:')
print(score_predicted(train,gold))

print('Predicted score:')
print(score_predicted(pred,gold))

print('Combined score:')
print(score_predicted(train.unite(pred),gold))


suggested_df = sim.suggested_matches_df(train)

se

suggested_df.head(50)



sim.to('cpu')
