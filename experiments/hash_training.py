import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, confusion_df, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity
from nama.strings import simplify_corp


for training_set in ['canlobby_client_names','opensecrets_client_names']:
    print(f'\n{training_set}')

    gold = nama.read_csv(nama.root_dir/'training'/'data'/f'{training_set}.csv')

    train = nama.Matcher(gold.strings()).unite(simplify_corp)

    sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
    sim.to('cuda:2')

    history_df = sim.train(train,
                            use_counts=False,
                            augment=False,
                            transformer_lr=1e-4,
                            score_lr=10,
                            max_epochs=1,
                            batch_size=64,
                            verbose=True,
                            restore_best=False,
                            early_stopping=False,
                            max_grad_norm=1
                           )

    pred = sim.predict(gold.strings())
    sim.to('cpu')

    print('training score:')
    print(score_predicted(train,gold))

    print('Predicted score:')
    print(score_predicted(pred,gold))

    print('Combined score:')
    print(score_predicted(train.unite(pred),gold))






    break


suggested_df = sim.suggested_matches_df(train,min_score=0.5)


suggested_df.sample(50).sort_values('score',ascending=False)



suggested_df['impact'].sum()



suggested_df.head(50)


pred.to_df().head(50)
