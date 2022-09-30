from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, confusion_df, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity


results = []
for training_set in ['canlobby_client_names','opensecrets_client_names']:
    for seed in range(1,11):
        print(f'\n Testing {training_set}, seed={seed}')

        gold = nama.read_csv(nama.root_dir/'training'/'data'/f'{training_set}.csv')

        train,test = split_on_groups(gold,0.8,seed=seed)

        sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
        sim.to('cuda:1')

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
        sim.to('cpu')

        for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
            pred = sim.predict(test.strings(),threshold=threshold)
            scores = score_predicted(pred,gold)
            scores['training_set'] = training_set
            scores['seed'] = seed
            scores['threshold'] = threshold
            results.append(scores)


results_df = pd.DataFrame(results)

results_df.to_csv(nama.root_dir/'experiments'/'gold_train_results.csv')

results_df[results_df['F1'] >0 ] \
            .groupby(['training_set','threshold']) \
            .mean() \
            .reset_index() \
            .plot('precision','recall')

results_df[results_df['F1'] >0 ] \
            .groupby(['training_set','threshold']) \
            .mean() \
            .reset_index() \
            .groupby('training_set') \
            .plot('threshold',['precision','recall','F1'])

results_df[results_df['F1'] >0 ] \
            .assign(log_errors=lambda x: np.log10(x['FP'] + x['FN'])) \
            .groupby(['training_set','threshold']) \
            .mean() \
            .reset_index() \
            .groupby('training_set') \
            .plot('threshold','log_errors')

results_df[results_df['accuracy'] >0.9 ] \
            .groupby(['training_set','threshold']) \
            ['accuracy'] \
            .mean() \
            .tail(50)
#
# results_df[results_df['threshold']>0.95][['seed','threshold','precision','recall']]
#
# suggested_df = sim.suggested_matches_df(test,min_score=0.5)
#
#
# suggested_df.sample(50).sort_values('score',ascending=False)
#
#
# suggested_df['impact'].sum()
#
#
#
# suggested_df.head(50)
#
#
# pred.to_df().head(50)
#
# seed = 7
#
# pred = sim.predict(test.strings(),threshold=1.0)
#
# score_predicted(pred,gold)
