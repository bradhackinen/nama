import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity
from nama.strings import simplify_corp

gold = nama.read_csv(data_dir/'training_data'/'canlobby_clients_manual.csv')

results = []

train_kwargs = {
                    'max_epochs': 1,
                    'warmup_frac': 0.1,
                    'calibration_frac': 0,
                    'transformer_lr':1e-5,
                    'score_lr':10,
                    'use_counts':True,
                    'batch_size':8,
                    }

for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=2)):

    sim = EmbeddingSimilarity(prompt='Organization: ')
    sim.to('cuda:2')

    history_df = sim.train(train,verbose=True,**train_kwargs)

    for half in False,True:
        if half:
            sim.half()

        for threshold in tqdm(np.linspace(0,1,11),desc='scoring'):
            pred = sim.predict(test,threshold=threshold,progress_bar=False)

            scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

            scores.update(train_kwargs)

            scores['fold'] = fold
            scores['threshold'] = threshold
            scores['half'] = half

            results.append(scores)

        sim.embedding_model.clear_cache()

    sim.to('cpu')

results_df = pd.DataFrame(results)




import matplotlib.pyplot as plt

run_cols = ['half']

mean_results_df = results_df.groupby(run_cols+['threshold']).mean().reset_index()

ax = plt.subplot()
for run_vals, df in mean_results_df.groupby(run_cols):
    df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
plt.show()

ax = plt.subplot()
for run_vals, df in mean_results_df.groupby(run_cols):
    df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
plt.show()


# Using half() has a negligable effect on F1 score
results_df.groupby(['fold','threshold'])['F1'].std().describe()

results_df.groupby('half')['F1'].mean()
