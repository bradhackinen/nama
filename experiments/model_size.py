import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity
from nama.strings import simplify_corp

gold = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')

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
    for model_size in ['base','large']:

        sim = EmbeddingSimilarity(prompt='Organization: ',
                                    model_name=f'roberta-{model_size}',
                                    d=None)
        sim.to('cuda:3')

        history_df = sim.train(train,verbose=True,**train_kwargs)

        for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):
            pred = sim.predict(test,threshold=threshold,progress_bar=False,pairwise=False)

            scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

            scores.update(train_kwargs)

            scores['fold'] = fold
            scores['threshold'] = threshold
            scores['model_size'] = model_size

            results.append(scores)

        sim.to('cpu')

    # break

results_df = pd.DataFrame(results)



run_cols = ['model_size']

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


mean_results_df.groupby(run_cols)['F1'].max()

sim.embedding_model.V_cached.size()
