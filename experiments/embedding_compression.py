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

gold = nama.read_csv(nama.root_dir/'training'/'data'/'opensecrets_train.csv')

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

for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=1)):
    for d in [8,16]:  # [128,256,512,None]:
        for model_size in ['base','large']:
            print(f'{fold=} {d=} {model_size=}')

            sim = EmbeddingSimilarity(prompt='Organization: ',
                                        model_name=f'roberta-{model_size}',
                                        d=d)
            sim.to('cuda:0')

            history_df = sim.train(train,verbose=True,**train_kwargs)

            for half in False,True:
                if half:
                    sim.half()

                test_embeddings = sim(test)

                for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
                    pred = test_embeddings.predict_matcher(threshold=threshold,progress_bar=False)

                    scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

                    scores.update(train_kwargs)

                    scores['fold'] = fold
                    scores['threshold'] = threshold
                    scores['d'] = d
                    scores['half'] = half
                    scores['model_size'] = model_size

                    results.append(scores)

            sim.to('cpu')

    break


results_df = pd.DataFrame(results)

results_df.loc[results_df['d'].isnull() & (results_df['model_size'] == 'base'),'d'] = 768
results_df.loc[results_df['d'].isnull() & (results_df['model_size'] == 'large'),'d'] = 1024


results_df.to_csv(nama.root_dir / 'experiments' / 'embedding_compression_results.csv',index=False)


run_cols = ['model_size','d','half']

mean_results_df = results_df.groupby(run_cols+['threshold']).mean().reset_index()
mean_results_df = mean_results_df[mean_results_df['F1'] > 0]
# mean_results_df = mean_results_df[mean_results_df['d'].isin([400,'None'])]

ax = plt.subplot()
for run_vals, df in mean_results_df.groupby(run_cols):
    df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
ax.set_xlim(0,1)
ax.set_ylim(0,1.05)
plt.show()

ax = plt.subplot()
for run_vals, df in mean_results_df.groupby(run_cols):
    df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.show()


mean_results_df.groupby(run_cols)['F1'].quantile(0.75)


q = 0.75
df = mean_results_df[mean_results_df['half']].copy()
df['q'] = df.groupby(run_cols)['F1'].transform(lambda x: x.quantile(q))
df = df \
        [df['F1'] >= df['q']] \
        .groupby(['model_size','d']) \
        [['F1','threshold']] \
        .mean() \
        .reset_index()

fig,ax = plt.subplots()
for model_size in ['base','large']:
    ax.scatter('d','F1',label=model_size,data=df[df['model_size'] == model_size])
ax.set_xscale('log',base=2)
ax.set_ylim(None,1)
