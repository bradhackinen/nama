import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarityModel

gold = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')


results = []

train_kwargs = {
                    'max_epochs': 1,
                    'warmup_frac': 0.1,
                    'transformer_lr':1e-5,
                    'score_lr':10,
                    'use_counts':True,
                    'batch_size':8,
                    'score_decay':0,
                    'dispersion':0,
                    'alpha':50,
                    'model_name':'roberta-base',
                    'd':256,
                    'upper_case':True
                    }


for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=1)):

    sim = EmbeddingSimilarityModel(prompt='Organization: ',**train_kwargs)
    sim.to('cuda:1')

    history_df = sim.train(train,verbose=True,**train_kwargs)

    # Cache embeddings for repeated prediction
    test_embeddings = sim.embed(test)

    for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
        pred = test_embeddings.predict(threshold=threshold,progress_bar=False)

        scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

        scores.update(train_kwargs)

        scores['fold'] = fold
        scores['threshold'] = threshold

        results.append(scores)

    break

    sim.to('cpu')

results_df = pd.DataFrame(results)


history_df.assign(log_loss=np.log(history_df['global_loss'])) \
            ['log_loss'] \
            .rolling(100).mean() \
            .plot(title='Log Loss')
plt.show()

for c in 'batch_pos_target','global_mean_cos','score_alpha':
    history_df[c].rolling(100).mean().plot(title=c)
    plt.show()


run_cols = list(train_kwargs.keys())

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


mean_results_df.groupby(run_cols)['F1'].quantile(0.8)
