import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarityModel

canlobby = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')
opensecrets = nama.read_csv(nama.root_dir/'training'/'data'/'opensecrets_train.csv')

gold = canlobby + opensecrets

results = []

train_kwargs = {
                    'max_epochs': 1,
                    'warmup_frac': 0.1,
                    'calibration_frac': 0,
                    'transformer_lr':1e-5,
                    'score_lr':10,
                    'use_counts':True,
                    'batch_size':8,
                    'score_decay':1e-6,
                    }

# model_defs = {
#     'nama_base':{'d':64,'model_name':'roberta-base'},
#     'nama_large':{'d':256,'model_name':'roberta-large'}
#     }


for seed in [1]:
    for score_decay in [0,1e-8,1e-7,1e-6,1e-5,1e-4]:
        train_kwargs['score_decay'] = score_decay

        for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=seed)):

            print(f'{seed=}, {score_decay=}, {fold=}')

            sim = EmbeddingSimilarityModel(
                                    prompt='Organization: ',
                                    model_name='roberta-large',
                                    d=256)
            sim.to('cuda:1')

            history_df = sim.train(train,verbose=True,**train_kwargs)

            history_df.assign(log_loss=np.log(history_df['global_loss'])) \
                        ['log_loss'] \
                        .rolling(100).mean() \
                        .plot()
            plt.show()

            for half in False,True:
                if half:
                    sim.half()

                # Cache embeddings for repeated prediction
                test_embeddings = sim(test)

                for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
                    pred = test_embeddings.predict_matcher(threshold=threshold,progress_bar=False)

                    scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

                    scores.update(train_kwargs)

                    scores['seed'] = seed
                    scores['fold'] = fold
                    scores['threshold'] = threshold
                    scores['half'] = half
                    scores['alpha'] = sim.score_model.alpha.item()

                    results.append(scores)

            # break

            sim.to('cpu')

results_df = pd.DataFrame(results)

results_df.to_csv(nama.root_dir/'experiments'/'score_decay_results.csv')

run_cols = ['score_decay','half']

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

results_df.groupby(run_cols)['F1'].quantile(0.8)

# results_df.groupby(run_cols)[['alpha','F1']].quantile(0.9).plot.scatter()
