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
                    'calibration_frac': 0,
                    'transformer_lr':1e-5,
                    'score_lr':10,
                    'use_counts':True,
                    'batch_size':8,
                    'regularization':0,
                    'norm':1.0,
                    'alpha_decay':1e-6,
                    }

for regularization in [0.0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
    train_kwargs['regularization'] = regularization

    for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=2)):

        print(f'{regularization=}, {fold=}')

        sim = EmbeddingSimilarityModel(prompt='Organization: ',
                                        model_name='roberta-large')
        sim.to('cuda:3')

        history_df = sim.train(train,verbose=True,**train_kwargs)

        history_df.assign(log_loss=np.log(history_df['global_loss'])) \
                    ['log_loss'] \
                    .rolling(100).mean() \
                    .plot()
        plt.show()

        # Cache embeddings for repeated prediction
        test_embeddings = sim(test)

        for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):
            pred = test_embeddings.predict_matcher(threshold=threshold,progress_bar=False)

            scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

            scores.update(train_kwargs)

            scores['fold'] = fold
            scores['threshold'] = threshold
            scores['alpha'] = sim.score_model.alpha.item()

            results.append(scores)

        # break

        sim.to('cpu')

results_df = pd.DataFrame(results)

results_df.to_csv(nama.root_dir/'experiments'/'regularization_results.csv')

run_cols = ['regularization']

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


results_df.groupby(run_cols)['F1'].quantile(0.5)
