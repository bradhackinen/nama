import numpy as np
import pandas as pd
from tqdm import tqdm

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity


gold = nama.read_csv(data_dir/'training_data'/'canlobby_client_names.csv')


results = []
for epochs in [1,2,3]:
    for warmup_frac in [0.1,0.5]:
        for batch_size in [8,64]:
            for lr in [1e-5,5e-5,1e-4]:
                for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=1)):

                    print(f'\nTesting {epochs=}, {warmup_frac=}, {batch_size=}, {lr=}, {fold=}')

                    sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
                    sim.to('cuda:2')
                    history_df = sim.train(train,
                                            use_counts=True,
                                            max_epochs=epochs,
                                            warmup_frac=warmup_frac,
                                            calibration_frac=0,
                                            batch_size=batch_size,
                                            transformer_lr=lr,
                                            )

                    for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
                        pred = sim.predict(test,threshold=threshold,progress_bar=False)

                        scores = score_predicted(pred,test,use_counts=True)

                        scores['fold'] = fold
                        scores['threshold'] = threshold
                        scores['epochs'] = epochs
                        scores['warmup_frac'] = warmup_frac
                        scores['batch_size'] = batch_size
                        scores['lr'] = lr

                        results.append(scores)

                    sim.to('cpu')


results_df = pd.DataFrame(results)

results_df.to_csv(data_dir/'experiments'/'scheduler_results.csv')


import matplotlib.pyplot as plt

run_cols = ['epochs','warmup_frac','batch_size','lr']

mean_results_df = results_df.groupby(run_cols + ['threshold']).mean().reset_index()

ax = plt.subplot()
for run_vals, df in mean_results_df.groupby(run_cols):
    df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=1)
plt.show()

ax = plt.subplot()
for run_vals, df in mean_results_df.groupby(run_cols):
    df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=1)
plt.show()

# results_df.plot('recall','precision')
# results_df.plot('threshold','F1')

#
mean_results_df \
    .sort_values('F1',ascending=False) \
    .groupby(run_cols) \
    .head(10) \
    .groupby(run_cols) \
    .mean() \
    .sort_values('F1',ascending=False) \
    [['threshold','F1']]


for c in run_cols:
    df = mean_results_df \
        .sort_values('F1',ascending=False) \
        .groupby(run_cols) \
        .head(5) \
        .groupby(c) \
        ['F1'].mean() \

    print(df)
