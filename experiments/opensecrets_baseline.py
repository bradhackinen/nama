import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted    # split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarityModel
# from nama.strings import simplify_corp

opensecrets = nama.read_csv(nama.root_dir/'training'/'data'/'opensecrets_train.csv')
canlobby = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')

results = []
train_kwargs = {
                    'max_epochs': 1,
                    'warmup_frac': 0.1,
                    'calibration_frac': 0,
                    'transformer_lr':1e-5,
                    'score_lr':1,
                    'use_counts':True,
                    'batch_size':8,
                    'score_decay':0,
                    'dispersion':0,
                    'add_upper':True,
                    }

for model_name in 'roberta-base','roberta-large':
    for transformer_lr in [1e-5,0]:
        train_kwargs['transformer_lr'] = transformer_lr

        sim = EmbeddingSimilarityModel(prompt='Organization: ',model_name=model_name,d=256)
        sim.to('cuda:2')

        history_df = sim.train(opensecrets,verbose=True,**train_kwargs)

        for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
            for group_threshold in None,threshold:
                pred = sim.predict(canlobby,threshold=threshold,group_threshold=group_threshold,progress_bar=False)

                scores = score_predicted(pred,canlobby,use_counts=train_kwargs['use_counts'])

                scores.update(train_kwargs)

                scores['model'] = model_name
                scores['threshold'] = threshold
                scores['group_threshold'] = group_threshold is not None

                results.append(scores)

        sim.to('cpu')

        results_df = pd.DataFrame(results)

results_df.to_csv(nama.root_dir/'experiments'/'opensecrets_to_canlobby_results.csv')

run_cols = ['model','transformer_lr','group_threshold']

ax = plt.subplot()
for run_vals, df in results_df.groupby(run_cols):
    df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
plt.show()

ax = plt.subplot()
for run_vals, df in results_df.groupby(run_cols):
    df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
plt.show()


results_df.groupby(run_cols)['F1'].quantile(0.9)
