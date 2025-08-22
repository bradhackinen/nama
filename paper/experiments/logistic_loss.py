import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups
from nama import SimilarityModel, load_similarity_model
from nama.embedding_similarity.scoring_model import LogisticSimilarityScore

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down sample for testing
gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':0.01,
    'use_counts':False,
    'batch_size':6,
    'add_upper':True,
    }

pretrain_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':0,
    'projection_lr':0,
    'score_lr':0.01,
    'use_counts':False,
    'batch_size':6,
    'add_upper':True,
    }

exp_sim_hparams = {
    'd':128,
    'model_name':'roberta-base',
    'alpha':20,
    }

logistic_sim_hparams = {
    'd':128,
    'score_class':LogisticSimilarityScore,
    'loc':0.9,
    'scale':0.1,
    }

sim = SimilarityModel(prompt='Organization: ',**logistic_sim_hparams)
sim.to('cuda:3')


history_df = sim.train(train,verbose=True,**train_kwargs)

history_df.plot(y='global_loss',logy=True)

history_df.plot(y='score_loc')
history_df.plot(y='score_scale')



sim.score_model

history_df = sim.train(train,verbose=True,**train_kwargs)

history_df.plot(y='global_loss',logy=True)

history_df.plot(y=['score_alpha','score_beta'])

history_df.plot(y='global_mean_cos')

sim.score_model.score_to_cos(0.05)

results = []
test_embeddings = sim.embed(test)
for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):

    pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

    scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

    scores.update(train_kwargs)

    # scores['model'] = model_name
    # scores['half'] = half
    scores['threshold'] = threshold

    results.append(scores)

results_df = pd.DataFrame(results)

results_df.plot('threshold','F1')



results_df['F1'].max()

results_df.query('threshold == 0.5')['F1']

ax = plt.subplot()
for run_vals, df in results_df.groupby(run_cols):
    df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
plt.show()


results_df.groupby(run_cols)['F1'].quantile(0.8)