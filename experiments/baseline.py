import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarityModel

gold = nama.read_csv(data_dir/'training_data'/'combined_train.csv')

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

for fold,(train,test) in enumerate(kfold_on_groups(gold,k=5,seed=1)):

    sim = EmbeddingSimilarityModel(prompt='Organization: ',
                                    model_name='roberta-base',
                                    amp=True,
                                    d=256,
                                    **train_kwargs)
    sim.to('cuda:0')

    history_df = sim.train(train,verbose=True,amp=True,**train_kwargs)

    # Plot global loss history
    history_df.assign(log_loss=np.log(history_df['global_loss'])) \
                ['log_loss'] \
                .rolling(100).mean() \
                .plot()
    plt.show()

    for c in 'global_mean_cos','score_alpha','batch_pos_target':
        history_df[c].rolling(100).mean().plot(title=c)
        plt.show()

    for half in False,True:
        if half:
            sim.half()

        # Cache embeddings for repeated prediction
        test_embeddings = sim.embed(test)

        for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):
            pred = test_embeddings.predict(threshold=threshold,progress_bar=False)

            scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

            scores.update(train_kwargs)

            scores['fold'] = fold
            scores['half'] = half
            scores['threshold'] = threshold
            scores['alpha'] = sim.score_model.alpha.item()

            results.append(scores)

    break

    sim.to('cpu')

results_df = pd.DataFrame(results)

# Plot average F1 on hold-out test sets
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


mean_results_df.groupby(run_cols)['F1'].quantile(0.8)


# from nama.embedding_similarity import load_similarity_model
# sim2 = load_similarity_model(data_dir/'models'/'nama_large.bin')
# sim2.to('cuda:0')
#


# # Cache embeddings for repeated prediction
# test_embeddings = sim2.embed(test)
#
# for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):
#     pred = test_embeddings.predict(threshold=threshold,progress_bar=False)
#
#     scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])
#
#     scores.update(train_kwargs)
#
#     scores['fold'] = fold
#     scores['half'] = half
#     scores['threshold'] = threshold
#     scores['alpha'] = sim.score_model.alpha.item()
#
#     results.append(scores)



#
# V = sim.embed(['Costco Wholesale Corporation','COSTCO WHOLESALE CORPORATION']).V
# sim.score_model(V@V.T)
#
# V = sim.embed(['Costco Wholesale Corporation','COSTCO WHOLESALE CORP']).V
# sim.score_model(V@V.T)
#
#
#
# V = sim.embed(['Apple, Inc.','APPLE, INC.']).V
# sim.score_model(V@V.T)
#
# V = sim.embed(['MICROSOFT','Microsoft']).V
# sim.score_model(V@V.T)
