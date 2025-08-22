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
# gold,_ = split_on_groups(gold,0.1,seed=1)

big_test,sample = split_on_groups(gold,0.5,seed=1)

train,_ = split_on_groups(sample,0.5,seed=1)

med_test,_ = split_on_groups(big_test,0.5,seed=1)
small_test,_ = split_on_groups(big_test,0.25,seed=1)


train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':1,
    'use_counts':False,
    'batch_size':8,
    'top_k':4,
    'add_upper':True,
    }

hparams = {
    'd':None,
    'model_name':'roberta-base',
    'scale':50
    }


sim = SimilarityModel(prompt='Organization: ',**hparams)
sim.to('cuda:2')

history_df = sim.train(train,verbose=True,**train_kwargs)

sim.save(Path(data_dir)/'models'/'nama_score_adjustment_train.bin')
history_df.to_csv(Path(data_dir)/'models'/'nama_score_adjustment_train_history.csv')

history_df.plot(y='global_loss',logy=True)

# history_df.plot(y='score_loc')
# history_df.plot(y='score_scale')


sim.half()


results = []
for test_name,test_matches in [('small_test',small_test),('med_test',med_test),('big_test',big_test)]:
    test_embeddings = sim.embed(test_matches)
    for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

        pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

        scores = score_predicted(pred,test_matches,use_counts=train_kwargs['use_counts'])

        scores.update(train_kwargs)

        # scores['model'] = model_name
        # scores['half'] = half
        scores['n_pred_groups'] = len(pred.groups)
        scores['n_test_groups'] = len(test_matches.groups)
        scores['threshold'] = threshold
        scores['test'] = test_name

        results.append(scores)

# Compute corrected thresholds
# small_correction_factor = 0.5
results_df = pd.DataFrame(results)



min_loss_results = {}
for test_name,test_matches in [('small_test',small_test),('med_test',med_test),('big_test',big_test)]:
    test_embeddings = sim.embed(test_matches)
    pred = test_embeddings.unite_similar(threshold=0.2,loss_threshold=0.5,progress_bar=True)
    scores = score_predicted(pred,test_matches,use_counts=False)

    min_loss_results[test_name] = scores

min_loss_results = {d['test']:d for d in min_loss_results}


fig,ax = plt.subplots()
for i,(test_name,df) in enumerate(results_df.groupby('test')):
    df.plot('threshold','F1',label=test_name,ax=ax,color=f'C{i}')

    ax.axhline(min_loss_results[test_name]['F1'],ls=':',color=f'C{i}')














fig,ax = plt.subplots()
for test_name,df in results_df.groupby('test'):
    df.plot('threshold','n_pred_groups',label=test_name,ax=ax)
    ax.axhline(df['n_test_groups'].iloc[0],ls=':',color='k')



results_df['correction_factor'] = (results_df['n_pred_groups']/len(train.groups))


t = results_df['threshold']
results_df['corrected_threshold'] = t/(t + results_df['correction_factor']*(1-t))

fig,ax = plt.subplots()
for i,(test_name,df) in enumerate(results_df.groupby('test')):
    df.plot('threshold','n_pred_groups',label=test_name,ax=ax,c=f'C{i}')
    df.plot('corrected_threshold','n_pred_groups',ax=ax,ls=':',c=f'C{i}')

fig,ax = plt.subplots()
for i,(test_name,df) in enumerate(results_df.groupby('test')):
    df.plot('threshold','corrected_threshold',label=test_name,ax=ax,c=f'C{i}')


# gamma = 0.3
# results_df.loc[results_df['test']=='small_test','correction_factor'] = (len(small_test.groups)/len(train.groups))**gamma
# results_df.loc[results_df['test']=='med_test','correction_factor'] = (len(med_test.groups)/len(train.groups))**gamma
# results_df.loc[results_df['test']=='big_test','correction_factor'] = (len(big_test.groups)/len(train.groups))**gamma

gamma = 0.3
results_df['correction_factor'] = (results_df['n_pred_groups']/len(train.groups))**gamma

t = results_df['threshold']
results_df['corrected_threshold'] = t/(t + results_df['correction_factor']*(1-t))
# results_df['corrected_threshold'] = t**results_df['correction_factor']


fig,ax = plt.subplots()
for test_name,df in results_df.groupby('test'):
    # df.plot('threshold','F1',label=test_name,ax=ax)
    df.plot('corrected_threshold','F1',label=f'{test_name} corrected',ax=ax,ls=':')











results = []
for test_name,test_matches in [('small_test',small_test),('med_test',med_test),('big_test',big_test)]:
    test_embeddings = sim.embed(test_matches)
    for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

        pred = test_embeddings.unite_similar(threshold=threshold,group_threshold=threshold,progress_bar=False)

        scores = score_predicted(pred,test_matches,use_counts=train_kwargs['use_counts'])

        scores.update(train_kwargs)

        # scores['model'] = model_name
        # scores['half'] = half
        scores['n_pred_groups'] = len(pred.groups)
        scores['n_test_groups'] = len(test_matches.groups)
        scores['threshold'] = threshold
        scores['test'] = test_name

        results.append(scores)

# Compute corrected thresholds
# small_correction_factor = 0.5


g_results_df = pd.DataFrame(results)

fig,ax = plt.subplots()
for test_name,df in g_results_df.groupby('test'):
    df.plot('threshold','F1',label=test_name,ax=ax)




gamma = 0.3
g_results_df.loc[g_results_df['test']=='small_test','correction_factor'] = (len(small_test.groups)/len(train.groups))**gamma
g_results_df.loc[g_results_df['test']=='med_test','correction_factor'] = (len(med_test.groups)/len(train.groups))**gamma
g_results_df.loc[g_results_df['test']=='big_test','correction_factor'] = (len(big_test.groups)/len(train.groups))**gamma


t = g_results_df['threshold']
g_results_df['corrected_threshold'] = t/(t + g_results_df['correction_factor']*(1-t))


fig,ax = plt.subplots()
for test_name,df in g_results_df.groupby('test'):
    # df.plot('threshold','F1',label=test_name,ax=ax)
    df.plot('corrected_threshold','F1',label=f'{test_name} corrected',ax=ax,ls=':')

