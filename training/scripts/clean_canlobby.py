import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity,load
from nama.strings import simplify_corp

train_kwargs = {
                    'max_epochs': 1,
                    'warmup_frac': 0.1,
                    'calibration_frac': 0,
                    'transformer_lr':1e-5,
                    'score_lr':10,
                    'use_counts':True,
                    'batch_size':8,
                    }


if os.path.isfile(nama.root_dir/'training'/'data'/'canlobby_train.csv'):
    # Load the current combined canlobby matcher
    canlobby = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')
else:
    # ...or start a new one from scratch
    canlobby = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_clients_manual.csv')


model_file = nama.root_dir/'models'/'canlobby.pt'

if os.path.isfile(model_file):
    # Load the current trained model
    sim = load(model_file)
    sim.to('cuda:1')

else:
    # Otherwise, train a new one from scratch
    sim = EmbeddingSimilarity(prompt='Organization: ')
    sim.to('cuda:1')

    history_df = sim.train(canlobby,**train_kwargs)


# Build updated matcher using manual pairs and similarity model
matcher = canlobby.copy()

manual_df = pd.concat([
                    pd.read_csv(nama.root_dir/'training'/'data'/f'canlobby_{x}.csv')
                    for x in ['fn','fp']]) \
                .dropna() \
                .assign(is_match=lambda x: x['is_match'] == 'y')

# Unite the matched pairs
matcher = matcher.unite(manual_df[manual_df['is_match']][['string0','string1']].values)

# Separate the false negative pairs, automatically grouping strings according to similarity
separate_df = manual_df[~manual_df['is_match']].copy()
for i in [0,1]:
    separate_df[f'group{i}'] = [matcher[s] for s in separate_df[f'string{i}']]
separate_df = separate_df[separate_df['group0'] == separate_df['group1']]

for g,group_pairs_df in separate_df.groupby('group0'):
    group_strings = matcher.groups[g]
    group_matcher = sim.predict(group_strings,threshold=0,constraints=group_pairs_df,progress_bar=False)

    matcher = matcher \
                .split(group_strings) \
                .unite(group_matcher)

matcher.to_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')


# Train the similarity model on the updated matcher
sim.to('cpu')
sim = EmbeddingSimilarity(prompt='Organization: ')
sim.to('cuda:0')

history_df = sim.train(matcher,**train_kwargs)

# Save the updated similarity model
sim.save(model_file)


# Look for false negative pairs
# (from inspection, it appears that unmatched pairs with score>0.9 are almost always true matches)
fn_df = sim.top_scored_pairs(matcher,n=1000,is_match=False,min_score=0.6,sort_by=['impact','score'],ascending=False,
                                skip_pairs=manual_df[['string0','string1']].values)

fn_df[['string0','string1']] \
        .assign(is_match='') \
        .to_csv(nama.root_dir/'_review'/'canlobby_fn.csv',index=False)


# Look for potential false positive pairs (caused by name changes, subsidiaries, client/registrant relationships, etc)
# (from inspection, it appears that matched pairs with score<0.1 are almost never true matches)
fp_df = sim.top_scored_pairs(matcher,n=1000,is_match=True,max_score=0.4,sort_by=['impact','score'],ascending=[False,True],
                                skip_pairs=manual_df[['string0','string1']].values)

fp_df[['string0','string1']] \
        .assign(is_match='') \
        .to_csv(nama.root_dir/'_review'/'canlobby_fp.csv',index=False)





# Test trained similarity model on Canadian lobbying clients
test = can_clients
results = []
for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):

    pred = sim.predict(test,threshold=threshold,progress_bar=False)

    for unite_simplified in False,True:
        if unite_simplified:
            pred = pred.unite(simplify_corp)

        scores = score_predicted(pred,test,use_counts=True)

        scores['threshold'] = threshold
        scores['unite_simplified'] = unite_simplified

        results.append(scores)

results_df = pd.DataFrame(results)



import matplotlib.pyplot as plt

run_cols = ['unite_simplified']
run_vals = ['unite_simplified']

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


results_df.groupby(run_cols)['F1'].max()
