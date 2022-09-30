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


if os.path.isfile(data_dir/'training_data'/'canlobby_train.csv'):
    # Load the current combined canlobby matcher
    canlobby = nama.read_csv(data_dir/'training_data'/'canlobby_train.csv')
else:
    # ...or start a new one from scratch
    canlobby = nama.read_csv(data_dir/'training_data'/'canlobby_clients_manual.csv')


model_file = data_dir/'models'/'canlobby.pt'

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
                    pd.read_csv(data_dir/'training_data'/f'canlobby_{x}.csv')
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

matcher.to_csv(data_dir/'training_data'/'canlobby_train.csv')


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
        .to_csv(data_dir/'_review'/'canlobby_fn.csv',index=False)


# Look for potential false positive pairs (caused by name changes, subsidiaries, client/registrant relationships, etc)
# (from inspection, it appears that matched pairs with score<0.1 are almost never true matches)
fp_df = sim.top_scored_pairs(matcher,n=1000,is_match=True,max_score=0.4,sort_by=['impact','score'],ascending=[False,True],
                                skip_pairs=manual_df[['string0','string1']].values)

fp_df[['string0','string1']] \
        .assign(is_match='') \
        .to_csv(data_dir/'_review'/'canlobby_fp.csv',index=False)





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













# fp_df.tail(50)
# fp_df.sample(50).sort_values('score',ascending=True)
#
#
# # Load and apply maual corrections to the matcher
# matcher = (opensecrets_clients + opensecrets_registrants).unite(simplify_corp)
#
# manual_df = pd.concat([
#                     pd.read_csv(data_dir/'training_data'/f'opensecrets_{x}.csv')
#                     for x in ['fn','fp']]) \
#                 .assign(is_match=lambda x: x['is_match'] == 'y')
#
# # Unite the matched pairs
# matcher = matcher.unite(manual_df[manual_df['is_match']][['string0','string1']].values)
#
# # Separate the false negative pairs, automatically grouping strings according to similarity
# separate_df = manual_df[~manual_df['is_match']].copy()
# for i in [0,1]:
#     separate_df[f'group{i}'] = [matcher[s] for s in separate_df[f'string{i}']]
# separate_df = separate_df[separate_df['group0'] == separate_df['group1']]
#
# for g,group_pairs_df in separate_df.groupby('group0'):
#     group_strings = matcher.groups[g]
#     group_matcher = sim.predict(group_strings,threshold=0,constraints=group_pairs_df,progress_bar=False)
#
#     matcher = matcher \
#                 .split(group_strings) \
#                 .unite(group_matcher)
#
# matcher.to_csv(data_dir/'training_data'/'opensecrets_train.csv')
#
#
#
#
#
#
#
#
#
#
#
# constraints = group_pairs_df
# input = group_strings
# matcher = nama.Matcher(input)
# self = sim
# kwargs = {}
#
# sim.predict(group_strings,threshold=0,constraints=group_pairs_df,progress_bar=False)
#
# group_matcher.to_df()
#
# group_matcher.groups
# matcher['Yellow Corp'],matcher['Roadway Express']
#
# matcher.groups
#
#
# set(matcher.groups[g])
#
#
# # Re-train the embedding model on the modified matcher
# sim.to('cpu')
#
# sim = EmbeddingSimilarity(prompt='Organization: ')
# sim.to('cuda:0')
# history_df = sim.train(matcher,**train_kwargs)
#
#
#
#
# test = can_clients
# results = []
# for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):
#
#     pred = sim.predict(test,threshold=threshold,progress_bar=False)
#
#     for unite_simplified in False,True:
#         if unite_simplified:
#             pred = pred.unite(simplify_corp)
#
#         scores = score_predicted(pred,test,use_counts=True)
#
#         scores['threshold'] = threshold
#         scores['unite_simplified'] = unite_simplified
#
#         results.append(scores)
#
# results_df = pd.DataFrame(results)
#
#
# # results_df.groupby('unite_simplified').plot('recall','precision')
# # results_df.groupby('unite_simplified').plot('threshold','F1')
#
#
#
# run_cols = ['unite_simplified']
# run_vals = ['unite_simplified']
#
#
# results_df.groupby('unite_simplified')['F1'].max()
#
# import matplotlib.pyplot as plt
#
# ax = plt.subplot()
# for run_vals, df in results_df.groupby(run_cols):
#     df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, ncol=1)
# plt.show()
#
# ax = plt.subplot()
# for run_vals, df in results_df.groupby(run_cols):
#     df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, ncol=1)
# plt.show()
#
# ax = plt.subplot()
# for run_vals, df in results_df.groupby(run_cols):
#     df.plot('threshold',y=['recall','precision'],ax=ax,label=[f'{run_vals=}'])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, ncol=1)
# plt.show()
#
# results_df.groupby(run_cols)['F1'].max()
#
#
#
# fn_df = sim.top_scored_pairs(test,is_match=False,min_score=0.5,sort_by='score',ascending=False)
#
# # fn_df.tail(50)
# # fn_df.sample(50).sort_values('score',ascending=False)
#
# # Look for potential false positive pairs (caused by name changes, subsidiaries, client/registrant relationships, etc)
# # (from inspection, it appears that matched pairs with score<0.1 are almost never true matches)
# fp_df = sim.top_scored_pairs(test,is_match=True,max_score=0.4,sort_by='score',ascending=True)
#
#
# fn_df[['string0','string1']] \
#     .assign(is_match=np.nan) \
#     .to_csv(data_dir/'_review'/'canlobby_manual_pairs.csv',index=False)
#
#
#
# pred = sim.predict(test,threshold=threshold,progress_bar=False)
#
#
# def refine_groups(self,matcher,threshold):
#
#     matcher = matcher.copy()
#
#     for group_strings in tqdm(list(matcher.groups.values()),desc='Refining groups'):
#         group_matcher = nama.Matcher(group_strings)
#
#         distant_df = sim.top_scored_pairs(group_matcher,max_score=threshold,sort_by='score',ascending=True,progress_bar=False)
#
#         if len(distant_df):
#
#             distant_df['is_match'] = False
#
#             group_pred = sim.predict(group_matcher,threshold=threshold,constraints=distant_df,progress_bar=False)
#
#             matcher.split(group_strings,inplace=True)
#             matcher.unite(group_pred,inplace=True)
#
#     return matcher
#
#
#
# pred = sim.predict(test,threshold=0.5,progress_bar=False)
# refine_groups(sim,pred,threshold=0.6)
#
#
# test = can_clients
# results = []
# for threshold in tqdm(np.linspace(0.5,1,11),desc='scoring'):
#
#     pred = sim.predict(test,threshold=threshold,progress_bar=False)
#
#     for refine in False,True:
#         if refine:
#             pred = refine_groups(sim,pred,0.5*(1+threshold))
#
#         scores = score_predicted(pred,test,use_counts=True)
#
#         scores['threshold'] = threshold
#         scores['refine'] = refine
#
#         results.append(scores)
#
# results_df = pd.DataFrame(results)
#
# threshold=0.4
#
#
# run_cols = ['refine']
# run_vals = ['refine']
#
#
# results_df.groupby('refine')['F1'].max()
#
# import matplotlib.pyplot as plt
#
# ax = plt.subplot()
# for run_vals, df in results_df.groupby(run_cols):
#     df.plot('recall','precision',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, ncol=1)
# plt.show()
#
# ax = plt.subplot()
# for run_vals, df in results_df.groupby(run_cols):
#     df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, ncol=1)
# plt.show()
#
#
#
#
#
#
#
# f = data_dir/'models'/'opensecrets.pt'
# sim.save(data_dir/'models'/'opensecrets.pt')
#
# sim = torch.load(f)
#
#
# torch.save(sim,f)
#
# sim2 = EmbeddingSimilarity()
# sim2.load_state_dict(torch.load(f))
#
# sim2 = torch.load(f)
# sim2 = EmbeddingSimilarity(load_from=data_dir/'models'/'opensecrets_combined.pt')
#
# pred = sim2.predict(test,threshold=0.7,progress_bar=True)
#
#
history_df.assign(t=history_df['step']//2000) \
        .groupby('t')[['batch_loss']].mean() \
        .reset_index() \
        .assign(log_loss=lambda x: np.log(x['batch_loss'])) \
        .plot('t','log_loss')


history_df.assign(t=history_df['step']//2000) \
        .groupby('t')[['global_loss']].mean() \
        .reset_index() \
        .assign(log_loss=lambda x: np.log(x['global_loss'])) \
        .plot('t','log_loss')

#
# history_df.plot('step','score_lr')
# history_df.plot('step','transformer_lr')
#
# matcher = opensecrets.copy()
