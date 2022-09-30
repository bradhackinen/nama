from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from polyfuzz import PolyFuzz
from polyfuzz.models import RapidFuzz, EditDistance, TFIDF
from jellyfish import jaro_winkler_similarity

import nama
from nama.scoring import score_predicted, split_on_groups

gold = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')


models = {
    # 'Jaro-Winkler':EditDistance(scorer=jaro_winkler_similarity),
    'TFIDF-top1':TFIDF(top_n=1,min_similarity=0),
    # 'TFIDF-top5':TFIDF(top_n=5,min_similarity=0),
    # 'TFIDF-top10':TFIDF(top_n=10,min_similarity=0),
    'Levenshtein':RapidFuzz(n_jobs=1,score_cutoff=0.01),
    }

results = []

for model_name,model in models.items():
    for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):

        try:
            pf_model = PolyFuzz(model)

            pf_model.fit(gold.strings())

            pf_model.group(link_min_similarity=threshold)

            pf_matcher = gold.split_all().unite(pf_model.get_clusters().values())

            scores = score_predicted(pf_matcher,gold,use_counts=True)

            scores['threshold'] = threshold

            scores['model'] = model_name

            results.append(scores)

        except ValueError:
            print(f'Warning: {model_name} failed with {threshold=}')

results_df = pd.DataFrame(results)

# results_df.to_csv(nama.root_dir/'experiments'/'polyfuzz_canlobby_results.csv',index=False)

# mean_results_df = results_df#.groupby(run_cols+['threshold']).mean().reset_index()

run_cols = ['model']

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


results_df.sort_values('F1',ascending=False).head(1).T




# df = results_df.copy()
#
#
# df.loc[df['threshold'] <= 0,'precision'] = 0
# df.loc[df['threshold'] >= 1,'precision'] = 1
# df.loc[df['threshold'] <= 0,'recall'] = 1
#
#
#
# df['N'] = df[['TP','FP','TN','FN']].sum(axis=1)
#
#
# df.query('model=="Levenshtein"')[['TP','FP','TN','FN']]
#
# df.query('model=="TFIDF-top1"')\
#     .plot(y=['TP','FN','FP'],x='threshold')
#
# df.query('model=="TFIDF-top1"')\
#     [['FP','TN']] \
#     .sum(axis=1) \
#     .value_counts()
#
#     .plot(y=['TP','FN','FP'],x='threshold')
#
#
# df.query('model=="TFIDF-top1"')\
#     .plot(y=['precision','recall'],x='threshold')
#
#
# df.query('model=="TFIDF-top1"')[['threshold','FP']]
#
#
# t_groups = {}
# for threshold in [0.6419,0.64199,0.642]:
#
#     pf_model = PolyFuzz(TFIDF(top_n=1,min_similarity=0))
#
#     pf_model.fit(gold.strings())
#
#     pf_model.group(link_min_similarity=threshold)
#
#     pf_groups = pf_model.get_clusters().values()
#
#     t_groups[threshold] = pf_groups
#
#     pf_matcher = gold.split_all().unite(pf_groups)
#
#     scores = score_predicted(pf_matcher,gold,use_counts=True)
#     len(pf_groups)
#     scores
#
#     print(f"\n{threshold=}, {len(pf_groups)=}, {scores['FP']=}")
#     print(pf_matcher.__repr__())
#
#
# t_group_sets = {t:{tuple(sorted(g)) for g in groups} for t,groups in t_groups.items()}
#
# t_group_sets[0.6419] - t_group_sets[0.64199]
# t_group_sets[0.64199] - t_group_sets[0.6419]
#
# key_strings = {s.lower() for group in (t_group_sets[0.6419] ^ t_group_sets[0.64199]) for s in group}
#
# m_gold = gold.to_df()
#
#
# keep(key_strings)
#
# m0 = m_gold.split_all().unite(t_group_sets[0.6419])
# m1 = m_gold.split_all().unite(t_group_sets[0.64199])
#
# m0.to_df()
# m1.to_df()
# m_gold.to_df()
# 
# score_predicted(m0,m_gold)
# score_predicted(m1,m_gold)
#
#
# pf_model.fit(key_strings)
#
# pf_model.get_matches()
#
# pf_model.group(link_min_similarity=threshold)
#
# pf_groups = pf_model.get_clusters().values()
#
# t_groups[threshold] = pf_groups
#
# pf_matcher = gold.split_all().unite(pf_groups)
