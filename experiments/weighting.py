import numpy as np
import pandas as pd
from tqdm import tqdm

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity import EmbeddingSimilarity


gold = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_client_names.csv')


results = []
for train_counts in True,False:
    for seed in range(1,21):
        print(f'\n Testing {train_counts=}, {seed=}')

        train,test = split_on_groups(gold,0.5,seed=seed)

        sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
        sim.to('cuda:1')
        history_df = sim.train(train,use_counts=train_counts)

        for threshold in tqdm(np.linspace(0,1,51),desc='scoring'):
            pred = sim.predict(test,threshold=threshold,progress_bar=False)

            for test_counts in True,False:

                scores = score_predicted(pred,test,use_counts=test_counts)

                scores['seed'] = seed
                scores['threshold'] = threshold
                scores['train_counts'] = train_counts
                scores['test_counts'] = test_counts

                results.append(scores)

        sim.to('cpu')


results_df = pd.DataFrame(results)

results_df.to_csv(nama.root_dir/'experiments'/'weighting_results.csv')



import matplotlib.pyplot as plt

mean_results_df = results_df.groupby(['train_counts','test_counts','threshold']).mean().reset_index()

ax = plt.subplot()
for (train_counts,test_counts), df in mean_results_df.groupby(['train_counts','test_counts']):
    df.plot('recall','precision',ax=ax,label=f'train_counts={train_counts}, test_counts={test_counts}')

ax = plt.subplot()
for (train_counts,test_counts), df in mean_results_df.groupby(['train_counts','test_counts']):
    df.plot('threshold','F1',ax=ax,label=f'train_counts={train_counts}, test_counts={test_counts}')
