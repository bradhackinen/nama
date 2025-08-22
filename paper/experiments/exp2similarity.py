import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.similarity_model import SimilarityModel, ContrastiveModel
from nama.embedding_similarity.scoring_model import LogisticSimilarityScore,ExpSimilarityScore2

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down sample for testing
gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)




hparams = {
    'd':None,
    'model_name':'roberta-base',
    'score_class':ExpSimilarityScore2,
    'scale':50,
    'shape':1
    }

train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':1,
    'use_counts':False,
    'add_upper':True,
    }


sim = SimilarityModel(prompt='Organization',**hparams)
sim.to('cuda:2')

history_df = sim.train(train,verbose=True,**train_kwargs)

# sim.save(Path(data_dir)/'models'/'nama_10epoch_50pct_train.bin')
# history_df.to_csv(Path(data_dir)/'models'/'nama_10epoch_50pct_train_history.csv')

history_df.plot(y='global_loss',logy=True)
plt.show()
history_df.plot(y='score_scale')
plt.show()
# history_df.plot(y='score_shape')
# plt.show()



results = []
test_embeddings = sim.embed(test)
for threshold in tqdm(np.linspace(0.2,1,21),desc='scoring'):

    pred = test_embeddings.unite_similar(threshold=threshold,
                                        #  group_threshold=threshold,
                                         progress_bar=False)

    scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

    scores.update(train_kwargs)

    scores['threshold'] = threshold

    results.append(scores)

results_df = pd.DataFrame(results)

results_df.plot('threshold','F1')


results_df['F1'].max()







