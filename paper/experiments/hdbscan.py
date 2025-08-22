import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.similarity_model import SimilarityModel, ContrastiveModel
from nama.embedding_similarity.scoring_model import LogisticSimilarityScore,ExpSimilarityScore

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down sample for testing
# gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


hparams = {
    'd':64,
    'model_name':'roberta-base',
    }

train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':1e-4,
    'use_counts':False,
    'add_upper':True,
    }


# sim = SimilarityModel(prompt='Organization',**hparams)
# sim.to('cuda:0')

# history_df = sim.train(train,verbose=True,**train_kwargs)

# sim.save(Path(data_dir)/'models'/'nama_6d4_50pct_train.bin')
# history_df.to_csv(Path(data_dir)/'models'/'nama_6d4_50pct_train_history.csv')
# history_df.plot(y='global_loss',logy=True)
# history_df.plot(y='score_scale')

sim = nama.load_similarity_model(Path(data_dir)/'models'/'nama_6d4_50pct_train.bin')
sim.to('cuda:0')

test_embeddings = sim.embed(test)


# results = []
# for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):

#     pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

#     scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

#     scores.update(train_kwargs)

#     scores['threshold'] = threshold

#     results.append(scores)

# results_df = pd.DataFrame(results)

# results_df.plot('threshold','F1')


# results_df['F1'].max()


test_embeddings.half()

pred = test_embeddings.unite_similar(threshold=0.5,loss_threshold=None,progress_bar=False)

score_predicted(pred,test,use_counts=train_kwargs['use_counts'])


pred = test_embeddings.unite_similar(threshold=0.2,loss_threshold=0.5,progress_bar=True)

score_predicted(pred,test,use_counts=train_kwargs['use_counts'])




results = []
for threshold in tqdm(np.linspace(0.3,1,15),desc='scoring'):

    pred = test_embeddings.unite_similar(threshold=0.3,loss_threshold=threshold,progress_bar=False)

    scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

    scores.update(train_kwargs)

    scores['threshold'] = threshold

    results.append(scores)

results_df = pd.DataFrame(results)

results_df.plot('threshold','F1')


results_df['F1'].max()




# Cluster using HDBSCAN with euclidean distance
import hdbscan

# Get 

clusterer = hdbscan.HDBSCAN(min_cluster_size=2,metric='euclidean',cluster_selection_method='leaf')

clusterer.fit(test_embeddings.V.cpu().numpy())

clusterer.labels_

pred = test_embeddings._group_ids_to_match_data(clusterer.labels_)

score_predicted(pred,test,use_counts=train_kwargs['use_counts'])


from sklearn.neighbors import BallTree

tree = BallTree(test_embeddings.V.cpu().numpy(),metric='euclidean')

threshold_dist = np.sqrt(2*test_embeddings.score_model.score_to_cos(0.5))

tree.query_radius(test_embeddings.V.cpu().numpy(),r=threshold_dist,return_distance=True,dual_tree=True)

