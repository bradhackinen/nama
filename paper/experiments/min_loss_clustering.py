import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down sample for testing
# gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


sim = nama.load_similarity_model(Path(data_dir)/'models'/'nama_1epoch_top8_50pct_train.bin')
sim.to('cuda:2')

test_embeddings = sim.embed(test)


test_embeddings.half()


# results = []
# for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

#     pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

#     scores = score_predicted(pred,test,use_counts=False)

#     scores['threshold'] = threshold

#     results.append(scores)

# results_df = pd.DataFrame(results)

# results_df.plot('threshold','F1')

# results_df['F1'].max()



pred = test_embeddings.unite_similar(threshold=0.5,progress_bar=True)

score_predicted(pred,test,use_counts=False)

pred = test_embeddings.unite_similar(threshold=0.5,group_threshold=0.01,progress_bar=True)

score_predicted(pred,test,use_counts=False)

1 - sim.score_model.score_to_cos(0.5)
1 - sim.score_model.score_to_cos(0.25)


pred = test_embeddings.unite_similar(threshold=0.5,loss_threshold=0,progress_bar=True)

score_predicted(pred,test,use_counts=False)


# pred = test_embeddings.unite_similar(threshold=0.3,loss_threshold=0,progress_bar=True)

# score_predicted(pred,test,use_counts=train_kwargs['use_counts'])


