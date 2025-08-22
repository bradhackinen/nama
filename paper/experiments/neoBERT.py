import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.similarity_model import SimilarityModel, ContrastiveModel
from nama.embedding_similarity.scoring_model import LogisticSimilarityScore

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down sample for testing
# gold,_ = split_on_groups(gold,0.01,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


hparams = {
    'd':None,
    'model_name':'chandar-lab/NeoBERT',
    # 'model_name':'roberta-base',
    'scale':50,
    'trust_remote_code':True,
    }

# train_kwargs = {
#     'max_epochs': 3,
#     'warmup_frac': 0.1,
#     'transformer_lr':5e-6,
#     'score_lr':1,
#     'use_weights':False,
#     'add_upper':True,
#     'batch_size':8,
#     'grad_acc':8,
#     }



# train_kwargs = {
#     'max_epochs': 2,
#     'warmup_frac': 0.1,
#     'transformer_lr':5e-6,
#     'score_lr':1,
#     'add_upper':True,
#     'batch_size':8,
#     'grad_acc':1,
#     }


train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-5,
    'score_lr':10,
    'add_upper':True,
    'top_k':4,
    'grad_acc':1,
    }



nama_sim = SimilarityModel(prompt='Organization',**hparams)

nama_sim.to('cuda:1')

nama_history_df = nama_sim.train(train,verbose=True,**train_kwargs)

nama_sim.save(Path(data_dir)/'models'/'nama_NeoBERT_1epoch_top4_50pct_train.bin')
nama_history_df.to_csv(Path(data_dir)/'models'/'nama_NeoBERT_1epoch_top4_50pct_train_history.csv')

nama_history_df.plot(y='global_loss',logy=True)
plt.show()



# nama_history_df.plot(y='score_scale')
# nama_history_df['batch_pos_target'].rolling(10).mean().plot()
# nama_history_df['global_mean_cos'].rolling(1000).mean().plot()



test_embeddings = nama_sim.embed(test)
test_embeddings.half()



results = []
for threshold in tqdm(np.linspace(0.3,1,15),desc='scoring'):

    pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

    scores = score_predicted(pred,test,use_counts=False)

    scores.update(train_kwargs)

    scores['threshold'] = threshold

    results.append(scores)

results_df = pd.DataFrame(results)

results_df.plot('threshold','F1')

print(f'Simple Threshold F1: {results_df["F1"].max()}')



