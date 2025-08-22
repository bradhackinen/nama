import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.similarity_model import SimilarityModel, ContrastiveModel
from nama.embedding_similarity.scoring_model import ExpSimilarityScore2

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down-sample for testing
# gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


hparams = {
    'd':None,
    'model_name':'roberta-base',
    'scale':50,
    'shape':2,
    'add_upper':True,
    'prompt':'Organization:',
    'score_class':ExpSimilarityScore2,
    }


train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':10,
    'batch_top_k':True,
    'top_k':32,
    'batch_size': 16,
    'minibatch_size': 8,
    # 'batch_top_k':False,
    # 'top_k':1,
    # 'batch_size': 8,
    }



nama_sim = SimilarityModel(**hparams)
nama_sim.to('cuda:3')

nama_history_df = nama_sim.train(train,verbose=True,**train_kwargs)

# nama_sim.save(Path(data_dir)/'models'/'nama_1epoch_50pct_train.bin')
# nama_history_df.to_csv(Path(data_dir)/'models'/'nama_1epoch_50pct_train_history.csv')

nama_history_df['global_loss'].rolling(100).mean().plot(logy=True)
plt.title('Global Loss')
plt.show()


nama_history_df.plot(y='score_scale')
plt.title('Score Scale')
plt.show()
# nama_history_df.plot(y='score_shape')

nama_history_df['global_mean_cos'].rolling(1000).mean().plot()
plt.title('Mean Cosine Similarity')
plt.show()



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
plt.title('F1 vs Threshold')
plt.show()

print(f'Simple Threshold F1: {results_df["F1"].max()}')


# Refine the similarity score
train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':0,
    'projection_lr':0,
    'score_lr':1,
    # 'max_grad_norm': 1e-4,
    # 'batch_top_k':True,
    # 'top_k':32,
    # 'batch_size': 16,
    # 'batch_top_k':False,
    # 'top_k':1,
    'batch_size': 16,
    # 'minibatch_size': 8,
    }


history2_df = nama_sim.train(train,verbose=True,**train_kwargs)

history2_df['global_loss'].rolling(100).mean().plot(logy=True)
plt.title('Global Loss')
plt.show()

history2_df.plot(y='score_scale')
plt.title('Score Scale')
plt.show()


# pred = test_embeddings.unite_similar(threshold=0.3,loss_threshold=0.5,progress_bar=True)
# scores = score_predicted(pred,test,use_counts=False)

# print(f'Loss Threshold F1: {scores["F1"]}')




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
plt.title('F1 vs Threshold')
plt.show()

print(f'Post-score training F1: {results_df["F1"].max()}')