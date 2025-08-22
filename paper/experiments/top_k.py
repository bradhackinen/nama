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
# gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


hparams = {
    'd':None,
    'model_name':'roberta-base',
    'scale':50,
    }


results = []
for k in [1,2,3,4,8]:
    print(f'Top K: {k}')


    train_kwargs = {
        'max_epochs': 1,
        'warmup_frac': 0.1,
        'transformer_lr':5e-6,
        'score_lr':1,
        'add_upper':True,
        'batch_size':8,
        'top_k':k,
        'grad_acc':1,
        }
    

    sim = SimilarityModel(prompt='Organization',**hparams)
    sim.to('cuda:1')

    history_df = sim.train(train,verbose=True,**train_kwargs)

    sim.save(Path(data_dir)/'models'/f'nama_1epoch_top{k}_50pct_train.bin')
    history_df.to_csv(Path(data_dir)/'models'/f'nama_1epoch_top{k}_50pct_train_history.csv')

    history_df.plot(y='global_loss',logy=True)
    plt.title(f'Top {k} Training Loss')
    plt.show()

    # history_df.plot(y='score_scale')
    # history_df['batch_pos_target'].rolling(10).mean().plot()
    history_df['global_mean_cos'].rolling(10).mean().plot()
    plt.title(f'Top {k} Training Mean Cosine Similarity')
    plt.show()



    test_embeddings = sim.embed(test)
    test_embeddings.half()

    sim.to('cpu')

    for threshold in tqdm(np.linspace(0.3,1,15),desc='scoring'):

        pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

        scores = score_predicted(pred,test,use_counts=False)

        scores.update(train_kwargs)

        scores['threshold'] = threshold
        scores['top_k'] = k

        results.append(scores)


    results_df = pd.DataFrame(results)

    fig,ax = plt.subplots(figsize=(10,5))

    for k in results_df['top_k'].unique():
        results_df[results_df['top_k'] == k].plot('threshold','F1',ax=ax,label=k)

    plt.title('Top K F1 Score')
    plt.show()

    print(results_df.groupby('top_k')['F1'].max())



    pred = test_embeddings.unite_similar(threshold=0.5,loss_threshold=0.5,progress_bar=True)
    scores = score_predicted(pred,test,use_counts=False)

    print(f'Loss Threshold F1: {scores["F1"]}')

    test_embeddings.to('cpu')


pred = test_embeddings.unite_similar(threshold=0.4,loss_threshold=0.4,progress_bar=True)
score_predicted(pred,test,use_counts=False)

pred = test_embeddings.unite_similar(threshold=0.5,progress_bar=True)
score_predicted(pred,test,use_counts=False)

# test_embeddings.to('cuda:0')

results = []
for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

    pred = test_embeddings.unite_similar(threshold=threshold,loss_threshold=threshold,progress_bar=False)

    scores = score_predicted(pred,test,use_counts=False)

    scores.update(train_kwargs)

    scores['threshold'] = threshold

    results.append(scores)

results_df = pd.DataFrame(results)

results_df.plot('threshold','F1')

# results_df['F1'].max()


