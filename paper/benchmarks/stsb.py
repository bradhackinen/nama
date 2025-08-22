import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.stats import spearmanr

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.similarity_model import SimilarityModel


# Load the full STS-B dataset
dataset = load_dataset("sentence-transformers/stsb")

# Access the train, validation, and test splits
validation_dataset = dataset['validation']
test_dataset = dataset['test']


train_df = (pd.DataFrame(dataset['train'])
                # .query('(score < 0.25) | (score > 0.75)')
                .assign(match = lambda x: x['score'] > 0.5))

train = nama.from_df(train_df, pair_columns=['sentence1', 'sentence2'],match_column='match',
                     match_format='pairs')



hparams = {
    'd':None,
    'model_name':'roberta-base',
    'scale':50,
    }

train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':1,
    'add_upper':False,
    'batch_size':8,
    'top_k':4,
    }


sim = SimilarityModel(prompt='Sentence:',**hparams)
sim.to('cuda:0')

history_df = sim.train(train,verbose=True,**train_kwargs)

history_df.plot(y='global_loss',logy=True)
plt.show()


Vi = sim.embed(validation_dataset['sentence1']).V
Vj = sim.embed(validation_dataset['sentence2']).V

X = (Vi*Vj).sum(dim=1)

# Compute spearman correlation


spearmanr(X.cpu(),validation_dataset['score'])