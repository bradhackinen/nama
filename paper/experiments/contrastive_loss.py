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


train_kwargs = {
    'max_epochs': 60,
    'warmup_frac': 0.1,
    'transformer_lr':1e-2,
    # 'projection_lr':1e-3,
    'add_upper':True,
    'batch_size':1000,
    'chunk_size':8,
    }


hparams = {
    'd':None,
    'model_name':'roberta-base',
    }

contrastive_sim = ContrastiveModel(prompt='Organization: ',**hparams)
contrastive_sim.to('cuda:0')
# sim.to('cpu')


contrastive_history_df = contrastive_sim.train(train,verbose=True,**train_kwargs)

contrastive_sim.save(Path(data_dir)/'models'/'contrastive_n1000_60epoch_50pct_train.bin')
contrastive_history_df.to_csv(Path(data_dir)/'models'/'contrastive_n1000_60epoch_50pct_train_history.csv')

contrastive_history_df = pd.read_csv(Path(data_dir)/'models'/'contrastive_n1000_60epoch_50pct_train_history.csv')


contrastive_history_df.plot(y='batch_loss',logy=True)
# contrastive_history_df['batch_loss'].rolling(10).mean().plot(logy=True)
plt.show()



hparams = {
    'd':None,
    'model_name':'roberta-base',
    }

train_kwargs = {
    'max_epochs': 10,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':50,
    'use_counts':False,
    'add_upper':True,
    }


nama_sim = SimilarityModel(prompt='Organization',**hparams)
nama_sim.to('cuda:1')

nama_history_df = nama_sim.train(train,verbose=True,**train_kwargs)

nama_sim.save(Path(data_dir)/'models'/'nama_10epoch_50pct_train.bin')
nama_history_df.to_csv(Path(data_dir)/'models'/'nama_10epoch_50pct_train_history.csv')

nama_history_df.plot(y='global_loss',logy=True)
# nama_history_df['global_loss'].rolling(1000).mean().plot(logy=True)
# plt.show()

# results_df['F1'].max()
old_results_df = pd.read_csv(Path(data_dir)/'models'/'contrastive_v_nama_results.csv')


results = []
for model_name,sim in [('nama',nama_sim)]:#[('contrastive',contrastive_sim),('nama',nama_sim)]:

    test_embeddings = sim.embed(test)
    for threshold in tqdm(np.linspace(0.3,1,31),desc='scoring'):

        pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

        scores = score_predicted(pred,test,use_counts=False)

        scores.update(train_kwargs)

        # scores['model'] = model_name
        # scores['half'] = half
        scores['model'] = model_name
        scores['threshold'] = threshold

        results.append(scores)

results_df = pd.DataFrame(results)

results_df = pd.concat([old_results_df.query('model != "nama"'),results_df])


fig,ax = plt.subplots()
for model_name,df in results_df.groupby('model'):
    df.plot('threshold','F1',label=model_name,ax=ax)

print(results_df.groupby('model')['F1'].max())


results_df.to_csv(Path(data_dir)/'models'/'contrastive_v_nama_results.csv')


results_df = pd.read_csv(Path(data_dir)/'models'/'contrastive_v_nama_results.csv')



# # Compute mean within and between group similarity
# distance_results = []
# for model_name,sim in [('contrastive',contrastive_sim),('nama',nama_sim)]:
#     test_embeddings = sim.embed(test)
#     test_strings = test.strings()
#     for group in tqdm(test.groups.values()):

#         if len(group) < 2:
#             continue

#         V_group = test_embeddings[group].V
#         V_other = test_embeddings[[s for s in test_strings if s not in group]].V

#         X = V_group @ V_group.T


#         cos_within = (X.sum() - torch.diag(X).sum())/ (len(group)**2 - len(group))
#         cos_between = (V_group @ V_other.T).mean()

#         distance_results.append({
#             'model':model_name,
#             'group':group,
#             'n':len(group),
#             'cos_within':cos_within.item(),
#             'cos_between':cos_between.item(),
#             })



# distance_results_df = pd.DataFrame(distance_results)

# distance_results_df.groupby('model')[['cos_within','cos_between']].mean()

# distance_results_df.groupby('model')[['cos_within','cos_between']].std()















# results_df.query('threshold == 0.5')['F1']

# ax = plt.subplot()
# for run_vals, df in results_df.groupby(run_cols):
#     df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, ncol=1)
# plt.show()


# results_df.groupby(run_cols)['F1'].quantile(0.8)



# sim.score_model = CosineSimilarityScore()


# import torch
# class CosineSimilarityScore(torch.nn.Module):
#     '''
#     A placeholder similarity scoring model that passes through the cosine similarity
#     '''
#     def __init__(self,**kwargs):
#         super().__init__()

#     def __repr__(self):
#         return '<nama.CosineSimilarityScore>'

#     def forward(self,X):
#         return X
    
#     def score_to_cos(self,score):
#         return score
    
#     def get_params(self):
#         return {}
    






# class ContrastiveModel(SimilarityModel):
#     """
#     A contrastive learning model that inherits from SimilarityModel but uses NTXent loss
#     instead of a score model. This model directly optimizes cosine similarity between
#     embeddings of matched strings.
#     """
#     def __init__(self,
#                  embedding_class=EmbeddingModel,
#                  weighting_class=ExponentWeights,
#                  temperature=0.07,
#                  loss_type='NTXent',
#                  chunk_size=8,
#                  **kwargs):
#         # Initialize parent class with ContrastiveLossScore
#         super().__init__(
#             embedding_class=embedding_class,
#             score_class=CosineSimilarityScore,
#             weighting_class=weighting_class,
#             **kwargs
#         )

#         self.temperature = temperature
#         self.chunk_size = chunk_size
        
#         # Initialize gradient cache
#         from grad_cache import GradCache
#         self.grad_cache = GradCache(
#             models=[self.embedding_model],
#             chunk_sizes=chunk_size,
#             loss_fn=self._contrastive_loss,
#             get_rep_fn=lambda v: v
#         )

#     def _contrastive_loss(self, anchor, pos, neg, temperature=None):
#         """Compute contrastive loss between anchor, positive and negative samples"""
#         if temperature is None:
#             temperature = self.temperature
            
#         # Compute similarities
#         pos_sim = torch.exp(anchor @ pos.T / temperature)
#         neg_sim = torch.exp(anchor @ neg.T / temperature)
        
#         # Compute denominator (sum of all similarities)
#         denominator = pos_sim.sum() + neg_sim.sum()
        
#         # Compute loss
#         loss = -torch.log(pos_sim / denominator)
#         return loss.mean()

#     def train(self, training_matches, max_epochs=1, neg_samples=64, neg_batch_size=8,
#               transformer_lr=1e-5, projection_lr=1e-5, warmup_frac=0.1,
#               max_grad_norm=1, dropout=False, progress_bar=True, **kwargs):
#         """
#         Train the embedding model using contrastive loss with gradient caching.
#         Uses positive pairs from training data and constructs negative pairs randomly.
#         """

#         # Calculate number of training steps
#         num_training_steps = max_epochs * len(training_matches)
#         num_warmup_steps = int(warmup_frac * num_training_steps)

#         if transformer_lr or projection_lr:
#             embedding_optimizer = self.embedding_model.config_optimizer(transformer_lr, projection_lr)
#             embedding_scheduler = get_cosine_schedule_with_warmup(
#                 embedding_optimizer,
#                 num_warmup_steps=num_warmup_steps,
#                 num_training_steps=num_training_steps)

#         step = 0
#         self.history = []
#         self.val_scores = []
        
#         if dropout:
#             self.embedding_model.train()
#         else:
#             self.embedding_model.eval()
            
#         for epoch in range(max_epochs):
#             # Get all groups and shuffle them
#             groups = list(training_matches.groups.values())
#             random.shuffle(groups)

#             for group in tqdm(groups, desc=f'training epoch {epoch}', disable=not progress_bar):
#                 # Select anchor and positive samples
#                 anchor_string = random.choice(group)
#                 pos_string = random.choice(group)

#                 # Select negative samples
#                 neg_groups = random.sample([g for g in groups if g != group], neg_samples)
#                 neg_strings = [random.choice(g) for g in neg_groups]

#                 # Prepare input tensors for gradient cache
#                 anchor_input = [anchor_string]
#                 pos_input = [pos_string]
#                 neg_input = neg_strings

#                 # Compute loss using gradient cache
#                 loss = self.grad_cache(
#                     anchor_input,
#                     pos_input,
#                     neg_input,
#                     temperature=self.temperature
#                 )

#                 h = {'epoch': epoch, 'step': step}
#                 h['batch_loss'] = loss.detach().mean().item()
#                 h['batch_obs'] = len(neg_strings)

#                 # Update embeddings
#                 embedding_optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)

#                 embedding_optimizer.step()
#                 embedding_scheduler.step()

#                 try:
#                     h['transformer_lr'] = embedding_optimizer.param_groups[1]['lr']
#                 except:
#                     h['transformer_lr'] = 0

#                 try:
#                     h['projection_lr'] = embedding_optimizer.param_groups[-1]['lr']
#                 except:
#                     h['projection_lr'] = 0

#                 self.history.append(h)
#                 step += 1
            
#         return pd.DataFrame(self.history)

