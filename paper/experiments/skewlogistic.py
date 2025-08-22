import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.similarity_model import SimilarityModel, ContrastiveModel
from nama.embedding_similarity.scoring_model import SkewLogisticSimilarityScore

data_dir = Path(os.environ['NAMA_DATA'])

gold = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# down sample for testing
# gold,_ = split_on_groups(gold,0.1,seed=1)

test,train = split_on_groups(gold,0.5,seed=1)


hparams = {
    'd':None,
    'model_name':'roberta-base',
    'alpha':2,
    'beta':-1,
    'gamma':0,
    'add_upper':True,
    'prompt':'Organization:',
    'score_class':SkewLogisticSimilarityScore,
    }


full_train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':5e-6,
    'score_lr':[1e-3,1e-2,0],
    'batch_top_k':True,
    'top_k':32,
    'grad_acc':1,
    'batch_size': 16,
    'minibatch_size': 8,
    }


# Refine the similarity score
score_train_kwargs = {
    'max_epochs': 1,
    'warmup_frac': 0.1,
    'transformer_lr':0,
    'projection_lr':0,
    'score_lr':[1,0.1,0],
    'batch_size': 16,
    }




def history_plots(history_df,bandwidth=0.1):
    fig,axes = plt.subplots(1,3,figsize=(10,3))

    window = int(bandwidth*len(history_df))

    history_df['global_loss'].rolling(window).mean().plot(logy=True,ax=axes[0])
    axes[0].set_title('Global Loss')

    history_df['global_mean_cos'].rolling(window).mean().plot(ax=axes[1])
    axes[1].set_title('Mean Cosine Similarity')

    history_df['score_alpha'].rolling(window).mean().plot(ax=axes[2])
    history_df['score_beta'].rolling(window).mean().plot(ax=axes[2])
    history_df['score_gamma'].rolling(window).mean().plot(ax=axes[2])
    axes[2].set_title('Score Parameters')

    # Place plot to the right of the other plots
    axes[2].legend([r'$\alpha$',r'$\beta$',r'$\gamma$'],
                   loc='center left',bbox_to_anchor=(1.05,0.5),bbox_transform=axes[2].transAxes)

    return fig,axes



sim = SimilarityModel(**hparams)
sim.to('cuda:1')

# Train full model
history1_df = sim.train(train,verbose=True,**full_train_kwargs)

history_plots(history1_df)
plt.tight_layout()
plt.show()

# Refine similarity scoring model
history2_df = sim.train(train,verbose=True,**score_train_kwargs)

history_plots(history2_df)
plt.tight_layout()
plt.show()


# sim.save(Path(data_dir)/'models'/'1epoch_50pct_train.bin')
# history1_df.to_csv(Path(data_dir)/'models'/'1epoch_50pct_train_history1.csv')
# history2_df.to_csv(Path(data_dir)/'models'/'1epoch_50pct_train_history2.csv')




# history_df.plot(y='score_scale')
# history_df['batch_pos_target'].rolling(100).mean().plot()
# history_df['global_mean_cos'].rolling(1000).mean().plot()


@torch.no_grad()
def binned_prob(embeddings,matches,bw=0.1,batch_size=64,progress_bar=True,**kwargs):
    """
    Get the mean loss for all pairs of strings in the embeddings
    """
    if matches is not None:
        embeddings = embeddings.embed(matches)
        group_ids = embeddings._match_data_to_group_ids(matches)
    else:
        group_ids = torch.arange(len(embeddings)).to(embeddings.device)

    V = embeddings.V
    w = embeddings.w
    
    # Get the mean loss for all pairs of strings in the embeddings
    binned_probs = []
    for batch_start in tqdm(range(0,len(embeddings),batch_size),
                                desc='Scoring pairs',delay=1,smoothing=0.01,
                                disable=not progress_bar):

        V_i = V[batch_start:batch_start+batch_size]

        X = V_i@V.T
        Y = (group_ids[batch_start:batch_start+batch_size,None] == group_ids[None,:]).float()

        score = embeddings.score_model(X)

        for b in torch.arange(0,1,step=bw):
            mask = (score > b) & (score < b + bw)

            bin_size = mask.sum()
            if bin_size == 0:
                continue

            else:
                bin_matches = Y[mask].sum()
                bin_sum = score[mask].sum()
                bin_size = mask.sum()

                binned_probs.append({
                    'bin':b.item(),
                    'bin_size':bin_size.item(),
                    'bin_matches':bin_matches.item(),
                    'bin_sum':bin_sum.item()})

    binned_df = (pd.DataFrame(binned_probs)
                    .groupby('bin',as_index=False).sum()
                    .assign(score = lambda x: x['bin_sum']/x['bin_size'],
                            match_prob = lambda x: x['bin_matches']/x['bin_size']))

    return binned_df





test_embeddings = sim.embed(test)
test_embeddings.half()

train_embeddings = sim.embed(train)
train_embeddings.half()


train_probs_df = binned_prob(train_embeddings,train,bw=0.05,batch_size=64,progress_bar=True)
test_probs_df = binned_prob(test_embeddings,test,bw=0.05,batch_size=64,progress_bar=True)



fig,ax = plt.subplots(1,1,figsize=(5,5))

train_probs_df.plot('score','match_prob',logy=False,ax=ax,label='Train')
test_probs_df.plot('score','match_prob',logy=False,ax=ax,label='Test')

# add 45 degree line
ax.plot([0,1],[0,1],color='black',linestyle='--',lw=1)
ax.set_xlabel('Score')
ax.set_ylabel('Match Probability')

ax.set_xlim(0,1)
ax.set_ylim(0,1)

plt.show()


# import torch

# test_embeddings._fast_unite_similar(torch.arange(len(test_embeddings)).to(test_embeddings.device),threshold=0.5,progress_bar=True)

# test_embeddings.unite_similar(threshold=0.5,progress_bar=False)

results = []
for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

    pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

    scores = score_predicted(pred,test,use_counts=False)

    scores['threshold'] = threshold

    results.append(scores)

results_df = pd.DataFrame(results)

results_df.plot('threshold','F1')
plt.title('Threshold vs F1')
plt.show()
# print(f'Max F1: {results_df["F1"].max()}')
print(f'Simple Threshold F1: {results_df["F1"].max()}')






# self = test_embeddings
# batch_size = 64
# progress_bar = True
# threshold = 0.5
# group_ids = torch.arange(len(test_embeddings))

# test_embeddings.score_model.to('cpu')

# self = test_embeddings.score_model
# self.to('cuda:1')

# score = torch.tensor(threshold).to(test_embeddings.device)

# b = torch.exp(-self.beta)
# g = torch.exp(self.gamma)

# # cos = 1 - exp(-self.alpha)*(score**-g - 1)**(1/b)
# cos = 1 - (torch.exp(-b*self.alpha)*(score**(-1/g) - 1))**(1/b)

# cos = torch.clamp(cos,min=-1,max=1)


# # Non-pytorch version
# score = float(threshold)

# b = np.exp(-self.beta.item())
# g = np.exp(self.gamma.item())

# cos = 1 - (np.exp(-b*self.alpha.item())*(score**(-1/g) - 1))**(1/b)

# cos = np.clip(cos,-1,1)

# float(cos)


# with torch.no_grad():
#     V = self.V
#     cos_threshold = self.score_model.score_to_cos(threshold)

#     for batch_start in tqdm(range(0,len(self),batch_size),
#                                 desc='Predicting matches',delay=1,smoothing=0.01,
#                                 disable=not progress_bar):

#         i_slice = slice(batch_start,batch_start+batch_size)
#         j_slice = slice(batch_start+1,None)

#         g_i = group_ids[i_slice]
#         g_j = group_ids[j_slice]

#         # Find j's with jaccard > threshold ("matches")
#         batch_matched = (V[i_slice]@V[j_slice].T >= cos_threshold) \
#                         * (g_i[:,None] != g_j[None,:])

#         for k,matched in enumerate(batch_matched):
#             if matched.any():
#                 # Get the group ids of the matched j's
#                 matched_groups = g_j[matched]

#                 # Identify all embeddings in these groups
#                 ids_to_group = torch.isin(group_ids,matched_groups)

#                 # Assign all matched embeddings to the same group
#                 group_ids[ids_to_group] = g_i[k].clone()

# return self._group_ids_to_match_data(group_ids)




pred = test_embeddings.unite_similar(min_loss=True,progress_bar=True)
scores = score_predicted(pred,test,use_counts=False)

print(f'Loss Threshold F1: {scores["F1"]}')




# results = []
# for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

#     pred = test_embeddings.unite_similar(threshold=threshold,loss_threshold=threshold,progress_bar=False)

#     scores = score_predicted(pred,test,use_counts=False)

#     scores['threshold'] = threshold

#     results.append(scores)

# results_df = pd.DataFrame(results)

# results_df.plot('threshold','F1')

# results_df['F1'].max()

# plt.show()





# mean_loss = get_mean_loss(test_embeddings,test,batch_size=32,progress_bar=True)

# test_embeddings.weighting_function


# results = []
# for threshold in tqdm(np.linspace(0.2,1,17),desc='scoring'):

#     pred = test_embeddings.unite_similar(threshold=threshold,progress_bar=False)

#     mean_loss = get_mean_loss(test_embeddings,pred,batch_size=64,progress_bar=False)

#     results.append({'threshold':threshold,'mean_loss':mean_loss})

# results_df = pd.DataFrame(results)

# results_df.plot('threshold','mean_loss',logy=True)
# plt.title('Mean Loss vs Threshold')

# results_df['F1'].max()
# matches = test
# embeddings = test_embeddings




# sim.score_model


# import torch
# import numpy as np
# import torch.nn.functional as F
# from torch import exp,logit,sigmoid,log1p,log
# from matplotlib import pyplot as plt

# X = torch.linspace(-1,1,500)
# epsilon = 1e-6

# alpha = 3
# beta = -2
# gamma = -2

# b = torch.exp(torch.tensor(-beta))
# g = torch.exp(torch.tensor(gamma))

# Z = -log1p(-X)

# score = (1 + torch.exp(-b*(Z - alpha)))**-g

# # score = (1 + torch.exp(alpha*b)*(1 - X)**b)** -g

# score = torch.clamp(score,min=0,max=1.0)

# plt.plot(X,score)

# loss1 = F.binary_cross_entropy(score, torch.ones_like(score),reduction='none')
# loss0 = F.binary_cross_entropy(score, torch.zeros_like(score),reduction='none')

# fig,ax = plt.subplots(1,1,figsize=(5,5))
# ax.plot(X,loss1,label='Positive')
# ax.plot(X,loss0,label='Negative')


# sim.score_model.score_to_cos(0.5)

# sim.score_model(torch.tensor(sim.score_model.score_to_cos(0.5)))


# sim.to('cpu')
# with torch.no_grad():
#     plt.plot(X,sim.score_model(X))

