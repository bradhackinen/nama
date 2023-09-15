import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy,deepcopy
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup, logging
import requests
from io import BytesIO

from ..match_data import MatchData
from ..scoring import score_predicted
from .scoring_model import SimilarityScore
from .embeddings import Embeddings
from .embedding_model import EmbeddingModel

logging.set_verbosity_error()


class ExponentWeights():
    def __init__(self,weighting_exponent=0.5,**kwargs):
        self.exponent = weighting_exponent

    def __call__(self,counts):
        return counts**self.exponent


class SimilarityModel(nn.Module):
    """
    A combined embedding/scorer model that produces Embeddings objects
    as its primary output.

    - train() jointly optimizes the embedding_model and score_model using
      contrastive learning to learn from a training MatchData.
    """
    def __init__(self,
                    embedding_class=EmbeddingModel,
                    score_class=SimilarityScore,
                    weighting_class=ExponentWeights,
                    **kwargs):

        super().__init__()

        self.embedding_model = embedding_class(**kwargs)
        self.score_model = score_class(**kwargs)
        self.weighting_function = weighting_class(**kwargs)
        self.config = kwargs

        self.to(kwargs.get('device','cpu'))

    def to(self,device):
        super().to(device)
        self.embedding_model.to(device)
        self.score_model.to(device)
        self.device = device

    def save(self,savefile):
        torch.save({'metadata': self.config, 'state_dict': self.state_dict()}, savefile)

    @torch.no_grad()
    def embed(self,input,to=None,batch_size=64,progress_bar=True,**kwargs):
        """
        Construct an Embeddings object from input strings or a MatchData
        """

        if to is None:
            to = self.device

        if isinstance(input, MatchData):
            strings = input.strings()
            counts = torch.tensor([input.counts[s] for s in strings],device=self.device).float().to(to)

        else:
            strings = list(input)
            counts = torch.ones(len(strings),device=self.device).float().to(to)

        input_loader = DataLoader(strings,batch_size=batch_size,num_workers=0)

        self.embedding_model.eval()

        V = None
        batch_start = 0
        with tqdm(total=len(strings),delay=1,desc='Embedding strings',disable=not progress_bar) as pbar:
            for batch_strings in input_loader:

                v = self.embedding_model(batch_strings).detach().to(to)

                if V is None:
                    # Use v to determine dim and dtype of pre-allocated embedding tensor
                    # (Pre-allocating avoids duplicating tensors with a big .cat() operation)
                    V = torch.empty(len(strings),v.shape[1],device=to,dtype=v.dtype)

                V[batch_start:batch_start+len(batch_strings),:] = v

                pbar.update(len(batch_strings))
                batch_start += len(batch_strings)

        score_model = copy(self.score_model)
        score_model.load_state_dict(self.score_model.state_dict())
        score_model.to(to)

        weighting_function = deepcopy(self.weighting_function)

        return Embeddings(strings=strings,
                            V=V.detach(),
                            counts=counts.detach(),
                            score_model=score_model,
                            weighting_function=weighting_function,
                            device=to)

    def train(self,training_matches,max_epochs=1,batch_size=8,
                transformer_lr=1e-5,projection_lr=1e-5,score_lr=10,warmup_frac=0.1,
                max_grad_norm=1,dropout=False,
                progress_bar=True,
                **kwargs):

        """
        Train the embedding_model and score_model to predict match probabilities
        using the training_matches as a source of "correct" matches.
        Training algorithm uses contrastive learning with hard-positive
        and hard-negative mining to fine tune the embedding model to place
        matched strings near to each other in embedding space, while
        simulataneously calibrating the score_model to predict the match
        probabilities as a function of cosine distance
        """

        num_training_steps = max_epochs*len(training_matches)//batch_size
        num_warmup_steps = int(warmup_frac*num_training_steps)

        if transformer_lr or projection_lr:
            embedding_optimizer = self.embedding_model.config_optimizer(transformer_lr,projection_lr)
            embedding_scheduler = get_cosine_schedule_with_warmup(
                                        embedding_optimizer,
                                        num_warmup_steps=num_warmup_steps,
                                        num_training_steps=num_training_steps)
        if score_lr:
            score_optimizer = self.score_model.config_optimizer(score_lr)
            score_scheduler = get_linear_schedule_with_warmup(
                                        score_optimizer,
                                        num_warmup_steps=num_warmup_steps,
                                        num_training_steps=num_training_steps)

        step = 0
        self.history = []
        self.val_scores = []
        for epoch in range(max_epochs):

            global_embeddings = self.embed(training_matches)

            strings = global_embeddings.strings
            V = global_embeddings.V
            w = global_embeddings.w

            groups = torch.tensor([global_embeddings.string_map[training_matches[s]] for s in strings],device=self.device)

            # Normalize weights to make learning rates more general
            if w is not None:
                w = w/w.mean()

            shuffled_ids = list(range(len(strings)))
            random.shuffle(shuffled_ids)

            if dropout:
                self.embedding_model.train()
            else:
                self.embedding_model.eval()

            for batch_start in tqdm(range(0,len(strings),batch_size),desc=f'training epoch {epoch}',disable=not progress_bar):

                h = {'epoch':epoch,'step':step}

                batch_i = shuffled_ids[batch_start:batch_start+batch_size]

                # Recycle ids from the beginning to pad the last batch if necessary
                if len(batch_i) < batch_size:
                    batch_i = batch_i + shuffled_ids[:(batch_size-len(batch_i))]

                """
                Find highest loss match for each batch string (global search)

                Note: If we compute V_i with dropout enabled, it will add noise
                to the embeddings and prevent the same pairs from being selected
                every time.
                """
                V_i = self.embedding_model(strings[batch_i])

                # Update global embedding cache
                V[batch_i,:] = V_i.detach()

                with torch.no_grad():

                    global_X = V_i@V.T
                    global_Y = (groups[batch_i][:,None] == groups[None,:]).float()

                    if w is not None:
                        global_W = torch.outer(w[batch_i],w)
                    else:
                        global_W = None

                # Train scoring model only
                if score_lr:
                    # Make sure gradients are enabled for score model
                    self.score_model.requires_grad_(True)

                    global_loss = self.score_model.loss(global_X,global_Y,weights=global_W)

                    score_optimizer.zero_grad()
                    global_loss.nanmean().backward()
                    torch.nn.utils.clip_grad_norm_(self.score_model.parameters(),max_norm=max_grad_norm)

                    score_optimizer.step()
                    score_scheduler.step()

                    h['score_lr'] = score_optimizer.param_groups[0]['lr']
                    h['global_mean_cos'] = global_X.mean().item()
                    try:
                        h['score_alpha'] = self.score_model.alpha.item()
                    except:
                        pass

                else:
                    with torch.no_grad():
                        global_loss = self.score_model.loss(global_X,global_Y)

                h['global_loss'] = global_loss.detach().nanmean().item()

                # Train embedding model
                if (transformer_lr or projection_lr) and step <= num_warmup_steps + num_training_steps:

                    # Turn off score model updating - only want to train embedding here
                    self.score_model.requires_grad_(False)

                    # Select hard training examples
                    with torch.no_grad():
                        batch_j = global_loss.argmax(dim=1).flatten()

                        if w is not None:
                            batch_W = torch.outer(w[batch_i],w[batch_j])
                        else:
                            batch_W = None

                    # Train the model on the selected high-loss pairs
                    V_j = self.embedding_model(strings[batch_j.tolist()])

                    # Update global embedding cache
                    V[batch_j,:] = V_j.detach()

                    batch_X = V_i@V_j.T
                    batch_Y = (groups[batch_i][:,None] == groups[batch_j][None,:]).float()
                    h['batch_obs'] = len(batch_i)*len(batch_j)

                    batch_loss = self.score_model.loss(batch_X,batch_Y,weights=batch_W)

                    h['batch_nan'] = torch.isnan(batch_loss.detach()).sum().item()

                    embedding_optimizer.zero_grad()
                    batch_loss.nanmean().backward()

                    torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm=max_grad_norm)

                    embedding_optimizer.step()
                    embedding_scheduler.step()

                    h['transformer_lr'] = embedding_optimizer.param_groups[1]['lr']
                    h['projection_lr'] = embedding_optimizer.param_groups[-1]['lr']

                    # Save stats
                    h['batch_loss'] = batch_loss.detach().mean().item()
                    h['batch_pos_target'] = batch_Y.detach().mean().item()

                self.history.append(h)

        return pd.DataFrame(self.history)

    def unite_similar(self,input,**kwargs):
        embeddings = self.embed(input,**kwargs)
        return embeddings.unite_similar(**kwargs)

    def test(self,gold_matches, threshold=0.5, **kwargs):
        embeddings = self.embed(gold_matches, **kwargs)

        if (isinstance(threshold, float)):
            predicted = embeddings.unite_similar(threshold=threshold, **kwargs)
            scores = score_predicted(predicted, gold_matches, use_counts=True)

            return scores
        
        results = []
        for thres in threshold:
            predicted = embeddings.unite_similar(threshold=thres, **kwargs)

            scores = score_predicted(predicted, gold_matches, use_counts=True)
            scores["threshold"] = thres
            results.append(scores)

        
        return results

def debug_state_dict(f, **kwargs):
    return torch.load(f, map_location='cpu', **kwargs)

def load_similarity_model(f,map_location='cpu',**kwargs):
    checkpoint = torch.load(f, map_location=map_location, **kwargs)
    metadata = checkpoint['metadata']
    state_dict = checkpoint['state_dict']

    model = SimilarityModel(**metadata)
    model.load_state_dict(state_dict)

    return model
    #return torch.load(f,map_location=map_location,**kwargs)


def load_pretrained_model(model='base', map_location='cpu', **kwargs):

    if model.lower() not in ["base"]:
        raise ValueError("Model must be 'base'")

    response = requests.get(f"https://huggingface.co/beny2000/nama/resolve/main/nama-{model.lower()}.bin")
    model_data = BytesIO(response.content)
    
    return load_similarity_model(model_data, map_location=map_location, **kwargs)
    


