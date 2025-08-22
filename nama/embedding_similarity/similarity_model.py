import sys
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
from grad_cache import GradCache
import time
import datetime

from ..match_data import MatchData
from ..scoring import score_predicted
from .scoring_model import *
from .embeddings import Embeddings
from .embedding_model import EmbeddingModel

logging.set_verbosity_error()




class SimilarityModel(nn.Module):
    """
    A combined embedding/scoring model with a common interface for training and
    evaluation. The model is initialized with an embedding model and a scoring
    model. The embedding model is used to generate embeddings for input strings,
    and the scoring model is used to compute similarity scores between pairs of
    embeddings. The model can be trained using a set of training matches, and
    the embeddings can be saved and loaded for later use.
    

    """
    def __init__(self,
                    embedding_class=EmbeddingModel,
                    score_class=LogLogisticSimilarityScore,
                    **kwargs):

        super().__init__()

        self.embedding_model = embedding_class(**kwargs)
        self.score_model = score_class(**kwargs)
        self.config = kwargs

        self.to(kwargs.get('device','cpu'))

    def __repr__(self):
        kwargs = copy(self.config)
        kwargs['embedding_class'] = self.embedding_model.__class__.__name__
        kwargs['score_class'] = self.score_model.__class__.__name__

        return f"{self.__class__.__name__}({kwargs})"
    

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

        score_model = deepcopy(self.score_model)
        score_model.to(to)

        return Embeddings(strings=strings,
                            counts=counts.detach(),
                            V=V.detach(),
                            score_model=score_model,
                            device=to)

    def train(self,training_matches,max_epochs=1,batch_size=8,grad_acc=1,top_k=1,batch_top_k=False,
                minibatch_size = None,
                embedding_lr = 1e-5,
                transformer_lr = None,
                projection_lr=None,
                embedding_warmup = 0.1,
                score_lr = 10,
                score_warmup=0.5,
                max_grad_norm=1,
                dropout=False,
                progress_bar=True,
                weight_by_counts = False,
                weight_exponent = 0.5,
                **kwargs):
        
        """
        Train the embedding_model and score_model to predict match probabilities
        using the training_matches as a source of "correct" matches.
        Training algorithm uses pairwise similarity learning with high-loss mining
        to fine tune the embedding model to place matched strings near to each 
        other in embedding space, while simulataneously calibrating the score_model
        to predict the match probabilities as a function of cosine distance.
        """
        if minibatch_size is None:
            minibatch_size = batch_size

        num_training_steps = max_epochs*len(training_matches)//(batch_size*grad_acc)

        if transformer_lr is None:
            transformer_lr = embedding_lr
        if projection_lr is None:
            projection_lr = embedding_lr

        if transformer_lr or projection_lr:
            embedding_optimizer = self.embedding_model.config_optimizer(transformer_lr,projection_lr)
            embedding_scheduler = get_cosine_schedule_with_warmup(
                                        embedding_optimizer,
                                        num_warmup_steps=int(embedding_warmup*num_training_steps),
                                        num_training_steps=num_training_steps)
            
            embedding_optimizer.zero_grad()

        if score_lr:
            score_optimizer = self.score_model.config_optimizer(score_lr)
            score_scheduler = get_linear_schedule_with_warmup(
                                        score_optimizer,
                                        num_warmup_steps=int(score_warmup*num_training_steps),
                                        num_training_steps=num_training_steps)
            
            score_optimizer.zero_grad()


        self.embedding_model.train()
        
        if not dropout:
            for m in self.embedding_model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()    
            
        self.history = []
        self.val_scores = []
        for epoch in range(max_epochs):

            global_embeddings = self.embed(training_matches)

            strings = global_embeddings.strings
            V = global_embeddings.V
        
            if weight_by_counts:
                counts = torch.tensor([training_matches.counts[s] for s in strings],device=self.device).float()
                w = counts**weight_exponent
            else:
                w = None

            # Get string id of group names
            groups = torch.tensor([global_embeddings.string_map[training_matches[s]] for s in strings],device=self.device)

            # Normalize weights to make learning rates more general
            w = w/w.mean()

            shuffled_ids = list(range(len(strings)))
            random.shuffle(shuffled_ids)


            for batch_start in tqdm(range(0,len(strings),batch_size),desc=f'training epoch {epoch}',disable=not progress_bar):

                grad_step = (len(self.history) % grad_acc ==0) or (batch_start + batch_size >= len(strings))

                h = {'epoch':epoch,'grad_step':grad_step,'time':datetime.datetime.now()}

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

                with torch.autograd.set_detect_anomaly(True):
                    # Train scoring model only
                    if score_lr:
                        # Make sure gradients are enabled for score model
                        self.score_model.requires_grad_(True)

                        pairwise_global_loss = self.score_model.loss(global_X,global_Y,weights=global_W)

                        global_loss = pairwise_global_loss.nanmean()

                        global_loss.backward()

                        if grad_step:
                            torch.nn.utils.clip_grad_norm_(self.score_model.parameters(),max_norm=max_grad_norm)

                            score_optimizer.step()
                            score_scheduler.step()

                            score_optimizer.zero_grad()
                            

                        h['score_lr'] = score_optimizer.param_groups[0]['lr']
                        h['global_mean_cos'] = global_X.mean().item()
                        try:
                            h.update(self.score_model.get_params())
                        except:
                            pass

                    else:
                        with torch.no_grad():
                            pairwise_global_loss = self.score_model.loss(global_X,global_Y,weights=global_W)
                            global_loss = pairwise_global_loss.nanmean()

                h['global_loss'] = global_loss.item()    

                            
                # Train embedding model
                if (transformer_lr or projection_lr):

                    # Turn off score model updating - only want to train embedding here
                    self.score_model.requires_grad_(False)

                    # Select hard training examples
                    with torch.no_grad():

                        if batch_top_k:
                            _,idx = torch.topk(pairwise_global_loss.flatten(),k=top_k,largest=True)
                            # topk_pairs = idx.T.tolist()
                            topk_i,topk_j = torch.unravel_index(idx, pairwise_global_loss.shape)
                            # topk_j += batch_start
                            topk_i = topk_i.tolist()
                            topk_j = topk_j.tolist()
                        else:
                            
                            _,batch_js = pairwise_global_loss.topk(k=top_k,dim=1,largest=True,sorted=True)
                            topk_i = list(range(len(batch_i)))*top_k
                            # batch_js += batch_start
                            topk_j = [j for m_j in batch_js.T.tolist() for j in m_j]


                    n_pairs = len(topk_i)

                    for m in range(0,n_pairs,minibatch_size):

                        m_i = topk_i[m:m+minibatch_size]
                        m_batch_i = [batch_i[i] for i in m_i]
                        m_V_i = V_i[m_i]
                        
                        batch_j = topk_j[m:m+minibatch_size]
                        V_j = self.embedding_model(strings[batch_j])

                        # Update global embedding cache
                        V[batch_j,:] = V_j.detach()


                        batch_X = (m_V_i*V_j).sum(dim=1)
                        batch_Y = (groups[m_batch_i] == groups[batch_j]).float()
                        
                        if w is not None:
                            batch_W = w[m_batch_i] * w[batch_j]
                        else:
                            batch_W = None

                        batch_loss = self.score_model.loss(batch_X,batch_Y,weights=batch_W).nanmean()

                        batch_loss.backward(retain_graph=True)


                        V_j = None      
                        batch_X = None
                        batch_Y = None
                        batch_W = None


                    try:
                        h['transformer_lr'] = embedding_optimizer.param_groups[1]['lr']
                    except:
                        h['transformer_lr'] = 0

                    try:
                        h['projection_lr'] = embedding_optimizer.param_groups[-1]['lr']
                    except:
                        h['projection_lr'] = 0

                    # Save stats
                    # h['batch_loss'] = batch_loss.item()
                    # h['batch_pos_target'] = batch_Y.mean().item()

                    
                    
                    if grad_step:
                        torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm=max_grad_norm)

                        embedding_optimizer.step()
                        embedding_scheduler.step()

                        embedding_optimizer.zero_grad()


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
        
    def save(self, path):
        # record the classes so load_similarity_model() can re-instantiate them
        config = self.config.copy()
        config.update({
            'score_class':      self.score_model.__class__.__name__,
            'embedding_class':  self.embedding_model.__class__.__name__,
        })
        torch.save({
            'state_dict': self.state_dict(),
            'config': config
        }, path)


def _load_old_similarity_model(f, map_location='cpu', **kwargs):

    checkpoint = torch.load(f, map_location=map_location, **kwargs)

    # If the checkpoint is already a SimilarityModel, return it (for backwards compatibility)
    if isinstance(checkpoint, SimilarityModel):
        return checkpoint
    
    metadata = checkpoint['metadata']
    state_dict = checkpoint['state_dict']

    # Assign default classes 
    metadata['score_class'] = ExpSimilarityScore
    metadata['embedding_class'] = EmbeddingModel

    model = SimilarityModel(**metadata)
    model.load_state_dict(state_dict, strict=False)

    return model


def load_similarity_model(f,map_location='cpu',strict=True,**kwargs):

    # if f is a url, download it
    if isinstance(f, str) and f.startswith('http'):
        response = requests.get(f)
        f = BytesIO(response.content)

    checkpoint = torch.load(f, map_location=map_location, **kwargs)

    config = checkpoint.get('config', {})
    
    # Convert class names to their actual classes
    module = sys.modules[SimilarityModel.__module__]
    if 'score_class' in config:
        config['score_class'] = getattr(module, config['score_class'])
    if 'embedding_class' in config:
        config['embedding_class'] = getattr(module, config['embedding_class'])

    model = SimilarityModel(**config)
    model.load_state_dict(checkpoint['state_dict'],strict=strict)

    return model


    

class ContrastiveModel(SimilarityModel):
    """
    A contrastive learning model that inherits from SimilarityModel but uses NTXent loss
    instead of a score model. This model directly optimizes cosine similarity between
    embeddings of matched strings.
    """
    def __init__(self,
                 embedding_class=EmbeddingModel,
                 loss_type='NTXent',
                 **kwargs):
        # Initialize parent class with ContrastiveLossScore
        super().__init__(
            embedding_class=embedding_class,
            score_class=CosineSimilarityScore,
            **kwargs
        )

        assert loss_type in ['NTXent', 'InfoNCE'], "loss_type must be either 'NTXent' or 'InfoNCE'"
        self.loss_type = loss_type


    def _split_strings(self, strings, chunk_size):
        """Split a list of strings into chunks.
        Note: Need to format as a dict because GradCache assumes 
        a list is list of arguments and passes as *strings.
        """
        if isinstance(strings, str):
            strings = [strings]
        return [{'strings': strings[i:i + chunk_size]} for i in range(0, len(strings), chunk_size)]

    def _contrastive_loss(self, batch_embeddings, temperature=None):
        """Compute contrastive loss using in-batch negatives.
        batch_embeddings contains all embeddings (both anchors and positives)."""
        if temperature is None:
            temperature = self.temperature

        # Ensure we have a tensor of embeddings
        if not isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = torch.stack(batch_embeddings)

        # Split embeddings into anchors and positives
        n = batch_embeddings.shape[0] // 2
        anchors = batch_embeddings[:n]
        positives = batch_embeddings[n:]

        # Compute similarities between anchors and positives
        X = anchors @ positives.T

        # Compute NTXent loss
        Z = torch.exp(X / temperature)

        numerator = torch.diag(Z)
        denominator = torch.sum(Z, dim=1)

        if self.loss_type == 'InfoNCE':
            # Remove positive pairs from denominator
            denominator = denominator - numerator

        loss = -torch.log(numerator / denominator)

        return loss.mean()

    def train(self, training_matches, max_epochs=1, batch_size=32,chunk_size=8,temperature=0.07,
              transformer_lr=1e-5, projection_lr=1e-5, warmup_frac=0.1,
              max_grad_norm=1, dropout=False, progress_bar=True, **kwargs):
        """
        Train the embedding model using contrastive loss with gradient caching.
        Uses in-batch negative sampling where cross-group comparisons within the batch
        serve as negative examples.
        """

        # Calculate number of training steps
        num_training_steps = max_epochs * len(training_matches)
        num_warmup_steps = int(warmup_frac * num_training_steps)

        if transformer_lr or projection_lr:
            embedding_optimizer = self.embedding_model.config_optimizer(transformer_lr, projection_lr)
            embedding_scheduler = get_cosine_schedule_with_warmup(
                embedding_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)
        
        # Initialize gradient cache with proper embedding function and input splitting
        grad_cache = GradCache(
            models=[self.embedding_model],
            chunk_sizes=chunk_size,
            loss_fn=self._contrastive_loss,
            split_input_fn=self._split_strings
        )

        self.embedding_model.train()
        
        if not dropout:
            for m in self.embedding_model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
            
        step = 0
        self.history = []
        self.val_scores = []
        for epoch in range(max_epochs):
            # Get all groups and shuffle them
            groups = list(training_matches.groups.values())
            random.shuffle(groups)

            # Process groups in batches
            for i in tqdm(range(0, len(groups), batch_size), desc=f'training epoch {epoch}', disable=not progress_bar):
                batch_groups = groups[i:i + batch_size]
                
                # For each group, select a random string as anchor and another as positive
                anchor_strings = [random.choice(group) for group in batch_groups]
                pos_strings = [random.choice(group) for group in batch_groups]

                # Combine anchor and positive strings into a single batch
                all_strings = anchor_strings + pos_strings

                # Compute loss using gradient cache
                # The gradient cache will handle the embedding transformation internally
                embedding_optimizer.zero_grad()
                loss = grad_cache(
                    all_strings,
                    temperature=temperature,
                )
           
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)

                embedding_optimizer.step()
                embedding_scheduler.step()

                h = {'epoch': epoch, 'step': step, 'time': datetime.datetime.now()}
                h['batch_loss'] = loss.detach().mean().item()
                h['batch_obs'] = len(batch_groups)
                try:
                    h['transformer_lr'] = embedding_optimizer.param_groups[1]['lr']
                except:
                    h['transformer_lr'] = 0

                try:
                    h['projection_lr'] = embedding_optimizer.param_groups[-1]['lr']
                except:
                    h['projection_lr'] = 0

                self.history.append(h)
                step += 1
            
        return pd.DataFrame(self.history)

