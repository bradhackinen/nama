import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy,deepcopy
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,RobertaModel,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup, logging
from zipfile import ZipFile
import pickle
from io import BytesIO

import nama
from nama.scoring import score_predicted


logging.set_verbosity_error()


class TransformerProjector(nn.Module):
    """
    A basic wrapper around a Hugging Face transformer model.
    Takes a string as input and produces an embedding vector of size d.
    """
    def __init__(self,
                    model_class=RobertaModel,
                    model_name='roberta-base',
                    pooling='pooler',
                    normalize=True,
                    d=128,
                    prompt='',
                    device='cpu',
                    add_upper=True,
                    upper_case=False,
                    **kwargs):

        super().__init__()

        self.model_class = model_class
        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize
        self.d = d
        self.prompt = prompt
        self.add_upper = add_upper
        self.upper_case = upper_case

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.transformer = model_class.from_pretrained(model_name)
        except OSError:
            self.transformer = model_class.from_pretrained(model_name,from_tf=True)

        self.dropout = torch.nn.Dropout(0.5)

        if d:
            # Project embedding to a lower dimension
            # Initialization based on random projection LSH (preserves approximate cosine distances)
            self.projection = torch.nn.Linear(self.transformer.config.hidden_size,d)
            torch.nn.init.normal_(self.projection.weight)
            torch.nn.init.constant_(self.projection.bias,0)

        self.to(device)

    def to(self,device):
        super().to(device)
        self.device = device

    def encode(self,strings):
        if self.prompt is not None:
            strings = [self.prompt + s for s in strings]
        if self.add_upper:
            strings = [s + ' </s> ' + s.upper() for s in strings]
        if self.upper_case:
            strings = [s + ' </s> ' + s.upper() for s in strings]

        try:
            encoded = self.tokenizer(strings,padding=True,truncation=True)
        except Exception as e:
            print(strings)
            raise Exception(e)
        input_ids = torch.tensor(encoded['input_ids']).long()
        attention_mask = torch.tensor(encoded['attention_mask'])

        return input_ids,attention_mask

    def forward(self,strings):

        with torch.no_grad():
            input_ids,attention_mask = self.encode(strings)

            input_ids = input_ids.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)

        # with amp.autocast(self.amp):
        batch_out = self.transformer(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        return_dict=True)

        if self.pooling == 'pooler':
            v = batch_out['pooler_output']
        elif self.pooling == 'mean':
            h = batch_out['last_hidden_state']

            # Compute mean of unmasked token vectors
            h = h*attention_mask[:,:,None]
            v = h.sum(dim=1)/attention_mask.sum(dim=1)[:,None]

        if self.d:
            v = self.projection(v)

        if self.normalize:
            v = v/torch.sqrt((v**2).sum(dim=1)[:,None])

        return v

    def config_optimizer(self,transformer_lr=1e-5,projection_lr=1e-4):

        parameters = list(self.named_parameters())
        grouped_parameters = [
                {
                    'params': [param for name,param in parameters if name.startswith('transformer') and name.endswith('bias')],
                    'weight_decay_rate': 0.0,
                    'lr':transformer_lr,
                    },
                {
                    'params': [param for name,param in parameters if name.startswith('transformer') and not name.endswith('bias')],
                    'weight_decay_rate': 0.0,
                    'lr':transformer_lr,
                    },
                {
                    'params': [param for name,param in parameters if name.startswith('projection')],
                    'weight_decay_rate': 0.0,
                    'lr':projection_lr,
                    },
                ]

        # Drop groups with lr of 0
        grouped_parameters = [p for p in grouped_parameters if p['lr']]

        optimizer = torch.optim.AdamW(grouped_parameters)

        return optimizer


class ExpCosSimilarity(nn.Module):
    """
    A trainable similarity scoring model that estimates the probability
    of a match as the negative exponent of 1+cosine distance between
    embeddings:
        p(match|v_i,v_j) = exp(-alpha*(1-v_i@v_j))
    """
    def __init__(self,alpha=50,**kwargs):

        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def __repr__(self):
        return f'<nama.ExpCosSimilarity with {self.alpha=}>'

    def forward(self,X):
        # Z is a scaled distance measure: Z=0 means that the score should be 1
        Z = self.alpha*(1 - X)
        return torch.clamp(torch.exp(-Z),min=0,max=1.0)

    def loss(self,X,Y,weights=None,decay=1e-6,epsilon=1e-6):

        Z = self.alpha*(1 - X)

        # Put epsilon floor to prevent overflow/undefined results
        # Z = torch.tensor([1e-2,1e-3,1e-6,1e-7,1e-8,1e-9])
        # torch.log(1 - torch.exp(-Z))
        # 1/(1 - torch.exp(-Z))
        with torch.no_grad():
            Z_eps_adjustment = torch.clamp(epsilon-Z,min=0)

        Z += Z_eps_adjustment

        # Cross entropy loss with a simplified and (hopefully) numerically appropriate formula
        # TODO: Stick an epsilon in here to prevent nan?
        loss = Y*Z - torch.xlogy(1-Y,-torch.expm1(-Z))
        # loss = Y*Z - torch.xlogy(1-Y,1-torch.exp(-Z))

        if weights is not None:
            loss *= weights*loss

        if decay:
            loss += decay*self.alpha**2

        return loss

    def score_to_cos(self,score):
        if score > 0:
            return 1 + np.log(score)/self.alpha.item()
        else:
            return -99

    def config_optimizer(self,lr=10):
        optimizer = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=0)

        return optimizer


class ExponentWeights():
    def __init__(self,weighting_exponent=0.5,**kwargs):
        self.exponent = weighting_exponent

    def __call__(self,counts):
        return counts**self.exponent


class EmbeddingSimilarityModel(nn.Module):
    """
    A combined projector/scorer model that produces Embeddings objects
    as its primary output.

    - train() jointly optimizes the projector_model and score_model using
      contrastive learning to learn from a training Matcher.
    """
    def __init__(self,
                    projector_class=TransformerProjector,
                    score_class=ExpCosSimilarity,
                    weighting_class=ExponentWeights,
                    **kwargs):

        super().__init__()

        self.projector_model = projector_class(**kwargs)
        self.score_model = score_class(**kwargs)
        self.weighting_function = weighting_class(**kwargs)

        self.to(kwargs.get('device','cpu'))

    def to(self,device):
        super().to(device)
        self.projector_model.to(device)
        self.score_model.to(device)
        self.device = device

    def save(self,savefile):
        torch.save(self,savefile)

    @torch.no_grad()
    def embed(self,input,to=None,batch_size=64,progress_bar=True,**kwargs):
        """
        Construct an Embeddings object from input strings or a Matcher
        """

        if to is None:
            to = self.device

        if isinstance(input,nama.Matcher):
            strings = input.strings()
            counts = torch.tensor([input.counts[s] for s in strings],device=self.device).float().to(to)

        else:
            strings = list(input)
            counts = torch.ones(len(strings),device=self.device).float().to(to)

        input_loader = DataLoader(strings,batch_size=batch_size,num_workers=0)

        self.projector_model.eval()

        V = None
        batch_start = 0
        with tqdm(total=len(strings),delay=1,desc='Embedding strings',disable=not progress_bar) as pbar:
            for batch_strings in input_loader:

                v = self.projector_model(batch_strings).detach().to(to)

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

    def train(self,training_matcher,max_epochs=1,batch_size=8,
                score_decay=0,regularization=0,
                transformer_lr=1e-5,projection_lr=1e-5,score_lr=10,warmup_frac=0.1,
                max_grad_norm=1,dropout=False,
                validation_matcher=None,target='F1',restore_best=True,val_seed=None,
                validation_interval=1000,early_stopping=True,early_stopping_patience=3,
                verbose=False,progress_bar=True,
                **kwargs):

        """
        Train the projector_model and score_model to predict match probabilities
        using the training_matcher as a source of "correct" matches.
        Training algorithm uses contrastive learning with hard-positive
        and hard-negative mining to fine tune the projector model to place
        matched strings near to each other in embedding space, while
        simulataneously calibrating the score_model to predict the match
        probabilities as a function of cosine distance
        """

        if validation_matcher is None:
            early_stopping = False
            restore_best = False

        num_training_steps = max_epochs*len(training_matcher)//batch_size
        num_warmup_steps = int(warmup_frac*num_training_steps)

        if transformer_lr or projection_lr:
            embedding_optimizer = self.projector_model.config_optimizer(transformer_lr,projection_lr)
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

            global_embeddings = self.embed(training_matcher)

            strings = global_embeddings.strings
            V = global_embeddings.V
            w = global_embeddings.w

            groups = torch.tensor([global_embeddings.string_map[training_matcher[s]] for s in strings],device=self.device)

            # Normalize weights to make learning rates more general
            if w is not None:
                w = w/w.mean()

            shuffled_ids = list(range(len(strings)))
            random.shuffle(shuffled_ids)

            if dropout:
                self.projector_model.train()
            else:
                self.projector_model.eval()

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
                V_i = self.projector_model(strings[batch_i])

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

                    global_loss = self.score_model.loss(global_X,global_Y,weights=global_W,decay=score_decay)

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

                # Train projector model
                if (transformer_lr or projection_lr) and step <= num_warmup_steps + num_training_steps:

                    # Turn off score model updating - only want to train projector here
                    self.score_model.requires_grad_(False)

                    # Select hard training examples
                    with torch.no_grad():
                        batch_j = global_loss.argmax(dim=1).flatten()

                        if w is not None:
                            batch_W = torch.outer(w[batch_i],w[batch_j])
                        else:
                            batch_W = None

                    # Train the model on the selected high-loss pairs
                    V_j = self.projector_model(strings[batch_j.tolist()])

                    # Update global embedding cache
                    V[batch_j,:] = V_j.detach()

                    batch_X = V_i@V_j.T
                    batch_Y = (groups[batch_i][:,None] == groups[batch_j][None,:]).float()
                    h['batch_obs'] = len(batch_i)*len(batch_j)

                    batch_loss = self.score_model.loss(batch_X,batch_Y,weights=batch_W)

                    if regularization:
                        # Apply Global Orthogonal Regularization from https://arxiv.org/abs/1708.06320
                        gor_Y = (groups[batch_i][:,None] != groups[batch_i][None,:]).float()
                        gor_n = gor_Y.sum()
                        if gor_n > 1:
                            gor_X = (V_i@V_i.T)*gor_Y
                            gor_m1 = 0.5*gor_X.sum()/gor_n
                            gor_m2 = 0.5*(gor_X**2).sum()/gor_n
                            batch_loss += regularization*(gor_m1 + torch.clamp(gor_m2 - 1/self.projector_model.d,min=0))

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
                step += 1

                if (validation_matcher is not None) and not (step % validation_interval):

                    validation = len(self.validation_scores)
                    val_scores = self.test(validation_matcher)
                    val_scores['step'] = step - 1
                    val_scores['epoch'] = epoch
                    val_scores['validation'] = validation

                    self.validation_scores.append(val_scores)

                    # Print validation stats
                    if verbose:
                        print(f'\nValidation results at step {step} (current epoch {epoch})')
                        for k,v in val_scores.items():
                            print(f'    {k}: {v:.4f}')

                        print(list(self.score_model.named_parameters()))

                    # Update best saved model
                    if restore_best:
                        if val_scores[target] >= max(h[target] for h in self.validation_scores):
                            best_state = deepcopy({
                                            'state_dict':self.state_dict(),
                                            'val_scores':val_scores
                                            })

                    if early_stopping and (validation - best_state['val_scores']['validation'] > early_stopping_patience):
                        print(f'Stopping training ({early_stopping_patience} validation checks since best validation score)')
                        break

        if restore_best:
            print(f"Restoring to best state (step {best_state['val_scores']['step']}):")
            for k,v in best_state['val_scores'].items():
                print(f'    {k}: {v:.4f}')

            self.to('cpu')
            self.load_state_dict(best_state['state_dict'])
            self.to(self.device)

        return pd.DataFrame(self.history)

    def predict(self,input,**kwargs):
        embeddings = self.embed(input,**kwargs)
        return embeddings.predict(**kwargs)

    def test(self,gold_matcher,batch_size=32):

        embeddings = self.embed(gold_matcher)
        predicted = embeddings.predict(gold_matcher)

        scores = score_predicted(predicted,gold_matcher,use_counts=True)

        return scores


class Embeddings(nn.Module):
    """
    Stores embeddings for a fixed array of strings and provides methods for
    clustering the strings to create Matcher objects according to different
    algorithms.
    """
    def __init__(self,strings,V,score_model,weighting_function,counts,device='cpu'):
        super().__init__()

        self.strings = np.array(list(strings))
        self.string_map = {s:i for i,s in enumerate(strings)}
        self.V = V
        self.counts = counts
        self.w = weighting_function(counts)
        self.score_model = score_model
        self.weighting_function = weighting_function
        self.device = device

        self.to(device)

    def __repr__(self):
        return f'<nama.Embeddings containing {self.V.shape[1]}-d vectors for {len(self)} strings'

    def to(self,device):
        super().to(device)
        self.V = self.V.to(device)
        self.counts = self.counts.to(device)
        self.w = self.w.to(device)
        self.score_model.to(device)
        self.device = device

    def save(self,f):
        """
        Save embeddings in a simple custom zipped archive format (torch.save
        works too, but it requires huge amounts of memory to serialize large
        embeddings objects).
        """
        with ZipFile(f,'w') as zip:

            # Write score model
            zip.writestr('score_model.pkl',pickle.dumps(self.score_model))

            # Write score model
            zip.writestr('weighting_function.pkl',pickle.dumps(self.weighting_function))

            # Write string info
            strings_df = pd.DataFrame().assign(
                                        string=self.strings,
                                        count=self.counts.to('cpu').numpy())
            zip.writestr('strings.csv',strings_df.to_csv(index=False))

            # Write embedding vectors
            byte_io = BytesIO()
            np.save(byte_io,self.V.to('cpu').numpy(),allow_pickle=False)
            zip.writestr('V.npy',byte_io.getvalue())

    def __getitem__(self,arg):
        """
        Slice a matcher
        """
        if isinstance(arg,slice):
            i = arg
        elif isinstance(arg,nama.Matcher):
            return self[arg.strings()]
        elif hasattr(arg,'__iter__'):
            # Return a subset of the embeddings and their weights
            string_map = self.string_map
            i = [string_map[s] for s in arg]

            if i == list(range(len(self))):
                # Just selecting the whole matcher - no need to slice the embedding
                return copy(self)
        else:
            raise ValueError(f'Unknown slice input type ({type(input)}). Can only slice Embedding with a slice, matcher, or iterable.')

        new = copy(self)
        new.strings = self.strings[i]
        new.V = self.V[i]
        new.counts = self.counts[i]
        new.w = self.w[i]
        new.string_map = {s:i for i,s in enumerate(new.strings)}

        return new

    def embed(self,matcher):
        """
        Construct updated Embeddings with counts from the input Matcher
        """
        new = self[matcher]
        new.counts = torch.tensor([matcher.counts[s] for s in new.strings],device=self.device)
        new.w = new.weighting_function(new.counts)

        return new

    def __len__(self):
        return len(self.strings)

    def _matcher_to_group_ids(self,matcher):
        group_id_map = {g:i for i,g in enumerate(matcher.groups.keys())}
        group_ids = torch.tensor([group_id_map[matcher[s]] for s in self.strings]).to(self.device)
        return group_ids

    def _group_ids_to_matcher(self,group_ids):
        if isinstance(group_ids,torch.Tensor):
            group_ids = group_ids.to('cpu').numpy()

        strings = self.strings
        counts = self.counts.to('cpu').numpy()

        # Sort by group and string count
        g_sort = np.lexsort((counts,group_ids))
        group_ids = group_ids[g_sort]
        strings = strings[g_sort]
        counts = counts[g_sort]

        # Identify group boundaries and split locations
        split_locs = np.nonzero(group_ids[1:] != group_ids[:-1])[0] + 1

        # Get grouped strings as separate arrays
        groups = np.split(strings,split_locs)

        # Build the matcher
        matcher = nama.Matcher()
        matcher.counts = Counter({s:int(c) for s,c in zip(strings,counts)})
        matcher.labels = {s:g[-1] for g in groups for s in g}
        matcher.groups = {g[-1]:list(g) for g in groups}

        return matcher

    @torch.no_grad()
    def _fast_predict(self,threshold=0.5,base_matcher=None,progress_bar=True,batch_size=64):

        if base_matcher is not None:
            # self = self.embed(base_matcher)
            group_ids = self._matcher_to_group_ids(base_matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        for batch_start in tqdm(range(0,len(self),batch_size),
                                    delay=1,desc='Predicting matches',disable=not progress_bar):

            i_slice = slice(batch_start,batch_start+batch_size)
            j_slice = slice(batch_start+1,None)

            g_i = group_ids[i_slice]
            g_j = group_ids[j_slice]

            # Find j's with jaccard > threshold ("matches")
            batch_matched = (V[i_slice]@V[j_slice].T >= cos_threshold) \
                            * (g_i[:,None] != g_j[None,:])

            for k,matched in enumerate(batch_matched):
                if matched.any():
                    # Get the group ids of the matched j's
                    matched_groups = g_j[matched]

                    # Identify all embeddings in these groups
                    ids_to_group = torch.isin(group_ids,matched_groups)

                    # Assign all matched embeddings to the same group
                    group_ids[ids_to_group] = g_i[k].clone()

        return self._group_ids_to_matcher(group_ids)

    @torch.no_grad()
    def predict(self,
                threshold=0.5,
                group_threshold=None,
                separate_strings=[],
                base_matcher=None,
                batch_size=64,
                progress_bar=True):

        """
        Unite embedding strings according to predicted pairwise similarity.

        - "base_matcher" will be used to inialize the group_ids before uniting new matches
        - "theshold" sets the minimimum match similarity required to unite two strings.
            - Note that strings with similarity<threshold can end up matched if they are
              linked by a chain of sufficiently similar strings (matching is transitive).
              "group_threshold" can be used to add an additional constraing on the minimum
              similarity within each group.
        - "group_threshold" sets the minimum similarity required within a single group.
          If "group_threshold" != None, string pairs with similarity>threshold are identified
          and stored in order of similarity. Highest similarity strings are matched first,
          and before each time a pair of strings is united, the function checks if this will
          result in grouping any two strings with similarity<group_threshold. If so, this pair
          is skipped. This version of the algorithm is slower than the one used when
          "group_threshold=None.
        - "separate_strings" takes a list of strings that should never be united with each
          other (these strings will still be united with other strings)

        returns: Matcher object
        """
        # Use the faster prediction algorithm if possible
        if not (group_threshold or separate_strings):

            return self._fast_predict(
                        threshold=threshold,
                        base_matcher=base_matcher,
                        batch_size=batch_size,
                        progress_bar=progress_bar)

        if base_matcher is not None:
            # self = self.embed(base_matcher)
            group_ids = self._matcher_to_group_ids(base_matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)
        if group_threshold is not None:
            separate_cos = self.score_model.score_to_cos(group_threshold)

        # First collect all pairs to match (can be memory intensive!)
        matches = []
        cos_scores = []
        for batch_start in tqdm(range(0,len(self),batch_size),
                                    desc='Scoring pairs',
                                    delay=1,disable=not progress_bar):

            i_slice = slice(batch_start,batch_start+batch_size)
            j_slice = slice(batch_start+1,None)

            # Find j's with jaccard > threshold ("matches")
            batch_cos = V[i_slice]@V[j_slice].T

            # Search upper diagonal entries only
            # (note j_slice starting index is offset by one)
            batch_cos = torch.triu(batch_cos)

            bi,bj = torch.nonzero(batch_cos >= cos_threshold,as_tuple=True)

            if len(bi):
                # Convert batch index locations to global index locations
                i = bi + batch_start
                j = bj + batch_start + 1

                cos = batch_cos[bi,bj]

                # Can skip strings that are already matched in the base matcher
                unmatched = group_ids[i] != group_ids[j]
                i = i[unmatched]
                j = j[unmatched]
                cos = cos[unmatched]

                if len(i):
                    batch_matches = torch.hstack([i[:,None],j[:,None]])

                    matches.append(batch_matches.to('cpu').numpy())
                    cos_scores.append(cos.to('cpu').numpy())

        # Then unite the pairs in priority order, checking for violations of the
        # separation arguments
        if matches:
            matches = np.vstack(matches)
            cos_scores = np.hstack(cos_scores).T

            # Sort matches in descending order of score
            m_sort = cos_scores.argsort()[::-1]
            matches = matches[m_sort]

            # Set up tensors
            matches = torch.tensor(matches).to(self.device)
            separate_strings = set(separate_strings)
            separated = torch.tensor([s in separate_strings for s in self.strings]).to(self.device)

            n_matches = matches.shape[0]
            with tqdm(total=n_matches,desc='Uniting matches',
                        delay=1,disable=not progress_bar) as p_bar:

                while len(matches):

                    # Select the current match pair and remove it from the queue
                    match_pair = matches[0]
                    matches = matches[1:]

                    # Get the groups of the current match pair
                    g = group_ids[match_pair]

                    # Identify which strings should be united
                    to_unite = (group_ids == g[0]) | (group_ids == g[1])

                    # Flag whether uniting this pair will unite any separated strings
                    any_separated = separated[to_unite].sum() > 1

                    # Flag whether the new group will have three or more strings
                    singletons = to_unite.sum() < 3

                    if any_separated:
                        unite_ok = False
                    else:
                        if singletons:
                            unite_ok = True
                        else:
                            if group_threshold is None:
                                unite_ok = True
                            else:
                                V0 = V[group_ids == g[0],:]
                                V1 = V[group_ids == g[1],:]

                                unite_ok = (V0@V1.T).min() >= separate_cos

                    if unite_ok:

                        # Unite groups
                        group_ids[to_unite] = g[0]

                        # If we are uniting more than two strings, we can eliminate
                        # some redundant matches in the queue
                        if not singletons:
                            # Removed queued matches that are now in the same group
                            matches = matches[group_ids[matches[:,0]] != group_ids[matches[:,1]]]
                    else:
                        # Remove queued matches connecting these groups
                        matches = matches[torch.isin(group_ids[matches[:,0]],g,invert=True) \
                                            | torch.isin(group_ids[matches[:,1]],g,invert=True)]

                    # Update progress bar
                    p_bar.update(n_matches - matches.shape[0])
                    n_matches = matches.shape[0]

        return self._group_ids_to_matcher(group_ids)

    @torch.no_grad()
    def voronoi(self,seed_strings,threshold=0,base_matcher=None,progress_bar=True,batch_size=64):
        """
        Unite embedding strings with each string's most similar seed string.

        - "base_matcher" will be used to inialize the group_ids before uniting new matches
        - "theshold" sets the minimimum match similarity required between a string and seed string
          for the string to be matched. (i.e., setting theshold=0 will result in every embedding
          string to be matched its nearest seed string, while setting threshold=0.9 will leave
          strings that have similarity<0.9 with their nearest seed string unaffected)

        returns: Matcher object
        """

        if base_matcher is not None:
            # self = self.embed(base_matcher)
            group_ids = self._matcher_to_group_ids(base_matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        seed_ids = torch.tensor([self.string_map[s] for s in seed_strings]).to(self.device)
        V_seed = V[seed_ids]
        g_seed = group_ids[seed_ids]
        is_seed = torch.zeros(V.shape[0],dtype=torch.bool).to(self.device)
        is_seed[g_seed] = True

        for batch_start in tqdm(range(0,len(self),batch_size),
                                    delay=1,desc='Predicting matches',disable=not progress_bar):

            batch_slice = slice(batch_start,batch_start+batch_size)

            batch_cos = V[batch_slice]@V_seed.T

            max_cos,max_seed = torch.max(batch_cos,dim=1)

            # Get batch index locations where score > threshold
            batch_i = torch.nonzero(max_cos > cos_threshold)

            if len(batch_i):
                # Drop seed strings from matches (otherwise numerical precision
                # issues can allow seed strings to match to other strings)
                batch_i = batch_i[~is_seed[batch_slice][batch_i]]

                if len(batch_i):
                    # Get indices of matched strings
                    i = batch_i + batch_start

                    # Assign matched strings to the seed string's group
                    group_ids[i] = g_seed[max_seed[batch_i]]

        return self._group_ids_to_matcher(group_ids)

    @torch.no_grad()
    def _batch_scored_pairs(self,group_ids,batch_start,batch_size,
                            is_match=None,
                            min_score=None,max_score=None,
                            min_loss=None,max_loss=None):

        strings = self.strings
        V = self.V
        w = self.w

        # Create simple slice objects to avoid creating copies with advanced indexing
        i_slice = slice(batch_start,batch_start+batch_size)
        j_slice = slice(batch_start+1,None)

        X = V[i_slice]@V[j_slice].T
        Y = (group_ids[i_slice,None] == group_ids[None,j_slice]).float()
        if w is not None:
            W = w[i_slice,None]*w[None,j_slice]
        else:
            W = None

        scores = self.score_model(X)
        loss = self.score_model.loss(X,Y,weights=W)

        # Search upper diagonal entries only
        # (note j_slice starting index is offset by one)
        scores = torch.triu(scores)

        # Filter by match type
        if is_match is not None:
            if is_match:
                scores *= Y
            else:
                scores *= (1 - Y)

        # Filter by min score
        if min_score is not None:
            scores *= (scores >= min_score)

        # Filter by max score
        if max_score is not None:
            scores *= (scores <= max_score)

        # Filter by min loss
        if min_loss is not None:
            scores *= (loss >= min_loss)

        # Filter by max loss
        if max_loss is not None:
            scores *= (loss <= max_loss)

        # Collect scored pairs
        i,j = torch.nonzero(scores,as_tuple=True)

        pairs = np.hstack([
                                strings[i.cpu().numpy() + batch_start][:,None],
                                strings[j.cpu().numpy() + (batch_start + 1)][:,None]
                                ])

        pair_groups = np.hstack([
                                strings[group_ids[i + batch_start].cpu().numpy()][:,None],
                                strings[group_ids[j + (batch_start + 1)].cpu().numpy()][:,None]
                                ])

        pair_scores = scores[i,j].cpu().numpy()
        pair_losses = loss[i,j].cpu().numpy()

        return pairs,pair_groups,pair_scores,pair_losses

    def iter_scored_pairs(self,matcher=None,batch_size=64,progress_bar=True,**kwargs):

        if matcher is not None:
            self = self.embed(matcher)
            group_ids = self._matcher_to_group_ids(matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        for batch_start in tqdm(range(0,len(self),batch_size),desc='Scoring pairs',disable=not progress_bar):
            pairs,pair_groups,scores,losses = self._batch_scored_pairs(self,group_ids,batch_start,batch_size,**kwargs)
            for (s0,s1),(g0,g1),score,loss in zip(pairs,pair_groups,scores,losses):
                yield {
                        'string0':s0,
                        'string1':s1,
                        'group0':g0,
                        'group1':g1,
                        'score':score,
                        'loss':loss,
                        }


def load_embeddings(f):
    """
    Load embeddings from custom zipped archive format
    """
    with ZipFile(f,'r') as zip:
        score_model = pickle.loads(zip.read('score_model.pkl'))
        weighting_function = pickle.loads(zip.read('weighting_function.pkl'))
        strings_df = pd.read_csv(zip.open('strings.csv'),na_filter=False)
        V = np.load(zip.open('V.npy'))

        return Embeddings(
                            strings=strings_df['string'].values,
                            counts=torch.tensor(strings_df['count'].values),
                            score_model=score_model,
                            weighting_function=weighting_function,
                            V=torch.tensor(V)
                            )


def load_similarity_model(f,map_location='cpu',**kwargs):
    return torch.load(f,map_location=map_location,**kwargs)
