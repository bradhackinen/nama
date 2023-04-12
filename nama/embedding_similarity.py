import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaModel, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, logging
from zipfile import ZipFile
import pickle
from io import BytesIO

from .scoring import score_predicted
from .matcher import Matcher


logging.set_verbosity_error()


def load_similarity_model(model_file, map_location='cpu', **kwargs):
    """
    Load a similarity model from a saved model file.

    Parameters
    ----------
    model_file : str
        the file to load the model from
    map_location : str, optional
        This is a string that specifies the device to load the model on, defaults to "cpu".

    -------
    model: EmbeddingSimilarityModel
    A similarity model
    """
    return torch.load(model_file, map_location=map_location, **kwargs)


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
        """
        Initializes the TransformerProjector class.

        Parameters
        ----------
        model_class: The transformer model class
        model_name: str, optional (default='roberta-base')
            The name of the pretrained model to use.
        pooling: str, optional (default='pooler')
            The pooling method to use. Options are:
        normalize: bool, optional (default=True)
            Whether to normalize the embeddings to unit length.
        d: int, optional (default=128)
            The dimension of the embedding space. If you want to use the original embedding space, set this to None.
        prompt: str
            The prompt to use for the essay.
        device: str, optional (default='cpu')
            The device to run the model on.
        add_upper: bool, optional (default=True)
            If True, adds the upper-cased version of the prompt to the prompt.
        upper_case: bool, optional (default=False)
            If True, the input will be converted to upper case.
        """
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
            self.transformer = model_class.from_pretrained(
                model_name, from_tf=True)

        self.dropout = torch.nn.Dropout(0.5)

        if d:
            # Project embedding to a lower dimension
            # Initialization based on random projection LSH (preserves
            # approximate cosine distances)
            self.projection = torch.nn.Linear(
                self.transformer.config.hidden_size, d)
            torch.nn.init.normal_(self.projection.weight)
            torch.nn.init.constant_(self.projection.bias, 0)

        self.to(device)

    def to(self, device):
        """
        Moves the model to the specified device

        Parameters
        ----------
        device: str
            The device to which the model is to be moved
        """
        super().to(device)
        self.device = device

    def encode(self, strings):
        """
        Encodes the strings using the tokenizer

        Parameters
        ----------
        strings: list of str
            A list of strings to encode

        Returns
        -------
        input_ids: torch.tensor
            Input ids tensor
        attention_mask: torch.tensor
            Attention mask tensor
        """
        if self.prompt is not None:
            strings = [self.prompt + s for s in strings]
        if self.add_upper:
            strings = [s + ' </s> ' + s.upper() for s in strings]
        if self.upper_case:
            strings = [s + ' </s> ' + s.upper() for s in strings]

        try:
            encoded = self.tokenizer(strings, padding=True, truncation=True)
        except Exception as e:
            print(strings)
            raise Exception(e)
        input_ids = torch.tensor(encoded['input_ids']).long()
        attention_mask = torch.tensor(encoded['attention_mask'])

        return input_ids, attention_mask

    def forward(self, strings):
        """
        Encodes the string into batches, and then passes them through the transformer model

        Parameters
        ----------
        strings: list of str
            A list of strings to embed

        Returns
        -------
        v: torch.tensor
            The last hidden state of the transformer
        """
        with torch.no_grad():
            input_ids, attention_mask = self.encode(strings)

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
            h = h * attention_mask[:, :, None]
            v = h.sum(dim=1) / attention_mask.sum(dim=1)[:, None]

        if self.d:
            v = self.projection(v)

        if self.normalize:
            v = v / torch.sqrt((v**2).sum(dim=1)[:, None])

        return v

    def config_optimizer(self, transformer_lr=1e-5, projection_lr=1e-4):
        """
        Sets the configuration of the TransformerProjector optimizer

        Parameters
        ----------
        transformer_lr: float
            Learning rate for the transformer
        projection_lr: float
            Learning rate for the projection layer

        Returns
        -------
        optimizer: torch.optim.AdamW
            The configured optimizer
        """

        parameters = list(self.named_parameters())
        grouped_parameters = [{'params': [param for name,
                                          param in parameters if name.startswith('transformer') and name.endswith('bias')],
                               'weight_decay_rate': 0.0,
                               'lr': transformer_lr,
                               },
                              {'params': [param for name,
                                          param in parameters if name.startswith('transformer') and not name.endswith('bias')],
                               'weight_decay_rate': 0.0,
                               'lr': transformer_lr,
                               },
                              {'params': [param for name,
                                          param in parameters if name.startswith('projection')],
                               'weight_decay_rate': 0.0,
                               'lr': projection_lr,
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

    embeddings: p(match|v_i,v_j) = exp(-alpha*(1-v_i@v_j))
    """

    def __init__(self, alpha=50, **kwargs):
        """
        Initializes an a trainable alpha.

        Parameters
        ----------
        alpha: float, optional
            The alpha parameter in the paper, defaults to 50.
        """

        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def __repr__(self):
        """
        Returns a string representation of the object.
        """
        return f'<nama.ExpCosSimilarity with {self.alpha=}>'

    def forward(self, X):
        """
        Scores a set of embedding similarities, returns a score between 0 and 1.

        Parameters
        ----------
        X: torch tensor
            the distance between the embeddings vectors

        Returns
        -------
        The score of the embeddings: torch tensor
        """
        # Z is a scaled distance measure: Z=0 means that the score should be 1
        Z = self.alpha * (1 - X)
        return torch.clamp(torch.exp(-Z), min=0, max=1.0)

    def loss(self, X, Y, weights=None, decay=1e-6, epsilon=1e-6):
        """
        Calculates the cross entropy loss with a simplified and numerically appropriate formula.
        $ $

        Parameters
        ----------
        X: torch tensor
            The set of embedding similarities
        Y: torch tensor
            The true labels
        weights: torch tensor, optional
            a tensor used to weights for the loss
        decay: float, optional
            This is the regularization parameter. It's a hyperparameter that can be tuned, defaults to 1e-6.
        epsilon: float, optional
            This is a small number that is added to the denominator to prevent division by zero, defaults to 1e-6.

        Returns
        -------
        The cross entropy loss of the embeddings: torch tensor
        """

        Z = self.alpha * (1 - X)

        # Put epsilon floor to prevent overflow/undefined results
        with torch.no_grad():
            Z_eps_adjustment = torch.clamp(epsilon - Z, min=0)

        Z += Z_eps_adjustment

        loss = Y * Z - torch.xlogy(1 - Y, -torch.expm1(-Z))

        if weights is not None:
            loss *= weights * loss

        if decay:
            loss += decay * self.alpha**2

        return loss

    def score_to_cos(self, score):
        """
        It takes a score and returns a cosine similarity.

        Parameters
        ----------
        score: float
            the score of the item

        Returns
        -------
        The cosine similarity between the two vectors: float
        """
        if score > 0:
            return 1 + np.log(score) / self.alpha.item()
        else:
            return -99

    def config_optimizer(self, lr=10):
        """
        Sets the configuration of the ExpCosSimilarity optimizer

        Parameters
        ----------
        lr: float
            Learning rate for the loss function

        Returns
        -------
        optimizer: torch.optim.AdamW
            The configured optimizer
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0)

        return optimizer


class ExponentWeights():
    """The ExponentWeights class is used to calculate the weighted counts with a specified exponent."""

    def __init__(self, weighting_exponent=0.5, **kwargs):
        """
        Initialize the class with the specified weighting exponent.

        Parameters
        ----------
        weighting_exponent : float, optional
            The exponent to use for the weighting of the counts (default is 0.5).
        """
        self.exponent = weighting_exponent

    def __call__(self, counts):
        """
        Calculate the weighted counts using the specified exponent.

        Parameters
        ----------
        counts : np.ndarray
            The counts to be weighted.

        Returns
        -------
        np.ndarray
            The weighted counts.
        """
        return counts**self.exponent


class EmbeddingSimilarityModel(nn.Module):
    """
    A combined projector/scorer model that produces Embeddings objects
    as its primary output.
    """
    # TODO add save for model hyper params

    def __init__(self,
                 projector_class=TransformerProjector,
                 score_class=ExpCosSimilarity,
                 weighting_class=ExponentWeights,
                 **kwargs):
        """
        Initialize the EmbeddingSimilarityModel with the projector, score, and weighting models.

        Parameters
        ----------
        projector_class : The class of the projector model
        score_class : the similarity function used to compare the projected query and projected document
        weighting_class : This is the function that will be used to weight the scores of the different documents
        """
        super().__init__()

        self.projector_model = projector_class(**kwargs)
        self.score_model = score_class(**kwargs)
        self.weighting_function = weighting_class(**kwargs)

        self.to(kwargs.get('device', 'cpu'))

    def to(self, device):
        """
        Moves the model to a specified device

        Parameters
        ----------
        device : the device to run the model on
        """
        super().to(device)
        self.projector_model.to(device)
        self.score_model.to(device)
        self.device = device

    def save(self, savefile):
        """
        Saves the model to a file.

        Parameters
        ----------
        savefile : the file to save the model to
        """
        torch.save(self, savefile)

    def load_embeddings(self, f):
        """
        Load embeddings from custom embedding archive format.

        Parameters
        ----------
        f : the file path to the embeddings
        """
        raise NotImplementedError

        with ZipFile(f, 'r') as zip:
            score_model = pickle.loads(zip.read('score_model.pkl'))
            weighting_function = pickle.loads(
                zip.read('weighting_function.pkl'))
            strings_df = pd.read_csv(zip.open('strings.csv'), na_filter=False)
            V = np.load(zip.open('V.npy'))

            return Embeddings(
                strings=strings_df['string'].values,
                counts=torch.tensor(strings_df['count'].values),
                score_model=score_model,
                weighting_function=weighting_function,
                V=torch.tensor(V)
            )

    @torch.no_grad()
    def embed(
            self,
            input,
            to=None,
            batch_size=64,
            progress_bar=True,
            **kwargs):
        """
        Construct an Embeddings object from input strings or a Matcher.

        Parameters
        ----------
        input : a list of strings or a Matcher object
        to : the device to put the embeddings on. If you're using a GPU, you'll want to put them on the GPU (optional)
        batch_size : The number of strings to embed at once, defaults to 64 (optional)
        progress_bar : Whether to show a progress bar, defaults to True (optional)

        Returns
        -------
        An Embeddings object.
        """

        if to is None:
            to = self.device

        if isinstance(input, Matcher):
            strings = input.strings()
            counts = torch.tensor(
                [input.counts[s] for s in strings], device=self.device).float().to(to)

        else:
            strings = list(input)
            counts = torch.ones(
                len(strings),
                device=self.device).float().to(to)

        input_loader = DataLoader(
            strings, batch_size=batch_size, num_workers=0)

        self.projector_model.eval()

        V = None
        batch_start = 0
        with tqdm(total=len(strings), delay=1, desc='Embedding strings', disable=not progress_bar) as pbar:
            for batch_strings in input_loader:

                v = self.projector_model(batch_strings).detach().to(to)

                if V is None:
                    # Use v to determine dim and dtype of pre-allocated embedding tensor
                    # (Pre-allocating avoids duplicating tensors with a big .cat() operation)
                    V = torch.empty(
                        len(strings), v.shape[1], device=to, dtype=v.dtype)

                V[batch_start:batch_start + len(batch_strings), :] = v

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

    def train(
            self,
            training_matcher,
            max_epochs=1,
            batch_size=8,
            score_decay=0,
            regularization=0,
            transformer_lr=1e-5,
            projection_lr=1e-5,
            score_lr=10,
            warmup_frac=0.1,
            max_grad_norm=1,
            dropout=False,
            validation_matcher=None,
            target='F1',
            restore_best=True,
            val_seed=None,
            validation_interval=1000,
            early_stopping=True,
            early_stopping_patience=3,
            verbose=False,
            progress_bar=True,
            **kwargs):
        """
        Trains the projector_model and score_model to predict match probabilities
        using the training_matcher as a source of "correct" matches.

        Training algorithm uses contrastive learning with hard-positive
        and hard-negative mining to fine tune the projector model to place
        matched strings near to each other in embedding space, while
        simultaneously calibrating the score_model to predict the match
        probabilities as a function of cosine distance

        Parameters
        ----------
        training_matcher : :class:`Matching`
            A Matching object with training data.
        max_epochs : int, optional (default=1)
            The number of epochs to train for.
        batch_size : int, optional (default=8)
            The number of strings to use in each training batch.
        score_decay : float, optional (default=0)
            A regularization parameter for the score model, similar to L2 regularization, but applied to the cosine distance between embeddings.
        regularization : float, optional (default=0)
        transformer_lr : float
            Learning rate for the transformer.
        projection_lr : float
            Learning rate for the projection layer.
        score_lr : float, optional (default=10)
            Learning rate for the score model.
        warmup_frac : float
            The fraction of training steps to use for the warm-up period.
        max_grad_norm : float, optional (default=1)
            The maximum gradient norm to clip to.
        dropout : bool, optional (default=False)
            Whether to use dropout in the transformer model.
        validation_matcher : :class:`Matching`, optional
            A matcher object used to evaluate the model during training.
        target : str, optional (default='F1')
            The metric to use for early stopping.
        restore_best : bool, optional (default=True)
            If True, the model will be restored to the state with the best validation score.
        val_seed : int, optional
            A random seed for validation, if desired.
        validation_interval : int, optional (default=1000)
            How often to run validation checks.
        early_stopping : bool, optional (default=True)
            If True, stop training if the validation score doesn't improve for `early_stopping_patience` epochs.
        early_stopping_patience : int, optional (default=3)
            How many validation checks to wait before stopping training.
        verbose : bool, optional (default=False)
            Print out the training progress, defaults to False (optional)
        progress_bar: bool
            Whether to show a progress bar during training, defaults to True (optional)

        Returns
        -------
        The return is a tuple of two dataframes. The first dataframe is the history of the
        training. The second dataframe is the validation scores.
        """
        best_state = None
        if validation_matcher is None:
            early_stopping = False
            restore_best = False

        num_training_steps = max_epochs * len(training_matcher) // batch_size
        num_warmup_steps = int(warmup_frac * num_training_steps)

        if transformer_lr or projection_lr:
            embedding_optimizer = self.projector_model.config_optimizer(
                transformer_lr, projection_lr)
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
        self.validation_scores = []
        for epoch in range(max_epochs):

            global_embeddings = self.embed(training_matcher)

            strings = global_embeddings.strings
            V = global_embeddings.V
            w = global_embeddings.w

            groups = torch.tensor(
                [global_embeddings.string_map[training_matcher[s]] for s in strings], device=self.device)

            # Normalize weights to make learning rates more general
            if w is not None:
                w = w / w.mean()

            shuffled_ids = list(range(len(strings)))
            random.shuffle(shuffled_ids)

            if dropout:
                self.projector_model.train()
            else:
                self.projector_model.eval()

            for batch_start in tqdm(
                    range(
                        0,
                        len(strings),
                        batch_size),
                    desc=f'training epoch {epoch}',
                    disable=not progress_bar):

                h = {'epoch': epoch, 'step': step}

                batch_i = shuffled_ids[batch_start:batch_start + batch_size]

                # Recycle ids from the beginning to pad the last batch if
                # necessary
                if len(batch_i) < batch_size:
                    batch_i = batch_i + \
                        shuffled_ids[:(batch_size - len(batch_i))]

                """
                Find highest loss match for each batch string (global search)

                Note: If we compute V_i with dropout enabled, it will add noise
                to the embeddings and prevent the same pairs from being selected
                every time.
                """
                V_i = self.projector_model(strings[batch_i])

                # Update global embedding cache
                V[batch_i, :] = V_i.detach()

                with torch.no_grad():

                    global_X = V_i @ V.T
                    global_Y = (groups[batch_i][:, None]
                                == groups[None, :]).float()

                    if w is not None:
                        global_W = torch.outer(w[batch_i], w)
                    else:
                        global_W = None

                # Train scoring model only
                if score_lr:
                    # Make sure gradients are enabled for score model
                    self.score_model.requires_grad_(True)

                    global_loss = self.score_model.loss(
                        global_X, global_Y, weights=global_W, decay=score_decay)

                    score_optimizer.zero_grad()
                    global_loss.nanmean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.score_model.parameters(), max_norm=max_grad_norm)

                    score_optimizer.step()
                    score_scheduler.step()

                    h['score_lr'] = score_optimizer.param_groups[0]['lr']
                    h['global_mean_cos'] = global_X.mean().item()
                    try:
                        h['score_alpha'] = self.score_model.alpha.item()
                    except BaseException:
                        pass

                else:
                    with torch.no_grad():
                        global_loss = self.score_model.loss(global_X, global_Y)

                h['global_loss'] = global_loss.detach().nanmean().item()

                # Train projector model
                if (transformer_lr or projection_lr) and step <= num_warmup_steps + \
                        num_training_steps:

                    # Turn off score model updating - only want to train
                    # projector here
                    self.score_model.requires_grad_(False)

                    # Select hard training examples
                    with torch.no_grad():
                        batch_j = global_loss.argmax(dim=1).flatten()

                        if w is not None:
                            batch_W = torch.outer(w[batch_i], w[batch_j])
                        else:
                            batch_W = None

                    # Train the model on the selected high-loss pairs
                    V_j = self.projector_model(strings[batch_j.tolist()])

                    # Update global embedding cache
                    V[batch_j, :] = V_j.detach()

                    batch_X = V_i @ V_j.T
                    batch_Y = (groups[batch_i][:, None] ==
                               groups[batch_j][None, :]).float()
                    h['batch_obs'] = len(batch_i) * len(batch_j)

                    batch_loss = self.score_model.loss(
                        batch_X, batch_Y, weights=batch_W)

                    if regularization:
                        # Apply Global Orthogonal Regularization from
                        # https://arxiv.org/abs/1708.06320
                        gor_Y = (groups[batch_i][:, None] !=
                                 groups[batch_i][None, :]).float()
                        gor_n = gor_Y.sum()
                        if gor_n > 1:
                            gor_X = (V_i @ V_i.T) * gor_Y
                            gor_m1 = 0.5 * gor_X.sum() / gor_n
                            gor_m2 = 0.5 * (gor_X**2).sum() / gor_n
                            batch_loss += regularization * \
                                (gor_m1 + torch.clamp(gor_m2 - 1 / self.projector_model.d, min=0))

                    h['batch_nan'] = torch.isnan(
                        batch_loss.detach()).sum().item()

                    embedding_optimizer.zero_grad()
                    batch_loss.nanmean().backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=max_grad_norm)

                    embedding_optimizer.step()
                    embedding_scheduler.step()

                    h['transformer_lr'] = embedding_optimizer.param_groups[1]['lr']
                    h['projection_lr'] = embedding_optimizer.param_groups[-1]['lr']

                    # Save stats
                    h['batch_loss'] = batch_loss.detach().mean().item()
                    h['batch_pos_target'] = batch_Y.detach().mean().item()

                self.history.append(h)
                step += 1

                if (validation_matcher is not None) and not (
                        step % validation_interval):
                    print(
                        f'\nValidation results at step {step} (current epoch {epoch})')
                    validation = len(self.validation_scores)

                    with torch.no_grad():
                        val_scores = self.test(validation_matcher)

                        val_scores['step'] = step - 1
                        val_scores['epoch'] = epoch
                        val_scores['validation'] = validation

                        if val_scores:
                            self.validation_scores.append(val_scores)

                    # Print validation stats
                    if verbose:
                        print(
                            f'\nValidation results at step {step} (current epoch {epoch})')
                        for k, v in val_scores.items():
                            print(f'    {k}: {v:.4f}')

                        print(list(self.score_model.named_parameters()))

                    # Update best saved model
                    if restore_best and best_state:
                        if val_scores[target] >= max(
                                h[target] for h in self.validation_scores):
                            best_state = deepcopy({
                                'state_dict': self.state_dict(),
                                'val_scores': val_scores
                            })

                    if early_stopping and (
                            validation - best_state['val_scores']['validation'] > early_stopping_patience):
                        print(
                            f'Stopping training ({early_stopping_patience} validation checks since best validation score)')
                        break

        if restore_best:
            print(
                f"Restoring to best state (step {best_state['val_scores']['step']}):")
            for k, v in best_state['val_scores'].items():
                print(f'    {k}: {v:.4f}')

            self.to('cpu')
            self.load_state_dict(best_state['state_dict'])
            self.to(self.device)

        return pd.DataFrame(self.history), pd.DataFrame(self.validation_scores)

    def predict(self, input, **kwargs):
        """
        Predict the groupings for the given input.

        Parameters
        ----------
        input : Matcher
            the matcher to make the groupings predictions with
        **kwargs : dict
            Additional keyword arguments to pass the Embeddings.predict() function

        Returns
        -------
        Matcher
            An Matcher object containing the predicted groups
        """
        embeddings = self.embed(input, **kwargs)
        return embeddings.predict(**kwargs)

    def test(self, gold_matcher, embed=None):
        """
        Returns scores of predicted groupings vs the gold_matcher.

        Parameters
        ----------
        gold_matcher : Matcher
            The matcher compare the predictions with
        embed : Embeddings, optional
            The embeddings to use. If None, then the embeddings are generated from the gold_matcher

        Returns
        -------
        dict
            The scores of the predicted embeddings
        """
        if not embed:
            embeddings = self.embed(gold_matcher, verbose=False)
        else:
            embeddings = embed

        predicted = embeddings.predict()

        scores = score_predicted(predicted, gold_matcher)

        return scores


class Embeddings(nn.Module):
    """
    Stores embeddings for a fixed array of strings and provides methods for
    clustering the strings to create Matcher objects according to different
    algorithms.
    """

    def __init__(
            self,
            strings,
            V,
            score_model,
            weighting_function,
            counts,
            device='cpu'):
        """
        Initialize the Embeddings object.

        Parameters
        ----------
        strings : list
            List of strings that we want to be able to score.
        V : int
            The size of the vocabulary.
        score_model : function
            A function that takes in a string and returns a score.
        weighting_function : function
            A function that takes a list of counts and returns a list of weights.
        counts : dict
            Dictionary of counts of each string in the vocabulary.
        device : str
            The device to run the model on. Default: 'cpu'

        """
        super().__init__()

        self.strings = np.array(list(strings))
        self.string_map = {s: i for i, s in enumerate(strings)}
        self.V = V
        self.counts = counts
        self.w = weighting_function(counts)
        self.score_model = score_model
        self.weighting_function = weighting_function
        self.device = device

        self.to(device)

    def __repr__(self):
        """Return a string representation of the Embeddings object"""
        return f'<nama.Embeddings containing {self.V.shape[1]}-d vectors for {len(self)} strings'

    def to(self, device):
        """
        Moves the model to a specified device.

        Parameters
        ----------
        device : str
            The device to run the model on.
        """
        super().to(device)
        self.V = self.V.to(device)
        self.counts = self.counts.to(device)
        self.w = self.w.to(device)
        self.score_model.to(device)
        self.device = device

    def save(self, f):
        """
        Saves the embedding vectors, score model, weighting function, and strings and counts to an archive format.

        Parameters
        ----------
        f : str
            The file to save to.
        """
        with ZipFile(f, 'w') as zip:

            # Write score model
            zip.writestr('score_model.pkl', pickle.dumps(self.score_model))

            # Write score model
            zip.writestr(
                'weighting_function.pkl',
                pickle.dumps(
                    self.weighting_function))

            # Write string info
            strings_df = pd.DataFrame().assign(
                string=self.strings,
                count=self.counts.to('cpu').numpy())
            zip.writestr('strings.csv', strings_df.to_csv(index=False))

            # Write embedding vectors
            byte_io = BytesIO()
            np.save(byte_io, self.V.to('cpu').numpy(), allow_pickle=False)
            zip.writestr('V.npy', byte_io.getvalue())

    def __getitem__(self, arg):
        """
        Slices the embeddings and its matcher, and returns a new embedding with the same properties as the original, but with the slice applied.

        Parameters
        ----------
        arg : slice or Matcher or iter
            Slice of the Embeddings. Can be type `slice`, `Matcher` or `iter`.

        Returns
        -------
        Embedding
            A new `Embedding` object with the same attributes as the original, but with the strings, `V`, counts, `w`, and `string_map` attributes sliced according to the input.
        """
        if isinstance(arg, slice):
            i = arg
        elif isinstance(arg, Matcher):
            return self[arg.strings()]
        elif hasattr(arg, '__iter__'):
            # Return a subset of the embeddings and their weights
            string_map = self.string_map
            i = [string_map[s] for s in arg]

            if i == list(range(len(self))):
                # Just selecting the whole matcher - no need to slice the
                # embedding
                return copy(self)
        else:
            raise ValueError(
                f'Unknown slice input type ({type(input)}). Can only slice Embedding with a slice, matcher, or iterable.')

        new = copy(self)
        new.strings = self.strings[i]
        new.V = self.V[i]
        new.counts = self.counts[i]
        new.w = self.w[i]
        new.string_map = {s: i for i, s in enumerate(new.strings)}

        return new

    def embed(self, matcher):
        """
        Constructs an updated `Embeddings` object with counts from the input Matcher.

        Parameters
        ----------
        matcher : Matcher
            A `Matcher` object.

        Returns
        -------
        Embeddings
            A new `Embeddings` object with updated counts and weights.
        """
        new = self[matcher]
        new.counts = torch.tensor([matcher.counts[s]
                                  for s in new.strings], device=self.device)
        new.w = new.weighting_function(new.counts)

        return new

    def __len__(self):
        """
        Returns:
            int: Number of strings stored in the Embeddings object.
        """
        return len(self.strings)

    def _matcher_to_group_ids(self, matcher):
        """
        Convert a Matcher object to group IDs.

        Args:
            matcher: compiled regex object

        Returns:
            list of str: A list of strings
        """
        group_id_map = {g: i for i, g in enumerate(matcher.groups.keys())}
        group_ids = torch.tensor([group_id_map[matcher[s]]
                                 for s in self.strings]).to(self.device)
        return group_ids

    def _group_ids_to_matcher(self, group_ids):
        """
        Convert group IDs to a Matcher object.

        Args:
            group_ids: A tensor of group ids

        Returns:
            Matcher: A matcher object
        """
        if isinstance(group_ids, torch.Tensor):
            group_ids = group_ids.to('cpu').numpy()

        strings = self.strings
        counts = self.counts.to('cpu').numpy()

        # Sort by group and string count
        g_sort = np.lexsort((counts, group_ids))
        group_ids = group_ids[g_sort]
        strings = strings[g_sort]
        counts = counts[g_sort]

        # Identify group boundaries and split locations
        split_locs = np.nonzero(group_ids[1:] != group_ids[:-1])[0] + 1

        # Get grouped strings as separate arrays
        groups = np.split(strings, split_locs)

        # Build the matcher
        matcher = Matcher()
        matcher.counts = Counter({s: int(c) for s, c in zip(strings, counts)})
        matcher.labels = {s: g[-1] for g in groups for s in g}
        matcher.groups = {g[-1]: list(g) for g in groups}

        return matcher

    @torch.no_grad()
    def _fast_predict(
            self,
            threshold=0.5,
            base_matcher=None,
            progress_bar=True,
            batch_size=64):
        """
        For each embedding, find all embeddings that are similar to it and assign them to the same group.

        Parameters
        ----------
        threshold : float
            The threshold for the jaccard score.
        base_matcher : Matcher, optional
            A matcher object that has already been run on the embeddings. This is used to speed up the matching process.
        progress_bar : bool, optional
            Whether to show a progress bar, defaults to True.
        batch_size : int, optional
            The number of embeddings to process at once, defaults to 64.

        Returns
        -------
        Matcher
        A matcher object.
        """
        if base_matcher is not None:
            # self = self.embed(base_matcher)
            group_ids = self._matcher_to_group_ids(base_matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        for batch_start in tqdm(
                range(
                    0,
                    len(self),
                    batch_size),
                delay=1,
                desc='Predicting matches',
                disable=not progress_bar):

            i_slice = slice(batch_start, batch_start + batch_size)
            j_slice = slice(batch_start + 1, None)

            g_i = group_ids[i_slice]
            g_j = group_ids[j_slice]

            # Find j's with jaccard > threshold ("matches")
            batch_matched = (V[i_slice] @ V[j_slice].T >= cos_threshold) \
                * (g_i[:, None] != g_j[None, :])

            for k, matched in enumerate(batch_matched):
                if matched.any():
                    # Get the group ids of the matched j's
                    matched_groups = g_j[matched]

                    # Identify all embeddings in these groups
                    ids_to_group = torch.isin(group_ids, matched_groups)

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

        :param threshold: The minimum similarity required to unite two strings
        :param group_threshold: the minimum similarity required within a single group
        :param separate_strings: a list of strings that should never be united with each other (these
        strings will still be united with other strings)
        :param base_matcher: This is a matcher object that you can use to initialize the group_ids before
        uniting new matches
        :param batch_size: The number of embeddings to process at once, defaults to 64 (optional)
        :param progress_bar: Whether to show a progress bar, defaults to True (optional)
        :return: A Matcher object
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
        for batch_start in tqdm(range(0, len(self), batch_size),
                                desc='Scoring pairs',
                                delay=1, disable=not progress_bar):

            i_slice = slice(batch_start, batch_start + batch_size)
            j_slice = slice(batch_start + 1, None)

            # Find j's with jaccard > threshold ("matches")
            batch_cos = V[i_slice] @ V[j_slice].T

            # Search upper diagonal entries only
            # (note j_slice starting index is offset by one)
            batch_cos = torch.triu(batch_cos)

            bi, bj = torch.nonzero(batch_cos >= cos_threshold, as_tuple=True)

            if len(bi):
                # Convert batch index locations to global index locations
                i = bi + batch_start
                j = bj + batch_start + 1

                cos = batch_cos[bi, bj]

                # Can skip strings that are already matched in the base matcher
                unmatched = group_ids[i] != group_ids[j]
                i = i[unmatched]
                j = j[unmatched]
                cos = cos[unmatched]

                if len(i):
                    batch_matches = torch.hstack([i[:, None], j[:, None]])

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
            separated = torch.tensor(
                [s in separate_strings for s in self.strings]).to(self.device)

            n_matches = matches.shape[0]
            with tqdm(total=n_matches, desc='Uniting matches',
                      delay=1, disable=not progress_bar) as p_bar:

                while len(matches):

                    # Select the current match pair and remove it from the
                    # queue
                    match_pair = matches[0]
                    matches = matches[1:]

                    # Get the groups of the current match pair
                    g = group_ids[match_pair]

                    # Identify which strings should be united
                    to_unite = (group_ids == g[0]) | (group_ids == g[1])

                    # Flag whether uniting this pair will unite any separated
                    # strings
                    any_separated = separated[to_unite].sum() > 1

                    # Flag whether the new group will have three or more
                    # strings
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
                                V0 = V[group_ids == g[0], :]
                                V1 = V[group_ids == g[1], :]

                                unite_ok = (V0 @ V1.T).min() >= separate_cos

                    if unite_ok:

                        # Unite groups
                        group_ids[to_unite] = g[0]

                        # If we are uniting more than two strings, we can eliminate
                        # some redundant matches in the queue
                        if not singletons:
                            # Removed queued matches that are now in the same
                            # group
                            matches = matches[group_ids[matches[:, 0]]
                                              != group_ids[matches[:, 1]]]
                    else:
                        # Remove queued matches connecting these groups
                        matches = matches[torch.isin(group_ids[matches[:, 0]], g, invert=True)
                                          | torch.isin(group_ids[matches[:, 1]], g, invert=True)]

                    # Update progress bar
                    p_bar.update(n_matches - matches.shape[0])
                    n_matches = matches.shape[0]

        return self._group_ids_to_matcher(group_ids)

    @torch.no_grad()
    def voronoi(
            self,
            seed_strings,
            threshold=0,
            base_matcher=None,
            progress_bar=True,
            batch_size=64):
        """
        Unite embedding strings with each string's most similar seed string.

        - "base_matcher" will be used to inialize the group_ids before uniting new matches
        - "theshold" sets the minimimum match similarity required between a string and seed string
          for the string to be matched. (i.e., setting theshold=0 will result in every embedding
          string to be matched its nearest seed string, while setting threshold=0.9 will leave
          strings that have similarity<0.9 with their nearest seed string unaffected)

        returns: Matcher object

        :param seed_strings: a list of strings to use as seed strings
        :param threshold: The minimum similarity score required for a match, defaults to 0 (optional)
        :param base_matcher: The matcher object to use as a starting point. If None, then the embedding
        strings will be used as the starting point
        :param progress_bar: Whether to show a progress bar, defaults to True (optional)
        :param batch_size: The number of embedding strings to process at a time, defaults to 64
        (optional)
        """

        if base_matcher is not None:
            # self = self.embed(base_matcher)
            group_ids = self._matcher_to_group_ids(base_matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        V = self.V
        cos_threshold = self.score_model.score_to_cos(threshold)

        seed_ids = torch.tensor([self.string_map[s]
                                for s in seed_strings]).to(self.device)
        V_seed = V[seed_ids]
        g_seed = group_ids[seed_ids]
        is_seed = torch.zeros(V.shape[0], dtype=torch.bool).to(self.device)
        is_seed[g_seed] = True

        for batch_start in tqdm(
                range(
                    0,
                    len(self),
                    batch_size),
                delay=1,
                desc='Predicting matches',
                disable=not progress_bar):

            batch_slice = slice(batch_start, batch_start + batch_size)

            batch_cos = V[batch_slice] @ V_seed.T

            max_cos, max_seed = torch.max(batch_cos, dim=1)

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
    def _batch_scored_pairs(self, group_ids, batch_start, batch_size,
                            is_match=None,
                            min_score=None, max_score=None,
                            min_loss=None, max_loss=None):
        """
        A private method used to compute the scored pairs within a batch. Computes the scores and losses for all pairs of strings in the batch,
        and returns the pairs, their scores, and their losses

        :param group_ids: The group ids of the strings in the batch
        :param batch_start: the index of the first string in the batch
        :param batch_size: The number of pairs to score at a time
        :param is_match: True to return only matches, False to return only non-matches, None to return all
        pairs
        :param min_score: minimum score for a pair to be included in the results
        :param max_score: The maximum score for a pair to be included in the results
        :param min_loss: minimum loss for a pair to be included in the results
        :param max_loss: The maximum loss for a pair to be included in the results
        """
        strings = self.strings
        V = self.V
        w = self.w

        # Create simple slice objects to avoid creating copies with advanced
        # indexing
        i_slice = slice(batch_start, batch_start + batch_size)
        j_slice = slice(batch_start + 1, None)

        X = V[i_slice] @ V[j_slice].T
        Y = (group_ids[i_slice, None] == group_ids[None, j_slice]).float()
        if w is not None:
            W = w[i_slice, None] * w[None, j_slice]
        else:
            W = None

        scores = self.score_model(X)
        loss = self.score_model.loss(X, Y, weights=W)

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
        i, j = torch.nonzero(scores, as_tuple=True)

        pairs = np.hstack([
            strings[i.cpu().numpy() + batch_start][:, None],
            strings[j.cpu().numpy() + (batch_start + 1)][:, None]
        ])

        pair_groups = np.hstack([
                                strings[group_ids[i + batch_start].cpu().numpy()][:, None],
                                strings[group_ids[j + (batch_start + 1)].cpu().numpy()][:, None]
                                ])

        pair_scores = scores[i, j].cpu().numpy()
        pair_losses = loss[i, j].cpu().numpy()

        return pairs, pair_groups, pair_scores, pair_losses

    def iter_scored_pairs(
            self,
            matcher=None,
            batch_size=64,
            progress_bar=True,
            **kwargs):
        """
        A private method used to process and yield a batch of scored pairs of strings from the input matcher.

        :param matcher: a matcher objec
        :param batch_size: The number of pairs to score at a time, defaults to 64 (optional)
        :param progress_bar: Whether to show a progress bar, defaults to True (optional)
        :param kwargs: Additional keyword arguments to pass to the _batch_scored_pairs method.
        """

        if matcher is not None:
            self = self.embed(matcher)
            group_ids = self._matcher_to_group_ids(matcher)
        else:
            group_ids = torch.arange(len(self)).to(self.device)

        for batch_start in tqdm(
                range(
                    0,
                    len(self),
                    batch_size),
                desc='Scoring pairs',
                disable=not progress_bar):
            pairs, pair_groups, scores, losses = self._batch_scored_pairs(
                self, group_ids, batch_start, batch_size, **kwargs)
            for (s0, s1), (g0, g1), score, loss in zip(
                    pairs, pair_groups, scores, losses):
                yield {
                    'string0': s0,
                    'string1': s1,
                    'group0': g0,
                    'group1': g1,
                    'score': score,
                    'loss': loss,
                }
