import torch
import numpy as np
import torch.nn.functional as F
from scipy.special import logit


class CosineSimilarityScore(torch.nn.Module):
    '''
    A placeholder similarity scoring model that passes through the cosine similarity
    '''
    def __init__(self,**kwargs):
        super().__init__()

    def __repr__(self):
        return '<nama.CosineSimilarityScore>'

    def forward(self,X):
        return X
    
    def score_to_cos(self,score):
        return score
    
    def get_params(self):
        return {}



class ExpSimilarityScore(torch.nn.Module):
    """
    A trainable similarity scoring model that estimates the probability
    of a match as the negative exponent of 1+cosine distance between
    embeddings:
        p(match|v_i,v_j) = exp(-alpha*(1-v_i@v_j))
    """
    def __init__(self,alpha=50,**kwargs):

        super().__init__()

        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha)))

    def __repr__(self):
        return f'<nama.SimilarityScore with {self.alpha=}>'

    def forward(self,X):
        # Z is a scaled distance measure: Z=0 means that the score should be 1
        Z = self.alpha*(1 - X)
        return torch.clamp(torch.exp(-Z),min=0,max=1.0)

    def loss(self,X,Y,weights=None,epsilon=1e-6):

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

        return loss

    def score_to_cos(self,score):
        if score > 0:
            return 1 + np.log(score)/self.alpha.item()
        else:
            return -99

    def config_optimizer(self,lr=10):
        optimizer = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=0)

        return optimizer


class LogLogisticSimilarityScore(torch.nn.Module):
    """
    A trainable similarity scoring model that estimates the probability
    of a match as the sigmoid of the log-transformed cosine distance between
    embeddings:
        p(match|v_i,v_j) = logistic((z - loc)/exp(scale))

        where z = -log(1 - cos(v_i,v_j))
    """

    def __init__(self,loc=0,scale=0,**kwargs):

        super().__init__()

        self.loc = torch.nn.Parameter(torch.tensor(float(loc)))
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def __repr__(self):
        return f'<nama.LogLogisticSimilarityScore with loc={self.loc.item()} and scale={self.scale.item()}>'
    
    def forward(self,X,epsilon=1e-6):
        X = torch.clamp(X,min=-1+epsilon,max=1-epsilon)

        Z = -torch.log1p(-X)

        logits = (Z - self.loc)/torch.exp(self.scale)

        return torch.sigmoid(logits)
    
    def loss(self,X,Y,weights=None,epsilon=1e-6,clamp_logits=100):
        X = torch.clamp(X,min=-1+epsilon,max=1-epsilon)

        Z = -torch.log1p(-X)

        logits = (Z - self.loc)/torch.exp(self.scale)

        if clamp_logits:
            logits = torch.clamp(logits,min=-clamp_logits,max=clamp_logits)

        loss = F.binary_cross_entropy_with_logits(logits, Y,weight=weights,reduction='none')

        return loss
    
    def score_to_cos(self,score):
        # Do this computation in numpy to avoid fp16 precision issues
        score = float(score)

        Z = self.loc.item() + np.exp(self.scale.item())*logit(score)

        cos = 1 - np.exp(-Z)

        cos = np.clip(cos,-1,1)

        return float(cos)
    
    def config_optimizer(self,lr=1e-2):

        if isinstance(lr, (list, tuple, np.ndarray, torch.Tensor)):
            lr = [float(x) for x in lr]
            loc_lr, scale_lr = lr
        else:
            loc_lr = lr
            scale_lr = lr

        param_groups = [
            {'params': [self.loc], 'lr': loc_lr},
            {'params': [self.scale], 'lr': scale_lr}
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
        return optimizer
    
    def get_params(self):
        return {'score_loc':self.loc.item(),'score_scale':self.scale.item()}




