import torch
import numpy as np

class SimilarityScore(torch.nn.Module):
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
