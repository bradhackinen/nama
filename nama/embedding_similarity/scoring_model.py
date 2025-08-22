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




# class ExpSimilarityScore(torch.nn.Module):
#     def __init__(self,scale=50,shape=1,learn_shape=False,**kwargs):

#         '''
#         A trainable similarity scoring model that estimates the probability
#         of a match as the negative exponent of of the distance between
#         embeddings, where the distance is defined as:
            
#             d(v_i,v_j) = (1 - cos(v_i,v_j))^shape
        
#             p(match|v_i,v_j) = exp(-scale*d(v_i,v_j))
        
#         By default, `scale` is a trainable parameter while `shape` is fixed at initialization.

#         Parameters:
#         scale (float): Scale parameter for the similarity score
#         shape (float): Shape parameter for the similarity score
#         '''

#         super().__init__()

#         self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

#         if learn_shape:
#             self.shape = torch.nn.Parameter(torch.tensor(float(shape)))
#         else:
#             self.shape = torch.tensor(float(shape))

#     def __repr__(self):
#         return f'<nama.ExpSimilarityScore with scale={self.scale.item()} and shape={self.shape.item()}>'
    
#     def forward(self,X,epsilon=1e-8):
#         '''
#         Compute the match score given the cosine similarity between embeddings (X)

#         Parameters:
#         X (torch.Tensor): Cosine similarity between embeddings
#         epsilon (float): Small value to prevent overflow/undefined results
#         '''
#         # Transform cosine similarity X to distance Z > 0
#         Z = self.scale*torch.pow(torch.clamp(1-X, min=epsilon),self.shape)
  
#         score = self.shape*torch.exp(-Z)

#         score = torch.clamp(score,min=0,max=1.0)

#         return score

#     def loss(self,X,Y,weights=None,epsilon=1e-8):

#         # Transform cosine similarity X to distance Z > 0
#         Z = self.scale*torch.pow(torch.clamp(1-X, min=epsilon),self.shape)

#         # Put epsilon floor to prevent overflow/undefined results
#         Z = torch.clamp(Z,min=epsilon)

#         # Cross entropy loss with a simplified formula
#         # Equivalent to: loss = Y*Z - (1-Y)*log(1-exp(-Z))
#         loss = Y*Z - torch.xlogy(1-Y,-torch.expm1(-Z))

#         if weights is not None:
#             loss = weights*loss

#         return loss


#     def score_to_cos(self,score):
#         '''
#         Convert a match probability to a cosine similarity

#         Parameters:
#         score (float): Match probability
#         '''
        
#         if score > 0:
#             return 1 - (-np.log(score)/self.scale.item())**(1/self.shape.item())
#         else:
#             return -99

#     def config_optimizer(self,lr=10):
#         '''
#         Configure the optimizer for the model
#         '''
#         optimizer = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=0)

#         return optimizer
    
#     def get_params(self):
#         '''
#         Get the model parameters
#         '''
#         return {'score_scale':self.scale.item(),'score_shape':self.shape.item()}
    


# class LogisticSimilarityScore(torch.nn.Module):
#     """
#     A trainable similarity scoring model that estimates the probability
#     of a match as the logistic sigmoid of the cosine distance between
#     embeddings:
#         p(match|v_i,v_j) = logistic((cos(v_i,v_j) - loc)/scale)

#     """

#     def __init__(self,loc=0.9,scale=0.3,**kwargs):

#         super().__init__()

#         self.loc = torch.nn.Parameter(torch.tensor(float(loc)))
#         self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

#     def __repr__(self):
#         return f'<nama.LogisticSimilarityScore with {self.loc.item()=} and {self.scale.item()=}>'

#     def forward(self,X):
#         Z = (X - self.loc)/self.scale
#         return torch.sigmoid(Z)

#     def loss(self,X,Y,weights=None):
#         Z = (X - self.loc)/self.scale
#         loss = F.binary_cross_entropy_with_logits(Z, Y,reduction='none')

#         if weights is not None:
#             loss = weights*loss

#         return loss

#     def score_to_cos(self,score):
#         score = torch.tensor(score)
#         return self.loc.item() + self.scale.item()*torch.logit(score)

#     def config_optimizer(self,lr=10):
#         optimizer = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=0)

#         return optimizer
    
#     def get_params(self):
#         return {'score_loc':self.loc.item(),'score_scale':self.scale.item()}
    



# class LogisticLogitSimilarityScore(torch.nn.Module):
#     """
#     A trainable similarity scoring model that estimates the probability
#     of a match as the logistic sigmoid of the cosine distance between
#     embeddings:
#         p(match|v_i,v_j) = logistic(logit((1 + cos(v_i,v_j))/2) - loc)/exp(scale))

#     """

#     def __init__(self,loc=0,scale=0,**kwargs):

#         super().__init__()

#         self.loc = torch.nn.Parameter(torch.tensor(float(loc)))
#         self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

#     def __repr__(self):
#         return f'<nama.LogisticLogitSimilarityScore with {self.loc.item()=} and {self.scale.item()=}>'

#     def _cos_to_logit(self,cos,epsilon=1e-6,clamp=50):
#         """
#         Convert cosine similarity to logit scale
#         """
#         cos = torch.clamp(cos,min=-1+epsilon,max=1-epsilon)
#         log_odds = torch.log1p(cos) - torch.log1p(-cos)
#         logits = (log_odds - self.loc) * torch.exp(-self.scale)

#         if clamp:
#             logits = torch.clamp(logits,min=-clamp,max=clamp)

#         return logits

#     def forward(self,X,epsilon=1e-6):
#         # Z = (torch.logit((1 + X)/2,eps=epsilon) - self.loc)*torch.exp(-self.scale)
#         logits = self._cos_to_logit(X,epsilon=epsilon)

#         return torch.sigmoid(logits)

#     def loss(self,X,Y,weights=None,epsilon=1e-6):

#         logits = self._cos_to_logit(X,epsilon=epsilon)

#         loss = F.binary_cross_entropy_with_logits(logits, Y,reduction='none')

#         if weights is not None:
#             loss = weights*loss

#         return loss

#     def score_to_cos(self,score):
#         score = torch.tensor(score)
#         return 2*torch.sigmoid(self.loc + torch.exp(self.scale)*torch.logit(score)) - 1

#     def config_optimizer(self,lr=10):
#         optimizer = torch.optim.AdamW(self.parameters(),lr=lr,weight_decay=0)

#         return optimizer
    
#     def get_params(self):
#         return {'score_loc':self.loc.item(),'score_scale':self.scale.item()}


# class SkewLogisticSimilarityScore(torch.nn.Module):
#     """
#     A trainable similarity scoring model that estimates the probability
#     of a match as the logistic sigmoid of the cosine distance between
#     embeddings:
#         p(match|v_i,v_j) = 1/(1 + exp(-b*(z - alpha)))^g
#         z = -log(1 - cos(v_i,v_j))
#         b = exp(beta)
#         g = exp(gamma)
#     """

#     def __init__(self,alpha=0,beta=0,gamma=0,**kwargs):

#         super().__init__()

#         self.alpha = torch.nn.Parameter(torch.tensor(float(alpha)))
#         self.beta = torch.nn.Parameter(torch.tensor(float(beta)))
#         self.gamma = torch.nn.Parameter(torch.tensor(float(gamma)))

#     def __repr__(self):
#         return f'<nama.SkewLogSimilarityScore with alpha={self.alpha.item()}, beta={self.beta.item()}, gamma={self.gamma.item()}>'
    
#     def forward(self,X,epsilon=1e-6):
#         X = torch.clamp(X,min=-1+epsilon,max=1-epsilon)

#         Z = -torch.log1p(-X)

#         b = torch.exp(-self.beta)
#         g = torch.exp(self.gamma)

#         score = (1 + torch.exp(-b*(Z - self.alpha)))**-g

#         score = torch.clamp(score,min=epsilon,max=1-epsilon)

#         return score
    
#     def loss(self,X,Y,weights=None,epsilon=1e-6):

#         score = self.forward(X,epsilon=epsilon)

#         # print(score.min(),score.max())

#         loss = F.binary_cross_entropy(score, Y,reduction='none')
        
#         if weights is not None:
#             loss *= weights*loss
        
#         return loss
    
#     def score_to_cos(self,score):

#         # Do this computation in numpy to avoid fp16 precision issues
#         score = float(score)

#         b = np.exp(-self.beta.item())
#         g = np.exp(self.gamma.item())

#         cos = 1 - (np.exp(-b*self.alpha.item())*(score**(-1/g) - 1))**(1/b)

#         cos = np.clip(cos,-1,1)

#         return float(cos)
    
    
#     def config_optimizer(self, lr=1e-3):
#         """
#         Configure the optimizer with separate learning rates for each parameter
        
#         Parameters:
#         lr (float/list like): Base learning rate for all parameters
#         """
#         if isinstance(lr, (list, tuple, np.ndarray, torch.Tensor)):
#             lr = [float(x) for x in lr]
#             alpha_lr, beta_lr, gamma_lr = lr
#         else:
#             alpha_lr = lr
#             beta_lr = lr
#             gamma_lr = lr

#         param_groups = [
#             {'params': [self.alpha], 'lr': alpha_lr},
#             {'params': [self.beta], 'lr': beta_lr},
#             {'params': [self.gamma], 'lr': gamma_lr}
#         ]
        
#         optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
#         return optimizer
    
#     def get_params(self):
#         return {'score_alpha':self.alpha.item(),'score_beta':self.beta.item(),'score_gamma':self.gamma.item()}


