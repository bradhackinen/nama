import torch
from transformers import AutoTokenizer,AutoModel

class EmbeddingModel(torch.nn.Module):
    """
    A basic wrapper around a Hugging Face transformer model.
    Takes a string as input and produces an embedding vector of size d.
    """
    def __init__(self,
                    model_class=AutoModel,
                    model_name='roberta-base',
                    pooling='cls',
                    normalize=True,
                    d=None,
                    prompt='',
                    device='cpu',
                    add_upper=True,
                    **kwargs):

        super().__init__()

        self.model_class = model_class
        self.model_name = model_name
        self.pooling = pooling
        self.normalize = normalize
        self.d = d
        self.prompt = prompt
        self.add_upper = add_upper

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
            strings = [s + self.tokenizer.sep_token + s.upper() for s in strings]

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

        elif self.pooling == 'cls':
            h = batch_out['last_hidden_state']
            v = h[:,0]

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
