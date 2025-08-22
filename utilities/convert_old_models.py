import os
from pathlib import Path
import torch


import nama
from nama import load_similarity_model, SimilarityModel
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.scoring_model import ExpSimilarityScore
from nama.embedding_similarity.embedding_model import EmbeddingModel

data_dir = Path(os.environ['NAMA_DATA'])


train_kwargs = {
                'max_epochs': 1,
                'warmup_frac': 0.1,
                'transformer_lr':5e-6,
                'score_lr':30,
                'alpha':20,
                'use_counts':False,
                'batch_size':6,
                'add_upper':True,
                }

model_defs = {
    'nama-64':{'d':64,'model_name':'roberta-base'},
    'nama-128':{'d':128,'model_name':'roberta-base'},
    'nama-256':{'d':256,'model_name':'roberta-base'},
    'nama-768':{'d':None,'model_name':'roberta-base'},
    # 'nama_large':{'d':256,'model_name':'roberta-large'},
    }



os.makedirs(data_dir/'models'/'v0.1'/'reformatted',exist_ok=True)
for model_name,hparams in model_defs.items():

    # try:

        kwargs = dict(**train_kwargs,**hparams)
        kwargs['score_class'] = ExpSimilarityScore
        kwargs['embedding_class'] = EmbeddingModel

        checkpoint = torch.load(data_dir/'models'/'v0.1'/'original'/f'{model_name}.bin', map_location='cpu')

        metadata = checkpoint['metadata']

        kwargs.update(metadata)

        sim = SimilarityModel(**kwargs)
        sim.load_state_dict(checkpoint['state_dict'],strict=False)

        sim.save(data_dir/'models'/'v0.1'/'reformatted'/f'{model_name}.bin')

        print(f'Successfully converted {model_name}')

    # except Exception as e:
    #     print(f'Failed to convert {model_name} with error: {e}')
    #     continue


    # sim.embedding_model.d