import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

import optuna

import nama
from nama.scoring import score_predicted, split_on_groups
from nama import SimilarityModel, load_similarity_model


# 'roberta-base'
# 'bert-base-uncased'
# 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
# 'sentence-transformers/all-mpnet-base-v2'


def main(study_name,model_name,device,n_trials):

    data_dir = Path(os.environ['NAMA_DATA'])
    study_db = f'sqlite:///{data_dir}/hparam_search.db'

    gold_matches = nama.read_csv(Path(data_dir)/'training_data'/'opensecrets_train.csv')

    # gold_matches,_ = split_on_groups(gold_matches,0.01)

    training_matches,validation_matches = split_on_groups(gold_matches,0.8,seed=1)


    def objective(trial):

        hparams = {
                    'model_name':model_name,
                    'transformer_lr':trial.suggest_float('transformer_lr',1e-6,1e-4,log=True),
                    'score_lr':trial.suggest_float('score_lr',0.1,100,log=True),
                    'alpha':trial.suggest_float('alpha',0.1,100,log=True),
                    'batch_size':trial.suggest_int('batch_size',4,12),
                    'max_epochs': 1,
                    'warmup_frac': 0.1,
                    'use_counts':False,
                    'add_upper': 'uncased' not in model_name,
                    'prompt':'Organization'
        }

        for k in hparams:
            trial.set_user_attr(k,hparams[k])    
        
        sim = SimilarityModel(**hparams)
        sim.to(device)

        history_df = sim.train(training_matches,verbose=True,**hparams)

        pred_embeddings = sim.embed(validation_matches)
        pred_matches = pred_embeddings.unite_similar(threshold=0.5)

        scores = score_predicted(pred_matches,validation_matches,use_counts=hparams['use_counts'])

        sim.to('cpu')

        return scores['F1']


    study = optuna.create_study(direction="maximize",storage=study_db,load_if_exists=True,study_name=study_name)
    study.optimize(objective, n_trials=n_trials)




parser = ArgumentParser()

# parser.add_argument('--study_name',type=str,default='roberta-base-training')
parser.add_argument('--model_name',type=str,default='roberta-base')
parser.add_argument('--n_trials',type=int,default=25)
parser.add_argument('--device',type=str,default='cpu')


args = parser.parse_args()

main(args.model_name,args.model_name,args.device,args.n_trials)
