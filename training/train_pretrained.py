from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity import EmbeddingSimilarityModel, load_similarity_model

data_dir = Path('/home/brad/Dropbox/Data/nama')

gold = nama.read_csv(Path(data_dir)/'training_data'/'combined_train_no_compound.csv')
# gold_upper = nama.read_csv(data_dir/'training_data'/'combined_train_upper_case.csv')


train_kwargs = {
                'max_epochs': 1,
                'warmup_frac': 0.1,
                'transformer_lr':5e-6,
                'score_lr':30,
                'alpha':20,
                'use_counts':True,
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


for model_name,hparams in model_defs.items():

    print(f'Training {model_name}')

    sim = EmbeddingSimilarityModel(prompt='Organization: ',**hparams)
    sim.to('cuda:3')

    history_df = sim.train(gold,verbose=True,**train_kwargs)

    sim.to('cpu')

    save_file = Path(data_dir)/'models'/f'{model_name}.bin'
    print(f'Saving model as {save_file}')
    sim.save(save_file)

    history_df.assign(log_loss=np.log(history_df['global_loss'])) \
                ['log_loss'] \
                .rolling(1000).mean() \
                .plot()
    plt.show()


# Small in-sample test to make sure models trained and saved correctly
test,_ = split_on_groups(gold,0.1,seed=1)

results = []
for model_name in model_defs.keys():
    print(f'Verifying {model_name}')

    sim = load_similarity_model(Path(data_dir)/'models'/f'{model_name}.bin')
    sim.to('cuda:2')

    for half in False,True:

        if half:
            sim.half()

        test_embeddings = sim.embed(test)

        for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):

            pred = test_embeddings.predict(threshold=threshold,progress_bar=False)

            scores = score_predicted(pred,test,use_counts=train_kwargs['use_counts'])

            scores.update(train_kwargs)

            scores['model'] = model_name
            scores['half'] = half
            scores['threshold'] = threshold

            results.append(scores)

results_df = pd.DataFrame(results)

run_cols = ['model','half']

ax = plt.subplot()
for run_vals, df in results_df.groupby(run_cols):
    df.plot('threshold','F1',ax=ax,label=f'{run_vals=}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=1)
plt.show()


results_df.groupby(run_cols)['F1'].quantile(0.8)
