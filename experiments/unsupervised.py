import numpy as np
import pandas as pd
from tqdm import tqdm

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted
from nama.embedding_similarity import EmbeddingSimilarity
from nama.strings import simplify_corp


matchers = {}
for dataset in ['canlobby','opensecrets']:
    matchers[dataset+'_gold'] = nama.read_csv(nama.root_dir/'training'/'data'/f'{dataset}_client_names.csv')
    matchers[dataset+'_hash'] = nama.Matcher(matchers[dataset+'_gold'].strings()).unite(simplify_corp)

test_train_pairs = [
    # ('canlobby_gold','opensecrets_gold'),
    # ('canlobby_gold','canlobby_hash'),
    # ('opensecrets_gold','canlobby_gold'),
    ('opensecrets_gold','opensecrets_hash'),
    ]

results = []
for test_set,training_set in test_train_pairs:
    for seed in range(1,11):
        print(f'\n Testing {test_set} trained on {training_set}, seed={seed}')

        test = matchers[test_set]
        train = matchers[training_set]

        sim = EmbeddingSimilarity(prompt='Organization: ',d=None)
        sim.to('cuda:0')

        history_df = sim.train(train,
                                    use_counts=False,
                                    augment=False,
                                    transformer_lr=1e-4,
                                    score_lr=10,
                                    max_epochs=1,
                                    batch_size=64,
                                    verbose=False,
                                    max_grad_norm=1
                               )

        for threshold in tqdm(np.linspace(0,1,21),desc='scoring'):
            pred = sim.predict(test,threshold=threshold,progress_bar=False)

            for unite_hash in [False,True]:
                if unite_hash:
                    pred = pred.unite(simplify_corp)

                scores = score_predicted(pred,test)

                scores['test_set'] = test_set
                scores['training_set'] = training_set
                scores['seed'] = seed
                scores['threshold'] = threshold
                scores['unite_hash'] = unite_hash

                results.append(scores)

        sim.to('cpu')

results_df = pd.DataFrame(results)

results_df.to_csv(nama.root_dir/'experiments'/'unsupervised_results.csv')

results_df = results_df[(results_df['test_set'] != 'opensecrets_gold') | (results_df['training_set'] != 'canlobby_hash')]

import matplotlib.pyplot as plt

for test_set in ['canlobby_gold','opensecrets_gold']:
    print(test_set)
    mean_results_df = results_df[results_df['test_set'] == test_set] \
                            .groupby(['training_set','unite_hash','threshold']) \
                            .mean() \
                            .reset_index()
    ax = plt.subplot()
    for (training_set,unite_hash), df in mean_results_df.groupby(['training_set','unite_hash']):
        df.plot('recall','precision',ax=ax,label=f'train={training_set}, unite_hash={unite_hash}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07))
    plt.show()

    ax = plt.subplot()
    for (training_set,unite_hash), df in mean_results_df.groupby(['training_set','unite_hash']):
        df.plot('threshold','F1',ax=ax,label=f'train={training_set}, unite_hash={unite_hash}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07))
    plt.show()


results_df[['training_set','test_set']].drop_duplicates()
