import numpy as np
import pandas as pd
from tqdm import tqdm

import nama
from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, confusion_df, split_on_groups, kfold_on_groups


gold = nama.read_csv(data_dir/'training_data'/'canlobby_client_names.csv')

groups = sorted(gold.groups.keys())

all_train,test = split_on_groups(gold,0.8,seed=1)





token_model = TokenSimilarity()

scores_df = token_model.learn_threshold(all_train,use_counts=False,grid=np.linspace(0.75,1,25))

scores_df.plot(x='threshold',y='F1')
scores_df.plot(x='threshold',y='accuracy')

scores_df['F1'].max()


token_model.fit(test)
pred = token_model.predict()

score_predicted(pred,gold)

#
#
#
# # Then we can use the similarity model to predict matches between the matcher
# # strings. The predict method returns a new matcher.
#
# for fold,(train,validate) in enumerate(kfold_on_groups(all_train,k=4,seed=1)):
#
#     scores = []
#     for t in tqdm(np.linspace(1,0.8,21)):
#         pred = token_model.predict(validate.strings(),threshold=t)
#         s = score_predicted(pred,validate)
#         s['threshold'] = t
#         scores.append(s)
#
#     scores_df = pd.DataFrame(scores)
#
#
#
# predicted = token_model.predict(manual.strings(),threshold=0.95)
#
