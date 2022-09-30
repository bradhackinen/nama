import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

import nama
# from nama.token_similarity import TokenSimilarity
from nama.scoring import score_predicted, split_on_groups, kfold_on_groups
from nama.embedding_similarity import EmbeddingSimilarity, load

canlobby = nama.read_csv(data_dir/'training_data'/'canlobby_client_names.csv')

sim = load(data_dir/'models'/'nama_base.bin')

embeddings = sim.embed(canlobby)

pred = sim.pred(canlobby,threshold=0.5)

# Embeddings method syntax
predicted = embeddings.predict(canlobby,threshold=0.5)

matcher = embeddings.separate(predicted,['a','b','c'])

for pair in []:
    # embeddings.separate(matcher,pair,inplace=True)
    matcher = embeddings.separate(matcher,pair)


# Mutate syntax
predicted = embeddings.predict(threshold=0.5)

matcher = embeddings.mutate(matcher,separate=['a','b','c'])

for pair in []:
    matcher = embeddings.mutate(matcher,separate=pair)


# Matcher syntax
matcher = nama.Matcher(canlobby.strings)

similarity_model = EmbeddingSimilarityModel()

similarity_model.train(training_matcher)

pred = similarity_model.predict(test_matcher)


pred = matcher.unite(embeddings,threshold=0.5)

# pred = canlobby.split_all().unite(embeddings,threshold=0.5)

matcher = pred.separate(['a','b','c'],similarity_model=embeddings)

for pair in []:
    matcher = matcher.separate(pair,similarity_model=embeddings)



# Fit syntax

matcher = nama.Matcher(canlobby.strings)

similarity_model = EmbeddingSimilarityModel()

similarity_model.train(training_matcher)

fitted = similarity_model.fit(test_matcher)

pred = fitted.predict()

pred_fitted = fitted.fit(pred)

top_df = pred_fitted.top_scored_pairs()

# ------------------------------------------------------------------------------
# Final syntax:



# Matcher syntax
matcher = nama.Matcher(canlobby.strings)

similarity_model = EmbeddingSimilarityModel()

similarity_model.train(training_matcher)

embeddings = similarity_model.embed(test_matcher)

pred = embeddings.predict()

# Or,
pred = similarity_model.predict(test_matcher)

# pred = canlobby.split_all().unite(embeddings,threshold=0.5)

matcher = pred.separate(['a','b','c'],similarity_model=similarity_model)

for pair in []:
    matcher = matcher.separate(pair,similarity_model=similarity_model)
