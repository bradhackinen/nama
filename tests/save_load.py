import os
from pathlib import Path
import torch


import nama
from nama import load_similarity_model
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity.scoring_model import LogLogisticSimilarityScore

data_dir = Path(os.environ['NAMA_DATA'])


sim = None

test_kwargs = {
    'prompt':'Test prompt',
    'd':256,
    'score_class':LogLogisticSimilarityScore,
    'loc':2,
    'scale':-5,
}

sim = nama.SimilarityModel(**test_kwargs)

os.makedirs(data_dir/'save_load_test',exist_ok=True)

sim.save(data_dir/'save_load_test'/'test_model.bin')


checkpoint = torch.load(data_dir/'save_load_test'/'test_model.bin', map_location='cpu')

checkpoint['config']


sim = load_similarity_model(data_dir/'save_load_test'/'test_model.bin',strict=False)

sim.config


sim.score_model


# Test loadin an old v0.1 model
sim = load_similarity_model(data_dir/'models'/'nama-256.bin',strict=False)
sim.to('cuda')

gold = nama.read_csv(data_dir/'training_data'/'processed'/'gpt_reviewed_matches.csv')

# down sample for (in sample) test
test,_ = split_on_groups(gold,0.1,seed=1)


pred = sim.unite_similar(test)

scores = score_predicted(pred, test, use_counts=True)

assert scores['F1'] > 0.9

sim.embedding_model.prompt




checkpoint = torch.load(data_dir/'models'/'nama-256.bin', map_location='cpu')

checkpoint['metadata']
