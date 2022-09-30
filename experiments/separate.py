from pathlib import Path

import nama
from nama.scoring import score_predicted, split_on_groups
from nama.embedding_similarity import load_similarity_model,load_embeddings

gold = nama.read_csv(nama.root_dir/'training'/'data'/'canlobby_train.csv')


sim = load_similarity_model(Path(nama.root_dir)/'models'/'nama_base.bin')
sim.to('cuda:0')

test,_ = split_on_groups(gold,0.1,seed=1)

test_embeddings = sim.embed(test)

test_embeddings.save(nama.root_dir/'_review'/'temp_embeddings.bin')

test_embeddings = load_embeddings(nama.root_dir/'_review'/'temp_embeddings.bin')

test_embeddings.predict(threshold=0.5)

test_embeddings.predict_separated(unite_threshold=0.5)
test_embeddings.predict_separated(base_matcher=test,unite_threshold=0.5)

test_embeddings.predict_separated(base_matcher=test,unite_threshold=0.05)



pred = test_embeddings.predict(threshold=0.2)

pred.to_df().sample(50)

pred.keep().to_df()

group = pred.matches('Nordstar Capital LP')

embeddings = test_embeddings[group]

pred = embeddings.predict(threshold=0.2)

embeddings.predict_separated(['Nordstar Capital LP','Northstar Earth & Space Inc.'],threshold=0.2).to_df()

pred.separate(['Nordstar Capital LP','Northstar Earth & Space Inc.'],embeddings)


from collections import defaultdict

def separate(self,separated_strings,similarity_model,inplace=False,**kwargs):
    if not inplace:
        self = self.copy()

    # Identify which groups contain the separated strings
    group_map = defaultdict(list)
    for s in set(separated_strings):
        group_map[self[s]].append(s)

    for g,g_separated_strings in group_map.items():

        # If group contains strings to separate...
        if len(g_separated_strings) > 1:
            group_strings = self.groups[g]

            # Split the group strings
            self.split(group_strings,inplace=True)

            # Re-unite with new prediction that enforces separation
            embeddings = similarity_model[group_strings]
            predicted = embeddings.predict_separated(g_separated_strings,threshold=0)
            self.unite(predicted,inplace=True)

    return self

self = pred
separated_strings = ['Nordstar Capital LP','Northstar Earth & Space Inc.']
similarity_model = embeddings

sep = separate(pred,['Nordstar Capital LP','Northstar Earth & Space Inc.'],test_embeddings)

sep.matches('Nordstar Capital LP')
sep.matches('Northstar Earth & Space Inc.')

pred.matches('Nordstar Capital LP')

df = sep.to_df()

df[df]
