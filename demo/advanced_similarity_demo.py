import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter

import opensecrets
import rulemaking
import nama
from nama.embedding_similarity import Embeddings, load_embeddings,load_similarity_model

from interestgroups.config import data_dir, project_dir
from interestgroups.org_linking.utilities import clean_name


# First, we'll construct a tricky (but realistic) case using Disney subsidaries
test_matcher = nama.Matcher(
                Counter({
                    'Walt Disney World Resort':13,
                    'Walt Disney World':12,
                    'Disney World':11,
                    'DISNEY WORLD':10,
                    'Disney Worldwide Services':9,
                    'Disney Worldwide Services, Inc.':8,
                    'DISNEY WORLDWIDE SERVICES':7,
                    'DISNEY TV STUDIOS':6,
                    'DISNEY TELEVISON GROUP':5,
                    'Disney TV':4,
                    'DISNEY INTERACTIVE, INC.':3,
                    'Disney Interactive':2,
                    'Disney Interactive Media Group':1,
                    'Pixar Animation Studios':1
                    })
                )

test_matcher.to_df()

# Embed the strings using a pre-trained similarity model
sim = load_similarity_model(nama.root_dir/'models'/'nama_base.bin')
sim.to('cuda:1')

embeddings = sim.embed(test_matcher)

# Now we can do a simple prediction
embeddings.predict().to_df().sort_values('count',ascending=False)

# If we adjust the threshold, we get some rough matching that is OK
embeddings.predict(threshold=0.75).to_df().sort_values('count',ascending=False)

# We can get better matches if we use group_threshold to control the
# minimimum similarity of pairs allowed within each group. This prevents strings
# from being united if doing so will also unite two strings that are unlikely
# to be a match.
embeddings.predict(threshold=0.5,group_threshold=0.5).to_df()

# This is probably the best we can do without any additional information.

# However, if we have additional information (either manually collected, or from
# a database like wikidata) we might know that "Disney World" is actually a
# common way to refer to "Walt Disney World Resort", not "Disney Worldwide
# Services", which is a separate organization used primarily for lobbying.

# We can encode this information in a small matcher:
manual_matcher = nama.Matcher([
                            'Walt Disney World Resort',
                            'Disney World',
                            'Disney Worldwide Services'
                            ]) \
                        .unite([
                            'Walt Disney World Resort',
                            'Disney World',
                            ])

base_matcher = test_matcher \
                .unite(manual_matcher) \
                .unite(lambda s: s.lower())


base_matcher.to_df().sort_values('count',ascending=False)


voronoi = embeddings.voronoi(
                        base_matcher=base_matcher,
                        seed_strings=manual_matcher.strings(),
                        threshold=0.65)


voronoi.to_df().sort_values('count',ascending=False)

# Coarse clustering that respects manual matches and separations
pred = embeddings.predict(
                    threshold=0,
                    group_threshold=0.0,
                    base_matcher=voronoi,
                    separate_strings=manual_matcher.groups.keys(),
                    )

pred.to_df().sort_values('count',ascending=False)

#
#
# # Coarse clustering that respects manual matches and separations
# pred = embeddings.predict(
#         threshold=0.5,
#         group_threshold=0.5,
#         base_matcher=base_matcher,
#         target_strings=manual_matcher.strings(),
#         separate_strings=manual_matcher.groups.keys()
#         )
#
# pred.to_df().sort_values('count',ascending=False)
#
#
#
#
# # Coarse clustering that respects manual matches and separations
# pred = embeddings.predict(
#         threshold=0.5,
#         group_threshold=0.9,
#         target_strings=manual_matcher.strings()
#         )
#
# pred.to_df().sort_values('count',ascending=False)



# First, we'll construct a tricky (but realistic) case using Disney subsidaries
test_matcher = nama.Matcher(Counter({s.upper():c for s,c in test_matcher.counts.items()}))

# Suppose we have the following partial match information:
manual_matches_df = pd.DataFrame([
                        ('WALT DISNEY WORLD RESORT','WALT DISNEY WORLD RESORT'),
                        ('DISNEY WORLD','WALT DISNEY WORLD RESORT'),
                        ('DISNEY WORLDWIDE SERVICES','DISNEY WORLDWIDE SERVICES'),
                        ],columns=['string','group'])

manual_matcher = nama.from_df(manual_matches_df)

base_matcher = test_matcher \
                .unite(manual_matcher)

embeddings = sim.embed(test_matcher)



voronoi = embeddings.voronoi(
                        base_matcher=base_matcher,
                        seed_strings=manual_matcher.strings(),
                        threshold=0.65)


voronoi.to_df().sort_values('count',ascending=False)

# Coarse clustering that respects manual matches and separations
pred = embeddings.predict(
                    threshold=0.65,
                    group_threshold=0.65,
                    base_matcher=voronoi,
                    separate_strings=manual_matcher.groups.keys(),
                    )

pred.to_df().sort_values('count',ascending=False)

#
