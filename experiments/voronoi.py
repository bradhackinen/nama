import pandas as pd

import nama
# from nama import project_dir
from nama.embedding_similarity import load_similarity_model

strings = [
     '!ndigo',
     'INDIGO BOOKS LLC',
     'INDIGO HOLDINGS',
     'INDIGO NV',
     'INDIGO PRODUCTS COMPANY',
     'IndiGO',
     'Indiana County Transit Authority',
     'Indigo Ag Inc.',
     'Indigo Books & Music Inc.',
     'INDIGO ORGANIZING',
     'INDIGO SERVICE CORPORATION',
     'INDIGO IT LLC',
     'IndiGo',
     'INDIGO BOOKS',
     'INDIGO FINANCIAL GROUP INC.',
     'INDIGO IT',
     'Indigo',
     'INDIGO PRODUCTIONS, INC.',
     'INDIGO BOOKS & MUSIC INC',
     'Indigo Ag',
     'INDIGO-IP',
     'INDIGO INTERACTIVE',
     'INDIGO AG',
     'INDIGO',
     'Indigo Agriculture',
     ]

seed_strings = [
    '!ndigo',
    'IndiGO',
    'Indiana County Transit Authority',
    'Indigo',
    'Indigo Books & Music Inc.',
    'Indigo Agriculture'
    ]



sim = load_similarity_model(nama.root_dir/'models'/'nama_base.bin')

sim.to('cuda:3')


matcher = nama.Matcher(strings)
embeddings = sim.embed(matcher)
pred = embeddings.predict_voronoi(seed_strings=seed_strings,threshold=0.8)

pred.to_df()

strings = {s.upper() for s in strings}
seed_strings = {s.upper() for s in seed_strings}

matcher = nama.Matcher(strings)
embeddings = sim.embed(matcher)
pred = embeddings.predict_voronoi(seed_strings=seed_strings,threshold=0.5)

pred.to_df()
