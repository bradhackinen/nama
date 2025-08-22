import os
from pathlib import Path

import nama

data_dir = Path(os.environ['NAMA_DATA'])


embeddings = nama.load_embeddings(data_dir/'debug'/'never_match_failure'/'test_embeddings.zip')

never_match = [('AANP', 'ANP'), ('X', 'AANP'), ('ANP', 'Y')]

embeddings.unite_similar(
                threshold=0,
                never_match=never_match).to_df()



embeddings.unite_similar(
                threshold=0,
                never_match=[]).to_df()