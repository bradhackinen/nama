import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

import nama


data_dir = Path(os.environ['NAMA_DATA'])

names_df = pd.read_csv(Path(data_dir)/'training_data'/'raw'/'eulobby'/'Entity_name_ID.csv')

matches = (nama.MatchData(names_df['Entity_name'])
           .unite(names_df.groupby('TR_id')['Entity_name'].apply(list)))


matches.to_csv(data_dir/'training_data'/'processed'/'eulobby_raw_matches.csv')
