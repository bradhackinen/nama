from pathlib import Path
import os

import nama
from nama.scoring import split_on_groups


data_dir = Path(os.environ['NAMA_DATA'])
save_dir = data_dir/'paper'/'training_data'

gpt_reviewed = nama.read_csv(Path(data_dir)/'training_data'/'processed'/'gpt_reviewed_matches.csv')
canlobby = nama.read_csv(data_dir/'training_data'/'processed'/'canlobby_raw_matches.csv')



canlobby_test = canlobby.copy()
canlobby_train = gpt_reviewed.drop(canlobby.strings())

test,train = split_on_groups(canlobby_train,0.2,seed=123)


os.makedirs(save_dir,exist_ok=True)


train.to_csv(save_dir/'train.csv')
test.to_csv(save_dir/'20pct_test.csv')
canlobby_test.to_csv(save_dir/'canlobby_test.csv')
