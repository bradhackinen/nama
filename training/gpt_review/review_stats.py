import os
from pathlib import Path
import pandas as pd
import numpy as np
import regex as re
from tqdm import tqdm


data_dir = Path(os.environ['NAMA_DATA'])

review_dirs = sorted((data_dir/'training_data'/'processed'/'gpt_review'/'iterations').iterdir())


# Load GPT reviewed pairs
stats = []
for i,review_dir in enumerate(review_dirs):
    try:
        review_pairs_df = pd.read_csv(review_dir/'new_review_pairs.csv')
    except FileNotFoundError:
        print(f'No review file found in {review_dir}')
        continue

    try:
        gpt_df = pd.read_csv(review_dir/'gpt_reviewed.csv')
                  

    except FileNotFoundError:
        print(f'No GPT reviewed file found in {review_dir}')
        continue


    rating_pattern = r'Probability:\s*(\d)'
    gpt_df['gpt_rating'] = [int(re.search(rating_pattern,s,re.REVERSE).group(1)) if re.search(rating_pattern,s) else np.nan for s in gpt_df['gpt_rating']]

    gpt_df = pd.merge(review_pairs_df,gpt_df,on=['i','j'],how='left')
    
    gpt_df = (gpt_df
                .dropna()
                .sort_values(['gpt_rating','score'],ascending=False))


    stats.append({'iter':i,
                    'n_pairs':len(gpt_df),
                    'n_revise_match':((gpt_df['gpt_rating']<=2) & (gpt_df['training_match'])).sum(),
                    'n_revise_separate':((gpt_df['gpt_rating']>=4) & (~gpt_df['training_match'])).sum(),
                    'score_error':((gpt_df['gpt_rating']-1)/4 - gpt_df['score']).abs().mean(),
                        })
    # break

stats_df = pd.DataFrame(stats)

stats_df['n_revise'] = stats_df['n_revise_match'] + stats_df['n_revise_separate']
stats_df['n_agree'] = stats_df['n_pairs'] - stats_df['n_revise']
stats_df['frac_revise'] = (stats_df['n_revise'] / stats_df['n_pairs']).round(3)

stats_df.plot(kind='bar',y=['n_revise_match','n_revise_separate','n_agree'],stacked=True)

stats_df['score_error'].plot()

stats_df

# gpt_df[(gpt_df['gpt_rating']<=2) & (gpt_df['training_match'])].sample(10)
# gpt_df[(gpt_df['gpt_rating']>=4) & (~gpt_df['training_match'])].sample(10)

# gpt_df.groupby('gpt_rating')['score'].describe().T

