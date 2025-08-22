import os
from pathlib import Path
import pandas as pd
import numpy as np
import regex as re
from tqdm import tqdm

import nama
from nama.utils import simplify, get_always_never_match_conflicts
from nama.scoring import score_predicted, split_on_groups, get_confusion_df

data_dir = Path(os.environ['NAMA_DATA'])
device = 'cuda:2'

review_dirs = sorted((data_dir/'training_data'/'processed'/'gpt_review'/'iterations').iterdir())
current_review_dir = data_dir/'training_data'/'processed'/'gpt_review'/'iterations'/f'{len(review_dirs):03d}'


def load_raw_matches():
    
    # Load raw matches
    opensecrets_raw_matches = nama.read_csv(data_dir/'training_data'/'processed'/'opensecrets_raw_matches.csv')
    canlobby_raw_matches = nama.read_csv(data_dir/'training_data'/'processed'/'canlobby_raw_matches.csv')
    eulobby_raw_matches = nama.read_csv(data_dir/'training_data'/'processed'/'eulobby_raw_matches.csv')


    # Combine raw matches
    raw_matches = opensecrets_raw_matches + canlobby_raw_matches + eulobby_raw_matches

    # Unite by simplify - want to make sure we aren't missing easy matches
    raw_matches = raw_matches.unite(simplify)

    return raw_matches


def load_reviewed_pairs():
    
    # Load GPT reviewed pairs
    gpt_reviewed = []

    for review_dir in review_dirs:
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

        gpt_df = pd.merge(review_pairs_df,gpt_df,on=['i','j'],how='left')

        gpt_reviewed.append(gpt_df)

    if gpt_reviewed:
        
        gpt_df = pd.concat(gpt_reviewed)

        n_reviewed = len(gpt_df)

        print(f'Loaded {len(gpt_reviewed)} GPT reviewed files containing {n_reviewed} pairs')

        rating_pattern = r'Probability:\s*(\d)'
        gpt_df['gpt_rating'] = [int(re.search(rating_pattern,s,re.REVERSE).group(1)) if re.search(rating_pattern,s) else np.nan for s in gpt_df['gpt_rating']]

        gpt_df = (gpt_df
                    .dropna()
                    .sort_values(['gpt_rating','score'],ascending=False))

        print(f'Dropped {n_reviewed - len(gpt_df)} pairs with invalid GPT ratings')
        
        return gpt_df
    
    else:
        print('No GPT reviewed files found')
        return pd.DataFrame()

    #     always_match_pairs = [(i,j) for i,j in gpt_df[gpt_df['gpt_rating']>=4][['i','j']].values]
    #     never_match_pairs = [(i,j) for i,j in gpt_df[gpt_df['gpt_rating']<=2][['i','j']].values]

    #     return always_match_pairs,never_match_pairs
    
    # else:
    #     print('No reviewed pairs found')
    #     return [],[]


def unite_always_match_pairs(matches,always_match_pairs,never_match_pairs):

    matches = matches.copy()

    never_match_strings = {s:[] for pair in never_match_pairs for s in pair}
    for pair in never_match_pairs:
        never_match_strings[pair[0]].append(set(pair))
        never_match_strings[pair[1]].append(set(pair))

    skipped = []
    for i,j in tqdm(always_match_pairs,desc='Adding always match pairs'):
        gi = matches[i]
        gj = matches[j]

        if gi == gj:
            # Skip this match pair if they are already matched
            # print(f'Skipping "{i}" <--> "{j}" because they are already matched')
            continue
        
        # Check for conflicts with never match pairs
        new_group = set(matches.groups[gi] + matches.groups[gj])
        
        potential_conflict_groups = [g for s in new_group if s in never_match_strings for g in never_match_strings[s] ]
        conflict_groups = [g for g in potential_conflict_groups if len(g & new_group) == len(g)]

        if conflict_groups:
            # Skip this match pair if it conflicts with a never match pair
            # print(f'Skipping "{i}" <--> "{j}" because it conflicts with never match pairs')
            skipped.append((i,j))
            continue

        # If no conflicts, unite in always match data
        # print(f'Uniting "{i}" <--> "{j}"')
        matches._unite_strings([i,j])

    print(f'Skipped {len(skipped)} pairs due to conflicts with never match pairs')

    return matches


def load_training_data():

    if len(review_dirs):
        print(f'Loading reviewed matches from {review_dirs[-1]}')
        training_matches = nama.read_csv(review_dirs[-1]/'compiled_matches.csv')
    else:
        print('No reviewed matches found, loading raw matches')
        training_matches = load_raw_matches()

    return training_matches


training_matches = load_training_data()

# always_match_pairs,never_match_pairs = load_reviewed_pairs()

gpt_df = load_reviewed_pairs()

gpt_df.to_csv(data_dir/'training_data'/'processed'/'gpt_reviewed_pairs.csv',index=False)

always_match_pairs = [(i,j) for i,j in gpt_df[gpt_df['gpt_rating']>=4][['i','j']].values]
never_match_pairs = [(i,j) for i,j in gpt_df[gpt_df['gpt_rating']<=2][['i','j']].values]


print('Training similarity model')

# Match with similarity model, respecting never-match pairs
train_kwargs = {
                'max_epochs': 1,
                'warmup_frac': 0.1,
                'transformer_lr':5e-6,
                'score_lr':30,
                'alpha':20,
                'use_counts':False,
                'batch_size':8,
                'add_upper':True,
                'd':256,
                'model_name':'roberta-base',
                'prompt':'Organization: ',
                }


sim = nama.SimilarityModel(**train_kwargs)

sim.to(device)

history_df = sim.train(training_matches,verbose=True,**train_kwargs)

embeddings = sim.embed(training_matches)

embeddings.half()

os.makedirs(current_review_dir,exist_ok=True)

embeddings.save(current_review_dir/'embeddings.zip')

predicted = embeddings.unite_similar(
                group_threshold=0.5,
                never_match=never_match_pairs)



assert not get_always_never_match_conflicts(predicted,never_match_pairs)


matches = unite_always_match_pairs(predicted,always_match_pairs,never_match_pairs)


assert not get_always_never_match_conflicts(matches,never_match_pairs)



matches.to_csv(current_review_dir/'compiled_matches.csv')
matches.to_csv(data_dir/'training_data'/'processed'/'gpt_reviewed_matches.csv')


# matches = nama.read_csv(current_review_dir/'compiled_matches.csv')
# embeddings = nama.load_embeddings(current_review_dir/'embeddings.zip')


# Get new review pairs
review_matches,united_df = embeddings.unite_similar(
                        threshold=0.3,
                        never_match=never_match_pairs,
                        return_united=True)



united_df['training_match'] = [matches[i]==matches[j] for i,j in united_df[['i','j']].values]


mid_df = united_df.query('(score > 0.4) & (score < 0.6)')[['i','j','score','training_match']]


fp_df = united_df.query('~training_match & (score >= 0.6)')[['i','j','score','training_match']]



# Seek out extreme false negatives
fn_groups = set(get_confusion_df(review_matches,training_matches,use_counts=False)
            .query('FN>0')
            ['group_gold'])


fn_pairs = []
for group in tqdm(fn_groups,desc='Finding false negative pairs'):
    group_strings = training_matches.matched_to(group)

    if len(group_strings)>2:
        group_embeddings = embeddings[group_strings]
        _,group_united_df = group_embeddings.unite_similar(
                                    threshold=0,
                                    always_match=always_match_pairs,
                                    never_match=never_match_pairs,
                                    return_united=True,
                                    always_never_conflicts='ignore')
        
        group_fn_df = group_united_df[group_united_df['score']<0.5]
        if len(group_fn_df)>0:
            fn_pairs.extend([tuple(row) for row in group_fn_df[['i','j','score']].values])
    else:
        fn_pairs.append(tuple(group_strings + [np.nan]))


fn_df = pd.DataFrame(fn_pairs,columns=['i','j','score']).assign(training_match=True)
fn_df.loc[fn_df['score'].isnull(),'score'] = embeddings.score_pairs(fn_df.loc[fn_df['score'].isnull(),['i','j']])

fn_df = fn_df.sort_values('score',ascending=False)


review_df = pd.concat([fp_df,mid_df,fn_df])


# Filter out pairs that must always match
always_match = matches.split_all().unite(always_match_pairs)
review_df = review_df[[always_match[i]!=always_match[j] for i,j in review_df[['i','j']].values]]

# Filter out pairs that must never match
never_match = {(tuple(sorted(pair))) for pair in never_match_pairs}
review_df = review_df[[tuple(sorted(pair)) not in never_match for pair in review_df[['i','j']].values]]

review_df.to_csv(current_review_dir/'new_review_pairs.csv',index=False)

print(f'Selected {len(review_df)} pairs for review')





# review_df['score'].plot.hist(bins=100)



# review_df.query('score > 0.4').sample(10).sort_values('score',ascending=False)






# matches.matched_to('BP')


# review_df.sample(10).sort_values('score',ascending=False)



# # # JUNK TEST CODE ------------------------------------------------------
# # predicted.to_df().sample(10)


# # matches.matched_to('BP')
# # matches.matched_to('Boeing')


# df = matches.to_df()
# df[df['group'].str.contains('Boeing',case=False)]




# get_always_never_match_conflicts([['BOEING AIRCRAFT CO','BOEING CO']],never_match_pairs)

# embeddings.score_pairs([('BOEING AIRCRAFT CO','BOEING CO')])




# predicted,united_df = embeddings.unite_similar(
#                         threshold=0.4,
#                         group_threshold=0.4,
#                         never_match=never_match_pairs,
#                         return_united=True)


# united_df.query('score < 0.6')

# # JUNK TEST CODE ------------------------------------------------------





# get_always_never_match_conflicts(predicted,never_match_pairs)[-1]


# test_embeddings = embeddings[['AANP','ANP']]



# (test_embeddings.unite_similar(
#                 group_threshold=0.5,
#                 never_match=never_match_pairs)
#     .to_df())

# test_never_match_pairs = [(i,j) for i,j in never_match_pairs if i=='AANP' or j=='AANP']
# (test_embeddings.unite_similar(
#                 group_threshold=0.5,
#                 never_match=test_never_match_pairs)
#     .to_df())

# x = set(['AANP','ANP'])
# for i in range(1):
#     test_pairs = list(set((i,j) for i,j in never_match_pairs if i in x or j in x))
#     x = {i for pair in test_pairs for i in pair}

# (test_embeddings.unite_similar(
#                 group_threshold=0.5,
#                 never_match=test_pairs[4:5])
#     .to_df())

# (test_embeddings.unite_similar(
#                 group_threshold=0.5,
#                 never_match=test_pairs[4:])
#     .to_df())



# (test_embeddings.unite_similar(
#                 group_threshold=0.5,
#                 never_match=[('AANP', 'ANP'), ('X', 'AANP'), ('ANP', 'Y')])
#     .to_df())
