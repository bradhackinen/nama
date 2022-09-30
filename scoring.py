import pandas as pd
import random



def confusion_df(predicted_matcher,gold_matcher,use_counts=True):

    df = pd.merge(predicted_matcher.to_df(),gold_matcher.to_df().drop('count',axis=1),on='string',suffixes=['_pred','_gold'])

    if not use_counts:
        df['count'] = 1

    df['TP'] = (df.groupby(['group_pred','group_gold'])['count'].transform('sum') - df['count']) * df['count']
    df['FP'] = (df.groupby('group_pred')['count'].transform('sum') - df['count']) * df['count'] - df['TP']
    df['FN'] = (df.groupby('group_gold')['count'].transform('sum') - df['count']) * df['count'] - df['TP']
    df['TN'] = (df['count'].sum() - df['count'])*df['count'] - df['TP'] - df['FP'] - df['FN']

    return df


def confusion_matrix(predicted_matcher,gold_matcher,use_counts=True):

    df = confusion_df(predicted_matcher,gold_matcher,use_counts=use_counts)

    return {c:df[c].sum()//2 for c in ['TP','FP','TN','FN']}


def score_predicted(predicted_matcher,gold_matcher,use_counts=True,drop_self_matches=True):
    """
    Computes the F1 score of a predicted matcher relative to a gold matcher
    which is assumed to be correct.
    """

    scores = confusion_matrix(predicted_matcher,gold_matcher,use_counts=use_counts)

    n_scored = scores['TP'] + scores['TN'] + scores['FP'] + scores['FN']

    if use_counts:
        n_predicted = (sum(predicted_matcher.counts.values())**2 \
                        - sum(c**2 for c in predicted_matcher.counts.values()))/2
    else:
        n_predicted = (len(predicted_matcher)**2 \
                        - len(predicted_matcher))/2

    scores['coverage'] = n_scored/n_predicted

    if scores['TP']:
        scores['accuracy'] = (scores['TP'] + scores['TN']) / n_scored
        scores['precision'] = scores['TP'] / (scores['TP'] + scores['FP'])
        scores['recall'] = scores['TP'] / (scores['TP'] + scores['FN'])
        scores['F1'] = 2*(scores['precision']*scores['recall']) / (scores['precision'] + scores['recall'])

    else:
        scores['accuracy'] = 0
        scores['precision'] = 0
        scores['recall'] = 0
        scores['F1'] = 0

    return scores


def split_on_groups(matcher,frac=0.5,seed=None):
    if seed is not None:
        random.seed(seed)

    groups = list(matcher.groups.values())
    random.shuffle(groups)

    selected_groups = groups[:int(frac*len(groups))]
    selected_strings = [s for group in selected_groups for s in group]

    return matcher.keep(selected_strings),matcher.drop(selected_strings)


def kfold_on_groups(matcher,k=4,shuffle=True,seed=None):
    if seed is not None:
        random.seed(seed)

    groups = list(matcher.groups.keys())

    if shuffle:
        random.shuffle(groups)
    else:
        groups = sorted(groups)

    for fold in range(k):

        fold_groups = groups[fold::k]
        fold_strings = [s for g in fold_groups for s in matcher.groups[g]]

        yield matcher.drop(fold_strings),matcher.keep(fold_strings)