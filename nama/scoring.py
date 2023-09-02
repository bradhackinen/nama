import pandas as pd
import random


def confusion_df(predicted_matches, gold_matches, use_counts=True):
    """
    Computes the confusion matrix dataframe for a predicted match groups object relative to a gold match groups object.

    Parameters
    ----------
    predicted_matches : MatchData
        The predicted match groups object.
    gold_matches : MatchData
        The gold match groups object.
    use_counts : bool, optional
        Use the count of each string. If False, the count is set to 1.

    Returns
    -------
    df : pandas.DataFrame
        Confusion matrix dataframe with columns 'TP', 'FP', 'TN', and 'FN'.
    """

    df = pd.merge(
        predicted_matches.to_df(),
        gold_matches.to_df().drop(
            'count',
            axis=1),
        on='string',
        suffixes=[
            '_pred',
            '_gold'])

    if not use_counts:
        df['count'] = 1

    df['TP'] = (df.groupby(['group_pred', 'group_gold'])[
                'count'].transform('sum') - df['count']) * df['count']
    df['FP'] = (df.groupby('group_pred')['count'].transform(
        'sum') - df['count']) * df['count'] - df['TP']
    df['FN'] = (df.groupby('group_gold')['count'].transform(
        'sum') - df['count']) * df['count'] - df['TP']
    df['TN'] = (df['count'].sum() - df['count']) * \
        df['count'] - df['TP'] - df['FP'] - df['FN']

    return df


def confusion_matrix(predicted_matches, gold_matches, use_counts=True):
    """
    Computes the confusion matrix for a predicted match groups object relative to a gold match groups object.

    Parameters
    ----------
    predicted_matches : MatchData
        The predicted match groups object.
    gold_matches : MatchData
        The gold match groups object.
    use_counts : bool, optional
        Use the count of each string. If False, the count is set to 1.

    Returns
    -------
    confusion_matrix : dict
        Dictionary with keys 'TP', 'FP', 'TN', and 'FN', representing the values in the confusion matrix.
    """

    df = confusion_df(predicted_matches, gold_matches, use_counts=use_counts)

    return {c: df[c].sum() // 2 for c in ['TP', 'FP', 'TN', 'FN']}


def score_predicted(
        predicted_matches,
        gold_matches,
        use_counts=True,
        drop_self_matches=True):
    """
    Computes the F1 score of a predicted match groups object relative to a gold match groups object
    which is assumed to be correct.

    Parameters
    ----------
    predicted_matches : MatchData
        The predicted match groups object .
    gold_matches : MatchData
        The gold match groups object.
    use_counts : bool, optional
        Use the count of each string. If False, the count is set to 1.
    drop_self_matches : bool, optional
        Remove the matches between a string and itself.

    Returns
    -------
    scores : dict
        Dictionary with keys 'accuracy', 'precision', 'recall', 'F1', and 'coverage'.
    """

    scores = confusion_matrix(
        predicted_matches,
        gold_matches,
        use_counts=use_counts)

    n_scored = scores['TP'] + scores['TN'] + scores['FP'] + scores['FN']

    if use_counts:
        n_predicted = (sum(predicted_matches.counts.values())**2 -
                       sum(c**2 for c in predicted_matches.counts.values())) / 2
    else:
        n_predicted = (len(predicted_matches)**2
                       - len(predicted_matches)) / 2

    scores['coverage'] = n_scored / n_predicted

    if scores['TP']:
        scores['accuracy'] = (scores['TP'] + scores['TN']) / n_scored
        scores['precision'] = scores['TP'] / (scores['TP'] + scores['FP'])
        scores['recall'] = scores['TP'] / (scores['TP'] + scores['FN'])
        scores['F1'] = 2 * (scores['precision'] * scores['recall']) / \
            (scores['precision'] + scores['recall'])

    else:
        scores['accuracy'] = 0
        scores['precision'] = 0
        scores['recall'] = 0
        scores['F1'] = 0

    return scores


def split_on_groups(matches, frac=0.5, seed=None):
    """
    Splits the match groups object into two parts by given fraction.

    Parameters
    ----------
    matches : MatchData
        The match groups object to be split.
    frac : float, optional
        The fraction of groups to select.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    matches1, matches2 : tuple of match groups objects
        Tuple of two match groups objects.
    """
    if seed is not None:
        random.seed(seed)

    groups = list(matches.groups.values())
    random.shuffle(groups)

    selected_groups = groups[:int(frac * len(groups))]
    selected_strings = [s for group in selected_groups for s in group]

    return matches.keep(selected_strings), matches.drop(selected_strings)


def kfold_on_groups(matches, k=4, shuffle=True, seed=None):
    """
    Perform K-fold cross validation on groups of strings.

    Parameters
    ----------
    matches : object
        MatchData object to perform K-fold cross validation on.
    k : int, optional
        Number of folds to perform, by default 4.
    shuffle : bool, optional
        Whether to shuffle the groups before splitting, by default True.
    seed : int, optional
        Seed for the random number generator, by default None.

    Yields
    ------
    tuple : MatchData, MatchData
        A tuple of k match groups objects, the first for the training set and the second for the testing set for each fold.
    """
    if seed is not None:
        random.seed(seed)

    groups = list(matches.groups.keys())

    if shuffle:
        random.shuffle(groups)
    else:
        groups = sorted(groups)

    for fold in range(k):

        fold_groups = groups[fold::k]
        fold_strings = [s for g in fold_groups for s in matches.groups[g]]

        yield matches.drop(fold_strings), matches.keep(fold_strings)
