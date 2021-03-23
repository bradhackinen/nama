

def compare_positives(predicted_matcher,gold_matcher,use_counts=True):
    """
    Computes pairwise True Positives and False Positives for a predicted
    matcher relative to a gold matcher which is assumed to be correct.

    Strings which are not in the gold_matcher are ignored.
    """
    tp = 0
    fp = 0
    for group in predicted_matcher.groups.values():
        gold_group_counts = Counter()
        for s in group:
            if s in gold_matcher:
                gold_label = gold_matcher.labels[s]
                if use_counts:
                    gold_group_counts[gold_label] += predicted_matcher.counts[s]
                else:
                    gold_group_counts[gold_label] += 1

        total_counted = sum(gold_group_counts.values())
        for g,count in gold_c_counts.items():
            """
            All pairs of strings that are in the predicted group, and
            gold_matcher group g are True Positives.

            "count" gives the number of strings in this set, so there must be
            0.5*count*(count-1) unique unordered string pairs in this category.
            """
            tp += 0.5*count*(count-1)

            """
            All pairs of strings that are in predicted group, with one string in
            the gold_matcher group g and one string in a different gold_matcher
            group are False Positives.

            There must be count*(total_counted-count) unique unordered pairs in
            this category.

            We multiply this number by 0.5 to correct for the fact that each
            pair will be counted twice (once for each string in the pair).
            """
            fp += 0.5*count*(total-count)

    return {'TP':tp,'FP':fp}


def confusion_matrix(predicted_matcher,gold_matcher,use_counts=True):
    """
    Computes values of the confusion matrix comparing the predicted matcher
    to a gold matcher which is assumed to be correct.
    """

    positives = compare_positives(predicted_matcher,gold_matcher,use_counts=True)

    # "False Positives" indicate False Negatives when the matchers are reversed
    negatives = compare_positives(gold_matcher,predicted_matcher,use_counts=True)

     # The True Positive count returned from each function call always should be identical
    assert positives['TP'] == negatives['FP']

    tp = positives['TP']
    fp = positives['FP']
    fn = negatives['FP']

    if use_counts:
        n = sum(c for s,c in predicted_matcher.counts.items() if s in gold_matcher)
    else:
        n = sum(1 for s,c in predicted_matcher.counts.items() if s in gold_matcher)

    tn = n**2 - tp - fp - fn

    return {'TP':tp,'FP':fp,'TN',tn,'FN',fn}


def confusion_matrix_slow(predicted_matcher,gold_matcher,use_counts=True):
    """
    Computes values of the confusion matrix comparing the predicted matcher
    to a gold matcher which is assumed to be correct.
    """

    # The following code would work, but would be very slow.
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for s0 in predicted_matcher.strings():
        for s1 in predicted_matcher.strings():

            if s0 in gold_matcher and s1 in gold_matcher:

                if use_counts:
                    count = predicted_matcher.counts[s0]*predicted_matcher.counts[s1]
                else:
                    count = 1

                #Check if strings are in the same match group in the predicted matcher
                if gold_matcher[s0] == gold_matcher[s1]:
                    # check if strings are in the same match group in the gold matcher
                    if gold_matcher[s0] == gold_matcher[s1]:
                        tp += count
                    else:
                        fp += count
                else:
                    # check if strings are in the same match group in the gold matcher
                    if gold_matcher[s0] == gold_matcher[s1]:
                        fn += count
                    else:
                        tn += count

    return {'TP':tp,'FP':fp,'TN',tn,'FN',fn}


def score_predicted(predicted_matcher,gold_matcher,use_counts=True):
    """
    Computes the F1 score of a predicted matcher relative to a gold matcher
    which is assumed to be correct.
    """

    scores = confusion_matrix(predicted_matcher,gold_matcher,use_counts=True)

    scores['accuracy'] = scores['TP'] + scores['FP'] \
                        / (scores['TP'] + scores['TN'] + scores['FP'] + scores['FN'] )

    scores['precision'] = scores['TP'] / (scores['TP'] + scores['FN'])
    scores['recall'] = scores['TP'] / (scores['TP'] + scores['FP'])

    scores['F1'] = 2*(score['precision'] + scores['recall']) / (score['precision']*scores['recall'])

    return scores
