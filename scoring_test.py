import random
from collections import Counter

import nama
from nama import scoring


def _confusion_matrix_slow(predicted_matcher,gold_matcher):
    """
    A slow, simple version of the confusion matrix calculation for testing
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    strings = list(set(predicted_matcher.strings()) & set(gold_matcher.strings()))
    counts = predicted_matcher.counts
    n = len(strings)
    for i in range(n):
        for j in range(i+1,n):
            s0 = strings[i]
            s1 = strings[j]

            # Count the number of strings in this cell as if the counts represent
            # copies of the strings
            n_cell = counts[s0]*counts[s1]

            # Check if strings are in the same match group in the predicted matcher
            if predicted_matcher[s0] == predicted_matcher[s1]:
                # check if strings are in the same match group in the gold matcher
                if gold_matcher[s0] == gold_matcher[s1]:
                    tp += n_cell
                else:
                    fp += n_cell
            else:
                # check if strings are in the same match group in the gold matcher
                if gold_matcher[s0] == gold_matcher[s1]:
                    fn += n_cell
                else:
                    tn += n_cell

    return {'TP':tp,'FP':fp,'TN':tn,'FN':fn}


def rand_matcher(n=10,max_groups=None,max_count=3):
    if not max_groups:
        max_groups = n

    strings = Counter({str(i):random.randint(1,max_count) for i in range(n)})

    matcher = nama.Matcher()
    matcher = matcher.add_strings(strings)

    groups = {s:random.randint(1,max_groups) for s in matcher.strings()}
    matcher = matcher.unite(groups)

    return matcher


for i in range(10):
    rand_0 = rand_matcher(100)
    rand_1 = rand_matcher(100)

    assert scoring.confusion_matrix(rand_0,rand_1) == _confusion_matrix_slow(rand_0,rand_1)
