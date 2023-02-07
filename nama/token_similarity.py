from itertools import combinations
from collections import Counter
import pandas as pd
import numpy as np
import regex as re
from tqdm import tqdm

from .matcher import Matcher
from .scoring import score_predicted


def ngrams(string,n=2):
    for i in range(len(string)-n+1):
        yield string[i:i+n]


def nmgrams(string,n=1,m=3):
    for j in range(n,m+1):
        for i in range(len(string)-j+1):
            yield string[i:i+j]


def words(string):
    for m in re.finditer(r'[A-Za-z0-9]+',string):
        yield m.group(0)

def jaccard_similarity(set0,set1,weights):

    intersection = set0 & set1

    if not intersection:
        return 0

    union = set0 | set1
    denominator = sum(weights[t] for t in union)

    if not denominator:
        return 0

    numerator = sum(weights[t] for t in intersection)
    return numerator/denominator


def cosine_similarity(set0,set1,weights):

    intersection = set0 & set1

    if not intersection:
        return 0

    length0 = np.sqrt(sum(weights[t]**2 for t in set0))
    length1 = np.sqrt(sum(weights[t]**2 for t in set1))

    denominator = length0*length1
    if not denominator:
        return 0

    numerator = sum(weights[t]**2 for t in intersection)

    return numerator/denominator

class TokenSimilarity():

    def __init__(self,tokenizer=lambda s: nmgrams(s,2,3),weighting='tf-idf',
                        measure='jaccard',max_block_size=100):
        """
        Configures the token similarity model

        Arguments:
        tokenizer -- A function that takes a string an returns an iterable token
                     generator (default: bigrams and trigrams)
        weighting -- One of: None,"tf","idf", or ""tf-idf"
          measure -- Either "jaccard" or "cosine"
        """

        self.tokenizer = tokenizer
        self.weighting = weighting
        self.measure = measure
        self.max_block_size = max_block_size

        # Assign weighting function
        if callable(weighting):
            self.weight_func = weighting
        elif weighting == 'tf':
            self.weight_func = lambda t,f,d: f
        elif weighting == 'idf':
            self.weight_func = lambda t,f,d: 1/np.log(1+d)
        elif weighting == 'tf-idf':
            self.weight_func = lambda t,f,d: f/np.log(1+d)
        else:
            raise ValueError('Unknown weighting type')

        # Assign scoring function
        if callable(measure):
            self.score_func = measure
        elif measure == 'cosine':
            self.score_func = cosine_similarity
        elif measure == 'jaccard':
            self.score_func = jaccard_similarity
        else:
            raise ValueError('Unknown measure value')

        # Set threshold to None initially
        self.threshold = None

    def fit(self,matcher,learn_threshold=False,):
        self.strings = matcher.strings()

        # Loops will run a little faster if we make local references to functions
        tokenizer = self.tokenizer
        weight_func = self.weight_func

        # Tokenize strings
        tokenized = {s:list(tokenizer(s)) for s in self.strings}

        # Count occurrences of tokens
        counts = Counter(t for tokens in tokenized.values() for t in tokens)

        # Convert tokens to sets
        # (drops frequency information, and also makes membership tests faster)
        self.tokenized = {s:set(tokens) for s,tokens in tokenized.items()}

        # Count tokens again, this time tracking the number of strings
        # containing the token. In NLP this is usually referred to as the
        # "document count"
        doc_counts = Counter(t for tokens in tokenized.values() for t in tokens)
        self.doc_counts = doc_counts

        # Build weights
        self.weights = {t:weight_func(t,f,doc_counts[t]) for t,f in counts.items()}

    def learn_threshold(self,gold_matcher,objective='F1',grid=np.linspace(0.5,1,100),use_counts=False):
        """
        Uses train_matcher as a training set to choose the default similarity
        threshold.
        """
        self_copy = TokenSimilarity(tokenizer=self.tokenizer,weighting=self.weighting,measure=self.measure,max_block_size=self.max_block_size)
        self_copy.fit(gold_matcher)

        scores = []
        for t in tqdm(grid):
            pred = self_copy.predict(threshold=t)
            s = score_predicted(pred,gold_matcher,use_counts=use_counts)
            s['threshold'] = t
            scores.append(s)

        scores_df = pd.DataFrame(scores)

        self.threshold = scores_df[scores_df[objective] == scores_df[objective].max()]['threshold'].values[-1]

        return scores_df

    def test(self,test_matcher):
        """
        Evaluates the accuracy of the model using test_matcher as the
        gold-standard test set.

        Note: For a fair test, it may be important to ensure that the test and
        train matchers have no strings in common.
        """

        predicted = self.predict(test_matcher.strings())

        scores = scores = score_predicted(predicted,test_matcher)

        return scores

    def predict(self,strings=None,threshold=None):
        """
        Uses the similarity model to predict matches between the passed strings.
        """
        if strings:
            self.fit(strings)

        if threshold is None:
            if self.threshold is None:
                raise ValueError('Must set a threshold value, either by calling .learn_threshold(), or by passing a manually chosen value as an argument')
            else:
                threshold = self.threshold

        strings = self.strings
        tokenized = self.tokenized
        weights = self.weights
        score_func = self.score_func

        predicted = Matcher(strings)

        # Iterate over unique tokens
        for t,d in self.doc_counts.items():
            # Find all strings that share this token (the "block")
            if 2 <= d <= self.max_block_size:
                block = [s for s in strings if t in tokenized[s]]

                # Score all pairs in the block
                for s0,s1 in combinations(block,2):

                    # Can skip pairs that are already in the same group
                    if predicted[s0] != predicted[s1]:

                        # Unite strings with score >= threshold
                        score = score_func(tokenized[s0],tokenized[s1],weights)
                        if score >= threshold:
                            predicted.unite([s0,s1],inplace=True)

        return predicted


o=TokenSimilarity()