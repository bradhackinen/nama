import pandas as pd
import pytest
from nama.scoring import score_predicted


# Mock MatchData class for testing purposes
class MatchData:
    def __init__(self, data, counts=None):
        self.data = data
        self.counts = counts if counts is not None else {}
    
    def to_df(self):
        return pd.DataFrame(self.data)
    
    def __len__(self):
        return len(self.data)

# Happy path test
def test_score_predicted_happy_path():
    predicted_data = {'string': ['A', 'B', 'C'], 'group_pred': [1, 2, 1], 'count': [2, 1, 3]}
    gold_data = {'string': ['A', 'B', 'C'], 'group_gold': [1, 2, 1], 'count': [1, 2, 2]}
    predicted_matches = MatchData(predicted_data)
    gold_matches = MatchData(gold_data)
    
    result = score_predicted(predicted_matches, gold_matches)

    assert set(result.keys()) == {'accuracy', 'precision', 'recall', 'F1', 'coverage', 'FN', 'FP', 'TN', 'TP'}


# Sad path test - no true positive (TP) matches
def test_score_predicted_no_tp():
    predicted_data = {'string': ['A', 'B', 'C'], 'group_pred': [1, 2, 1], 'count': [2, 1, 3]}
    gold_data = {'string': ['X', 'Y', 'Z'], 'group_gold': [1, 2, 1], 'count': [1, 2, 2]}
    predicted_matches = MatchData(predicted_data)
    gold_matches = MatchData(gold_data)
    
    result = score_predicted(predicted_matches, gold_matches)
    
    assert result['accuracy'] == 0
    assert result['precision'] == 0
    assert result['recall'] == 0
    assert result['F1'] == 0

# Edge case test - empty input
def test_score_predicted_empty_input():
    predicted_data = {'string': [], 'group_pred': [], 'count': []}
    gold_data = {'string': [], 'group_gold': [], 'count': []}
    predicted_matches = MatchData(predicted_data)
    gold_matches = MatchData(gold_data)
    
    result = score_predicted(predicted_matches, gold_matches)
    
    assert result['accuracy'] == 0
    assert result['precision'] == 0
    assert result['recall'] == 0
    assert result['F1'] == 0

# Test with use_counts=False and drop_self_matches=False
def test_score_predicted_use_counts_false_drop_self_matches_false():
    predicted_data = {'string': ['A', 'B', 'C'], 'group_pred': [1, 2, 1], 'count': [2, 1, 3]}
    gold_data = {'string': ['A', 'B', 'C'], 'group_gold': [1, 2, 1], 'count': [1, 2, 2]}
    predicted_matches = MatchData(predicted_data)
    gold_matches = MatchData(gold_data)
    
    result = score_predicted(predicted_matches, gold_matches, use_counts=False, drop_self_matches=False)
    
    assert set(result.keys()) == {'accuracy', 'precision', 'recall', 'F1', 'coverage', 'FN', 'FP', 'TN', 'TP'}
