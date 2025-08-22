import re
import nama
import pandas as pd
from collections import Counter


# Preprocessors

def simplify(s):
    """
    A basic case string simplication function. Strips case and punctuation.

    Parameters
    ----------
    s : str
        The string to be simplified.

    Returns
    -------
    str
        The simplified string.
    """
    s = s.lower()
    s = re.sub('&', ' and ', s)
    s = re.sub(r'(?<=\S)[\'’´\.](?=\S)', '', s)
    s = re.sub(r'[\s\.,!@#$%^&*:;/\'"`´‘’“”\(\)_—\-]+', ' ', s)
    s = s.strip()

    return s


def simplify_corp(s):
    """
    A simplification function for corporations and law firms.
    Strips:
        - case & puctation
        - 'the' prefix
        - common corporate suffixes, including 'holding co'

    Parameters
    ----------
    s : str
        The string to be simplified.

    Returns
    -------
    str
        The simplified string.
    """
    s = simplify(s)
    if s.startswith('the '):
        s = s[4:]

    s = re.sub(
        '( (group|holding(s)?( co)?|inc(orporated)?|ltd|l ?l? ?[cp]|co(rp(oration)?|mpany)?|s[ae]|plc))+$',
        '',
        s,
        count=1)

    return s


def get_always_never_match_conflicts(always_match, never_match):

    # Convert always_match to a MatchData object
    if not isinstance(always_match, nama.MatchData):
        if isinstance(always_match[0], str):
            always_match = [always_match]

        strings = [s for g in always_match for s in g]

        always_match = nama.MatchData(strings).unite(always_match)

    # Ensure never_match is a nested list
    if all(isinstance(s,str) for s in never_match):
        never_match = [never_match]
    
    always_match_labels = always_match.labels

    # Find conflicts between never_match and always_match groups
    conflicts = []
    for g in never_match:
        g = sorted(list(g))
        g_labels = [always_match_labels.get(s,s) for s in g]
        g_label_counts = Counter(g_labels)
        if len(g_label_counts) < len(g):
            for label,count in g_label_counts.items():
                if count > 1:
                    conflict = {'conflict_strings':[s for i,s in enumerate(g) if g_labels[i]==label],
                                'never_match_group':g,
                                'always_match_group':always_match.groups[label]}
                    
                    conflicts.append(conflict)

    return conflicts


def check_always_never_match_conflicts(always_match, never_match,always_never_conflicts='error'):
    """
    Check for conflicts between always_match and never_match groups.
    If a conflict is found, raise a ValueError with the details of the conflict.

    Parameters
    ----------
    always_match : list
        List of strings or MatchData objects that should always match.
    never_match : list
        List of strings or MatchData objects that should never match.

    Raises
    ------
    ValueError
        If a conflict is found between always_match and never_match groups.
    """

    assert always_never_conflicts in ['error','warn','ignore'], 'always_never_conflicts must be one of "error", "warn", or "ignore"'
    
    conflicts = get_always_never_match_conflicts(always_match,never_match)

    if conflicts:
        if always_never_conflicts != 'ignore':
            
            conflicts_text = '\n'.join([f'{conflict["conflict_strings"]}' for conflict in conflicts])

            if always_never_conflicts == 'warn':
                warning_text = '\n'.join([
                                    f'The following groups of strings are required to always match *and* never match:',
                                    conflicts_text,
                                    'always_match relationships will take precedence.',
                                    'Use nama.utils.get_always_never_conflicts to see more information about the conflicts.'])
                print(warning_text)

            else:
                error_text = '\n'.join([
                                    f'The following groups of strings are required to always match *and* never match:',
                                    conflicts_text,
                                    'Use nama.utils.get_always_never_conflicts to see more information about the conflicts.'])
                    
                raise ValueError(error_text)
                    