import numpy as np
import re
import random
from tqdm import tqdm

def remove_random_char(s):
    if len(s) >= 2:
        i = np.random.randint(len(s))
        s = s[:i] + s[i+1:]
    return s


def replicate_random_char(s):
    if len(s) >= 2:
        i = np.random.randint(len(s))
        j = np.random.randint(len(s))
        s = s[:i] + s[j] + s[i:]
    return s


def strip_the(s):
    s = re.sub(r'(^the )|(, the$)','',s,flags=re.IGNORECASE)
    return s


def swap_and(s):
    if '&' in s:
        s = s.replace(' & ',' and ')
    else:
        s = re.sub(' and ',' & ',s,flags=re.IGNORECASE)
    return s


def strip_legal(s):
    s = re.sub(r',?( (group|holding(s)?( co)?|inc(orporated)?|ltd|l\.?l?\.?[cp]|co(rp(oration)?|mpany)?|s\.?[ae]|p\.?l\.?c)[,\.]*)$','',s,count=1,flags=re.IGNORECASE)
    return s


def to_acronym(s):
    s = strip_the(s)
    s = strip_legal(s)
    s = re.sub(' and ',' ',s,flags=re.IGNORECASE)

    tokens = s.split()
    if len(tokens) > 1:
        return ''.join(t[0].upper() for t in tokens if t.lower() not in ['of','the','for'])
    else:
        return s.upper()


def truncate_words(s):
    tokens = []
    for t in s.split():
        if len(t) >= 6:
            tokens.append(t[:np.random.randint(2,len(t)-2)])
        else:
            tokens.append(t)
    s = ' '.join(tokens)
    return s


default_mutations = [
                strip_legal,
                lambda s: strip_legal(strip_legal(s)),
                strip_the,
                swap_and,
                remove_random_char,
                replicate_random_char,
                to_acronym,
                truncate_words,
                lambda s: ' '.join(s.split()[:-1]),
                lambda s: re.sub(r',.*','',s,flags=re.IGNORECASE),
                lambda s: re.sub(r'[.,:;]','',s),
                lambda s: re.sub(r'[.,:;\-]',' ',s).strip(),
                lambda s: re.sub(r'(?<=\w)[aeiouy]','',s,flags=re.IGNORECASE),
                lambda s: s.title(),
                lambda s: s.upper()
            ]


def random_mutation(s,mutations=default_mutations,max_attempts=None,seed=None):
    if seed is not None:
        random.seed(seed)

    mutations = list(mutations)

    if not max_attempts:
        max_attempts = len(mutations)

    random.shuffle(mutations)
    for mutation in mutations[:max_attempts]:
        m = mutation(s)
        if m != s:
            return m
    return s


def augment_matcher(matcher,n=1,mutations=default_mutations,seed=None):
    if seed is not None:
        random.seed(seed)

    new_strings = {s:[random_mutation(s,mutations) for i in range(n)] for s in tqdm(matcher.strings(),desc='Augmenting matcher')}
    matcher = matcher.add_strings([s for m in new_strings.values() for s in m])
    matcher = matcher.unite([[s]+m for s,m in new_strings.items()])

    return matcher
