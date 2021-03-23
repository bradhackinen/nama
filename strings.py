import re

def simplify(s):
    """
    A basic case string simplication function. Strips case and punctuation.
    """
    s = s.lower()
    s = re.sub(' & ',' and ',s)
    s = re.sub(r'(?<=\S)[\'’´\.](?=\S)','',s)
    s = re.sub(r'[\s\.,:;/\'"`´‘’“”\(\)_—\-]+',' ',s)
    s = s.strip()

    return s


def simplify_corp(s):
    """
    A simplification function for corporations and law firms.
    Strips:
        - case & puctation
        - 'the' prefix
        - common corporate suffixes, including 'holding co'
    """
    s = simplify(s)
    if s.startswith('the '):
        s = s[4:]

    s = re.sub('( (group|holding(s)?( co)?|inc(orporated)?|ltd|l ?l? ?[cp]|co(rp(oration)?|mpany)?|s[ae]|plc))+$','',s,count=1)

    return s
