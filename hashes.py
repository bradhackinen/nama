import re

def basicHash(s):
    '''
    A simple case and puctuation-insensitive hash
    '''
    s = s.lower()
    s = re.sub(' & ',' and ',s)
    s = re.sub(r'(?<=\S)[\'’´\.](?=\S)','',s)
    s = re.sub(r'[\s\.,:;/\'"`´‘’“”\(\)_—\-]+',' ',s)
    s = s.strip()

    return s


def corpHash(s):
    '''
    A hash function for corporate subsidiaries
    Insensitive to
        -case & puctation
        -'the' prefix
        -common corporation suffixes, including 'holding co'
    '''
    s = basicHash(s)
    if s.startswith('the '):
        s = s[4:]

    s = re.sub('( (group|holding(s)?( co)?|inc(orporated)?|ltd|l ?l? ?[cp]|co(rp(oration)?|mpany)?|s[ae]|plc))+$','',s,count=1)

    return s
