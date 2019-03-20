import re

def ngrams(string,n=2):
    for i in range(len(string)-n+1):
        yield string[i:i+n]


def nmgrams(string,n=1,m=3):
    for j in range(n,m+1):
        for i in range(len(string)-j+1):
            yield string[i:i+j]


def simpleWords(string):
    for m in re.finditer(r'[A-Za-z0-9]+',string):
        yield m.group(0)
