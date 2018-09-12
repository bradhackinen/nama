# nama
_super fast fuzzy name matching using pytorch and skearn_

Nama solves the problem of merging datasets by name (particularly company names) when the names might not be represented identically.

Key Features:
- Match by manually coded pairs, hash collisions, or a novel algorithm for very fast, trainable fuzzy matching
- Combine multiple matching techniques in a _match graph_
- Pairwise linking via a `merge` function for pandas DataFrames
- Name clustering for linking many datasets
- Interrogate the match graph to see how each match was found and highlight important links

# Quick start
```python
import pandas as pd
import nama
from nama.hashes import *


df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])

# Initialize the matcher
matcher = Matcher()

# Add the strings we want to match to the match graph
matcher.addStrings(df1['name'])
matcher.addStrings(df2['name'])

# At this point we can merge on exact matches, but there isn't much point (equivalent to pandas merge function)
matcher.merge(df1,df2,on='name')

# Match strings if they share a hash string
# (corphash removes common prefixes and suffixes (the, inc, co, etc) and makes everything lower-case)
matcher.matchHash(corpHash)

# Now merge will find all the matches we want except  'ABC Inc.' <--> 'A.B.C. INCORPORATED'
matcher.merge(df1,df2,on='name')

# Use fuzzy matching to find likely misses (GPU accelerated with cuda=True)
matcher.matchSimilar(min_score=0)

# Review fuzzy matches
connectionsDF = matcher.matchesDF()

# Add manual matches
matcher.addMatch('ABC Inc.','A.B.C. INCORPORATED')
matcher.addMatch('XYZ Co.','X Y Z CO')

# Drop remaining fuzzy matches from the graph
matcher.filterMatches(lambda m: m['source'] == 'similarity')

# Final merge
matcher.merge(df1,df2,on='name')

# We can also cluster names and assign ids to each
clusterDF = matcher.clustersDF()
```



# Introduction

Nama is built around the concept of a _match graph_. The match graph is a network of strings where edges represent matches between pairs. Matching is performed by first building the match graph, and then looking for connected strings, either in a pairwise way (is "_ABC Inc._" connected to "_the ABC company_"?) or by clustering components of connected strings. The match graph allows multiple types of matches to be combined, and it allows nama to infer that if A is linked to B and B is linked to C that A should also be linked to C.


 (powered by the excellent `networkx` module)


 There are many measures of text similarity between strings, but when the number of names is large (say, a few hundred thousand or more in each list), pairwise comparison takes a very long time. Nama uses multiple passes to efficiently match names:
1. Direct substition of training pairs
2. Matching by string 'hash' collisions (for example, linking all strings that have the same lower-case representation)
3. A novel neural network-based string embedding algorithm that produces vector representations of each name and uses an efficient nearest neighbors search to find fuzzy matches in linear time. Powered by PyTorch and scikit-learn.

## Requirements
- PyTorch 0.4
- sklearn
- networkx
- pandas
- numpy
- regex
- matplotlib
- seaborn
