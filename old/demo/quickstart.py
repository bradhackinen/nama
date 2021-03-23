import os
import pandas as pd
from nama.matcher import Matcher
from nama.hashes import corpHash
from nama.rnnEmbedding import loadRnnEmbeddingModel

from nama.defaults import *


# Quick start ------------------------------------------------------------------

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

# Use fuzzy matching to find likely misses (GPU accelerated with device='cuda')
similarityModel = loadRnnEmbeddingModel(os.path.join(modelDir,'demoModel.bin'))

# Preview similar matches without applying them
# (Useful for choosing a cutoff or manually reviewing each one)
matcher.suggestMatches(similarityModel,min_score=0)

# Add all similarity matches with score >= 0.5
# (Alternatively, use the matcher.applyMatchDF method to add selected matches)
matcher.matchSimilar(similarityModel,min_score=0.5)

# Review matches - looks good!
matcher.matchesDF()

# Final merge
matcher.merge(df1,df2,on='name')

# We can also cluster names by connected component and assign ids to each
matcher.componentsDF()
matcher.componentSummaryDF()

# Or review matches that critical for linking components
matcher.matchImpactsDF()

# Or visualize the match graph
matcher.plotMatches()


# # matcher.plotMatches()
#
# matcher.plotMatches()
# matcher.plotMatches('xyz')
#
# matcher.addMatch('xyz','123')
# matcher.addMatch('456','123')
#
#
#
#
# # Graph simplification
# matcher.addMatch('google','1')
# matcher.addMatch('1','2')
# matcher.addMatch('2','Google Inc')
# matcher.addMatch('alphabet inc','3')
# matcher.addMatch('3','4')
# matcher.addMatch('4','5')
# matcher.addMatch('5','3')
# matcher.addMatch('google inc','6')
#
#
# matcher.simplify()
# matcher.plotMatches()
#
#
# # Match review
