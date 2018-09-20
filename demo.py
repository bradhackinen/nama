import pandas as pd
import nama

df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns='name')
df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns='name')

# Initialize the matcher
matcher = nama.matcher()

# Add the strings we want to match to the match graph
matcher.addStrings(df1['name'])
matcher.addStrings(df2['name'])

# At this point we can merge on exact matches, but there isn't much point (equivalent to pandas merge function)
matcher.merge(df1,df2,on='name')

# Connect strings if they share a hash string
# (corphash removes common prefixes and suffixes (the, inc, co, etc) and makes everything lower-case)
matcher.connectByHash(nama.hash.corphash)

# Now merge will find all the matches we want except  'ABC Inc.' <--> 'A.B.C. INCORPORATED'
matcher.merge(df1,df2,on='name')

# Use fuzzy matching to find likely misses (GPU accelerated with cuda=True)
matcher.loadSimilarityModel(cuda=True)
matcher.connectBySimilarity(min_score=0.5)

# Review fuzzy matches
fuzzyDF = matcher.edges(fuzzy=True)

# Add manual match
matcher.connect('ABC Inc.','A.B.C. INCORPORATED')

# Drop other fuzzy matches from the graph
matcher.disconnectBySimilarity(min_score=1)

# Final merge, ignoring fuzzy matches
matcher.merge(df1,df2,on='name')


# We can also cluster names and assign ids to each
clusterDF = matcher.clusters()
