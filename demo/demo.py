import os
import pandas as pd

import nama

# Create some simple dataframes to match
df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])

print(f'Toy data:\ndf1=\n{df1}\ndf2=\n{df2}')

# Nama is built around an object called a "matcher", which holds matching
# information about a set of strings and partitions the strings into
# non-overlapping groups.
#   - Strings in the same group are considered "matched"
#   - Strings in different groups are not matched.
# Nama provides tools for creating, modifying, saving, and loading matchers.
# Then matchers can be used to generate unique group ids for a set of
# strings, or perform two-way merges between pandas dataframes according to
# the match groups.

# We start matching by creating an empty matcher
matcher = nama.Matcher()

# First we need to add all the strings we want to match to the matcher
# (in this case the strings the name column of each dataframe)
matcher = matcher.add(df1['name'])
matcher = matcher.add(df2['name'])

# Initially, strings are automatically assigned to singleton groups
# (Groups are automatically labelled according to the most common string,
# with ties broken alphabetically)
print(f'Initial string groups:\n{matcher.groups}')

# At this point we can merge on exact matches, but there isn't much point
# (equivalent to pandas merge function)
print(f"Exact matching with singleton groups:\n{matcher.merge_dfs(df1,df2,on='name')}")

# To get better results, we need to modify the matcher.
# Unite merges all groups that contain the passed strings.
matcher = matcher.unite(['X Y Z CO','XYZ Co.'])
print(f'Updated string groups:\n{matcher.groups}')

# Unite is very flexible. We can pass a single set of strings, a nested list
# of strings, or mapping from strings to group labels. The mapping can even
# be a function that evaluates strings and generates a label.
# This makes it very simple to do hash collision matching.

# Hash collision matching works by matching any strings that have the same hash.
# A hash could be almost anything, but one useful way to do collision matching
# is to match strings that are identical after simplifying both strings.

# Nama provides some useful simplification functions in nama.strings.
# simplify_corp strips punctuation and capitalization, and removes common parts
# of names like starting with "the", or ending with "inc" or "ltd".

from nama.strings import simplify_corp

# Make a new matcher for comparison
corp_matcher = nama.Matcher(matcher.strings())

# Unite strings with the same simplified representation
corp_matcher = corp_matcher.unite(simplify_corp)

print(f'Groups after uniting by simplify_corp:\n{corp_matcher.groups}')

# Another useful approach to matching is to construct a similarity measure
# between strings. The standard way to do this is to break strings into "tokens"
# (words or short substrings) and use a measure like a weighted jaccard
# similarity index to summarize the overlap between the tokens in pairs of
# strings. The token_similarity module provides tools for matching based on
# token similarity.

# First, create a TokenSimilarity model. This can be customized with different
# tokenizers, similarity measures, and token weighting methods.

from nama.token_similarity import TokenSimilarity

token_model = TokenSimilarity()

# In the future: Use a training set to automatically pick the optimal similarity
# threshold for uniting strings.
# For now: Just set the threshold manually.

# Then we can use the similarity model to predict matches between the matcher
# strings. The predict method returns a new matcher.
token_matcher = token_model.predict(matcher.strings(),threshold=0.05)


# The nama.plot() function can help visualize the how strings are grouped in
# multiple matchers at the same time.

nama.plot([corp_matcher,token_matcher],matcher.strings(),matcher_names=['corp_matcher','token_matcher'])

# Notice that the combination of the two matchers correctly groups all the
# strings. It is often useful to combine multiple matching techniques.

# We can integrate the corp and token matchers into the original matcher
# with unite.

matcher = matcher.unite(corp_matcher)
matcher = matcher.unite(token_matcher)

nama.plot(matcher,matcher.strings())

# Now merging the dataframes gives us the desired output
print(f"Merging with the final matcher:\n{matcher.merge_dfs(df1,df2,on='name')}")

# The matcher can also be converted to a dataframe if we want to cluster the
# the names in one dataset or create a mapping to string groups that can be used
# accross multiple datasets.

print(f'Matcher as a dataframe:\n{matcher.to_df()}')

# Finally, we can save the matcher in csv format for later use

from nama import root_dir

demo_dir = root_dir/'demo'

if not os.path.isdir(demo_dir):
    os.makedirs(demo_dir)

matcher.to_csv(demo_dir/'matcher.csv')

# ...and load it again at a later time

loaded_matcher = nama.read_csv(demo_dir/'matcher.csv')

# Visually verify that the saved and loaded matchers are the same
nama.plot([matcher,loaded_matcher],matcher.strings(),matcher_names=['saved','loaded'])
