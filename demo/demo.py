import pandas as pd

import nama

# Create some simple dataframes with names to match
df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])
df3 = pd.DataFrame(['A.B.C. Inc.','XYZ Company, The'],columns=['name'])

print(f'Toy data:\ndf1=\n{df1}\ndf2=\n{df2}\ndf3=\n{df3}')

# Nama is built around an object called a "MatchData", which holds matching
# information about a set of strings and partitions the strings into
# non-overlapping groups.
#   - Strings in the same group are considered "matched"
#   - Strings in different groups are not matched.
# Nama provides tools for creating, modifying, saving, and loading MatchData objects.
# Then MatchData objects can be used to generate unique group ids for a set of
# strings, or perform two-way merges between pandas dataframes according to
# the match groups.

# We start matching by creating an empty MatchData object
matcher = nama.MatchData()

# First we need to add all the strings we want to match to the MatchData object
# (in this case the strings the name column of each dataframe)
matcher = matcher.add_strings(df1['name'])
matcher = matcher.add_strings(df2['name'])
matcher = matcher.add_strings(df3['name'])

# Initially, strings are automatically assigned to singleton groups
# Printing a MatchData object will show the first 50 strings, with empty lines indicating
# separations between groups.
print(f'\nInitial matcher:\n\n{matcher}')

# At this point we can merge on exact matches, but there isn't much point
# (equivalent to pandas merge function)
print(f"Exact matching with singleton groups:\n{matcher.merge_dfs(df1,df2,on='name')}")

# To get better results, we need to modify the matcher.
# Unite merges all groups that contain the passed strings.
matcher = matcher.unite(['X Y Z CO','XYZ Co.'])
print(f'\nUpdated matcher:\n\n{matcher}')

# Unite is very flexible. We can pass a single set of strings, a nested list
# of strings, or mapping from strings to group labels. The mapping can even
# be a function that evaluates strings and generates a label.
# This makes it very simple to do hash collision matching.

# Hash collision matching works by matching any strings that have the same hash.
# A hash could be almost anything, but one useful way to do collision matching
# is to match strings that are identical after simplifying both strings.

# Nama provides some useful simplification functions in nama.utils.
# simplify_corp strips punctuation and capitalization, and removes common parts
# of names like starting with "the", or ending with "inc" or "ltd".

from nama.utils import simplify_corp

# Make a new matcher for comparison and split all the groups for a fresh start
corp_matcher = matcher.split_all()

# Unite strings with the same simplified representation
corp_matcher = corp_matcher.unite(simplify_corp)

print(f'\nMatcher after uniting by simplify_corp:\n\n{corp_matcher}')

# Now merging the dataframes gives us better results
print(f"Merging with corp_matcher:\n{corp_matcher.merge_dfs(df1,df2,on='name')}")

# The matcher can also be converted to a dataframe if we want to cluster the
# the names in one dataset or create a mapping to string groups that can be used
# across multiple datasets.

print(f'Matcher as a dataframe:\n{corp_matcher.to_df()}')

# Finally, we can save the matcher in csv format for later use
corp_matcher.to_csv('matcher.csv')

# ...and load it again at a later time
loaded_matcher = nama.read_csv('matcher.csv')

print(f'Loaded matcher:\n{loaded_matcher}')