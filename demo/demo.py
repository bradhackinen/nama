import os
import pandas as pd

import nama

# Create some simple dataframes with names to match
df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])
df3 = pd.DataFrame(['A.B.C. Inc.','XYZ Company, The'],columns=['name'])

print(f'Toy data:\ndf1=\n{df1}\ndf2=\n{df2}\ndf3=\n{df3}')

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
matcher = matcher.add_strings(df1['name'])
matcher = matcher.add_strings(df2['name'])
matcher = matcher.add_strings(df3['name'])

# Initially, strings are automatically assigned to singleton groups
# Printing a matcher will show the first 50 strings, with empty lines indicating
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

# Nama provides some useful simplification functions in nama.strings.
# simplify_corp strips punctuation and capitalization, and removes common parts
# of names like starting with "the", or ending with "inc" or "ltd".

from nama.strings import simplify_corp

# Make a new matcher for comparison and split all the groups for a fresh start
corp_matcher = matcher.split_all()

# Unite strings with the same simplified representation
corp_matcher = corp_matcher.unite(simplify_corp)

print(f'\nMatcher after uniting by simplify_corp:\n\n{corp_matcher}')

# Another useful approach to matching is to construct a similarity measure
# between strings. The standard way to do this is to break strings into "tokens"
# (words or short substrings) and use a measure like a weighted jaccard
# similarity index to summarize the overlap between the tokens in pairs of
# strings. The token_similarity module provides tools for matching based on
# token similarity.

# First, create a TokenSimilarity model. This can be customized with different
# tokenizers, similarity measures, and token weighting methods.

from nama.embedding_similarity import EmbeddingSimilarityModel

# Creating a new similarity model
similarity_model = EmbeddingSimilarityModel()

# Training a similarity model
similarity_model.train(training_matcher)


# Loading and saving
save_file = nama.root_dir/'_review'/'temp.bin'
similarity_model.save(save_file)

similarity_model = nama.embedding_similarity.load(save_file)




# Then we can use the similarity model to predict matches between the matcher
# strings. The predict method returns a new matcher.
predicted = similarity_model.predict(matcher,threshold=0.5)

separated = similarity_model.separate(predicted,['A','B','C'])




matcher.iter_scored_pairs(simi)

similarity_model.iter_scored_pairs(matcher)





predicted = matcher.predict(similarity_model,threshold=0.5)


separated = matcher.separate(['A','B','C'],similarity_model)






similarity_model.top_scored_pairs(matcher,is_match=False,min_score=0,sort_by='score',ascending=False)

print(sim_matcher)
# The nama.plot() function can help visualize the how strings are grouped in
# multiple matchers at the same time.

nama.plot([corp_matcher,sim_matcher],matcher.strings(),matcher_names=['corp_matcher','token_matcher'])

# Notice that the combination of the two matchers correctly groups all the
# strings. It is often useful to combine multiple matching techniques.

# We can integrate the corp and token matchers into the original matcher
# with unite.

matcher = matcher.unite(corp_matcher)
matcher = matcher.unite(sim_matcher)

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
