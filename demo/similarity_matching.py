import os
from pathlib import Path
import pandas as pd

import nama

# Locate the pre-trained model directory:
#   - This line can be changed to point to a different model directory
#   - Or the environment variable "NAMA_DATA" can be set to avoid changing this line
model_dir = Path(os.environ['NAMA_DATA'])/'models'

# First, we'll construct a tricky (but realistic) case using Disney subsidaries
# The counts here is just hypothetical. Counts are used to:
# - Choose a representative string as a name for each group.
# - Optionally weight strings when evaluating accuracy.

test_names = pd.DataFrame(['Walt Disney World Resort']*13
                + ['Walt Disney World']*12
                + ['Disney World']*11
                + ['DISNEY WORLD']*10
                + ['Disney Worldwide Services']*9
                + ['Disney Worldwide Services, Inc.']*8
                + ['DISNEY WORLDWIDE SERVICES']*7
                + ['DISNEY TV STUDIOS']*6
                + ['DISNEY TELEVISON GROUP']*5
                + ['Disney TV']*4
                + ['DISNEY INTERACTIVE, INC.']*3
                + ['Disney Interactive']*2
                + ['Disney Interactive Media Group']*1
                + ['Pixar Animation Studios'],
                columns=['name'])

# Nama uses match data objects to store strings, their counts, and group memberships
# We can convert a list or column of raw names to a match_data object like so:
test_data = nama.MatchData(test_names['name'])

# By default, strings are assigned to singleton groups.

# Match data objects can be converted to dataframes for easy viewing
print('\nTest data (unmatched names):')
print(test_data.to_df())

# Load a pre-trained similarity model
sim = nama.load_similarity_model(model_dir/'nama-256.bin')

# Send the model to the GPU
sim.to('cuda:0')

# Embed the strings using the similarity model
# Note: With a large number of strings, this can take a while. GPU acceleration helps a lot.
embeddings = sim.embed(test_data)


# Now we can do simple similarity-based matching
# Note: This can also take a long time with a large number of strings.
matches = embeddings.unite_similar()

print('\nSimple similarity matching results:')
print(matches.to_df().sort_values('count',ascending=False))

# If we adjust the threshold, we get some rough matching that is OK
matches = embeddings.unite_similar(threshold=0.75)

print('\nSimple similarity matching with threshold=0.75:')
print(matches.to_df().sort_values('count',ascending=False))


# We can get better matches if we use group_threshold to control the
# minimimum similarity of pairs allowed within each group. This prevents strings
# from being united if doing so will also unite two strings that are unlikely
# to be a match.
# Note: Using a group threshold requires more processing time and memory than simple matching.
matches = embeddings.unite_similar(threshold=0.5,group_threshold=0.5)

print('\nSimilarity matching with group_threshold:')
print(matches.to_df().sort_values('count',ascending=False))

# This is probably the best we can do without any additional information.

# However, if we have additional information (either manually collected, or from
# a database like wikidata) we might know that "Disney World" is actually a
# common way to refer to "Walt Disney World Resort", not "Disney Worldwide
# Services", which is a separate organization used primarily for lobbying.
# We can improve our matching by specifying:
# - Sets of strings that should always be matched
# - Sets of strings that should never be matched

# Similarity-base matching that respects manual matches and separations
matches = embeddings.unite_similar(
                    threshold=0.5,
                    group_threshold=0.5,
                    always_match=[['Walt Disney World Resort','Disney World']],
                    never_match=[['Disney World','Disney Worldwide Services']],
                    )

print('\nSimilarity matching with manual matches and separations:')
print(matches.to_df().sort_values('count',ascending=False))

# If we had more manual data, we could extend both the always_match and never_match 
# lists with additional groups of strings that should be matched or separated.


# Finally, we might want to review the pairs that were matched or separated. 
# Setting return_united=True returns an additional list of pairs that were united
# to construct the final matc groups. Reviewing pairs with moderate similarity can
# be an effecient way to identify additional manual matches and separations.

matches,united_df = embeddings.unite_similar(
                    threshold=0.5,
                    group_threshold=0.5,
                    always_match=[['Walt Disney World Resort','Disney World']],
                    never_match=[['Disney World','Disney Worldwide Services']],
                    return_united=True
                    )

print('\nPairs that were united:')
print(united_df)