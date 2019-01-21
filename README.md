# nama
_Fast, flexible name matching for large datasets_

`nama` solves the problem of merging datasets by name (particularly company names) when the names might not be represented identically. It is particularly useful when you have thousands or millions of names. In this case manually finding correct matches is almost impossible, and even algorithms that iterate over all the possible pairs of names will be very slow. `nama` will probably not completely replace the need to manually review pairs of names, but it makes the task it much more efficient by quickly generating potential matches and providing tools to prioritize matches for review and adjustment.


Key Features:
- Match by manually coded pairs, hash collisions, or a novel algorithm for very fast, trainable fuzzy matching
- Combine multiple matching techniques in a _match graph_
- Pairwise linking via a `merge` function for pandas DataFrames
- Name clustering for linking many datasets
- Interrogate the match graph to see how each match was found and highlight important links

# Quick start
The following code demonstrates how to match strings using hash collisions and similarity matching
```python
import os
import pandas as pd
from nama.matcher import Matcher
from nama.hashes import corpHash
from nama.similarity import loadSimilarityModel

from nama.defaults import *


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
similarityModel = loadSimilarityModel(os.path.join(modelDir,'demoModel.bin'),cuda=False)

# Preview similar matches without applying them
# (Useful for choosing a cutoff or manually reviewing each one)
matcher.suggestMatches(similarityModel,min_score=0)

# Add all similarity matches with score >= 0.5
# (Alternaticely, use the matcher.applyMatchDF method to add selected matches)
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
```

# Similarity scoring and training
Similarity models are trained on a match graph. `nama` trains a character-level recurrent neural network (RNN) to produce a vector for each string such that:
- Strings from the same component produce similar vectors (i.e., near in Euclidian space)
- Strings from different components are spread apart

When the `suggestMatches` or `matchSimilar` functions are called, `nama` generates the vectors for all strings and uses an efficient nearest neighbours-algorithm from `scikit-learn` to find close matches. The similarity score reflects the distance of the vectors in Euclidian space.

Because the similarity model finds it 'easier' to generate similar output from similar strings, misclassification is informative. String pairs with very high scores that aren't in the same component often indicate spelling mistakes or other errors in the match graph, and string pairs with very low score within a component suggest that two very different appearing names have been linked, which might be worth reviewing.

The following code demonstrates how to train a new similarity model on a small number of strings linked by hash collisions. Note that for large datasets (millions of strings), training a new model can take several hours.
```python
import nama
from nama.matcher import Matcher
from nama.similarity import SimilarityModel

from nama.defaults import *


# Initialize the matcher
matcher = Matcher(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO','ABC Inc.','XYZ Co.'])

# Add some corpHash matches
matcher.matchHash(nama.hashes.corpHash)


# Initalize a new, untrained similarity model
similarityModel = SimilarityModel(cuda=True,d=100,d_recurrent=100,recurrent_layers=2,bidirectional=True)


# Observe that untrained suggestions are poor quality (though not entirely useless - neat!)
matcher.suggestMatches(similarityModel)


# Train model using existing matches
similarityModel.train(matcher,epochs=1)

# Suggestions are now much better
matcher.suggestMatches(similarityModel,min_score=0)

# Save similarity model
similarityModel.save(os.path.join(modelDir,'demoModel.bin'))


# Too much training on a small set of strings can lead to over-fitting
similarityModel.train(matcher,epochs=3)

# --> All suggestions now have very low scores
matcher.suggestMatches(similarityModel,min_score=0)

```


# Installation
## Requirements
- Python 3
- networkx
- pandas
- numpy
- matplotlib
- PyTorch
- sci-kit learn
