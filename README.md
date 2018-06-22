# nama
_super fast fuzzy name matching using pytorch and skearn_

Nama solves the problem of merging datasets by name (particularly company names) when the names might not be represented identically. There are many measures of text similarity between strings, but when the number of names is large (say, a few hundred thousand or more in each list), pairwise comparison takes a very long time. Nama uses multiple passes to efficiently match names:
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
