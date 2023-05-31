# import nltk
# from nltk.corpus import stopwords
# from pathlib import Path
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import sys
# import torch
# import argparse
# from datetime import datetime
import re
# import os
# from pathlib import Path
from collections import Counter
# from itertools import islice
import pandas as pd
import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib as mplt

# from .scoring import score_predicted, split_on_groups
from .matcher import Matcher


# Preprocessors

def simplify(s):
    """
    A basic case string simplication function. Strips case and punctuation.

    Parameters
    ----------
    s : str
        The string to be simplified.

    Returns
    -------
    str
        The simplified string.
    """
    s = s.lower()
    s = re.sub(' & ', ' and ', s)
    s = re.sub(r'(?<=\S)[\'’´\.](?=\S)', '', s)
    s = re.sub(r'[\s\.,:;/\'"`´‘’“”\(\)_—\-]+', ' ', s)
    s = s.strip()

    return s


def simplify_corp(s):
    """
    A simplification function for corporations and law firms.
    Strips:
        - case & puctation
        - 'the' prefix
        - common corporate suffixes, including 'holding co'

    Parameters
    ----------
    s : str
        The string to be simplified.

    Returns
    -------
    str
        The simplified string.
    """
    s = simplify(s)
    if s.startswith('the '):
        s = s[4:]

    s = re.sub(
        '( (group|holding(s)?( co)?|inc(orporated)?|ltd|l ?l? ?[cp]|co(rp(oration)?|mpany)?|s[ae]|plc))+$',
        '',
        s,
        count=1)

    return s


# def remove_stopwords(text):
#     """
#     Remove stopwords from a string.

#     Parameters
#     ----------
#     text : str
#         The string to have stopwords removed.

#     Returns
#     -------
#     str
#         The string with stopwords removed.
#     """
#     try:
#         stop_words = set(stopwords.words('english'))
#     except Exception:
#         nltk.download('stopwords')
#         stop_words = set(stopwords.words('english'))

#     return ' '.join([word for word in text.split()
#                     if word.lower() not in stop_words])


# Matcher Tools

# def plot(matchers, strings, matcher_names=None, ax=None):
#     """
#     Plots strings and their parent groups for multiple matchers as a graph, with
#     groups represented as nodes that connect strings.

#     Parameters
#     ----------
#     matchers : Matcher or list of Matcher
#         a matcher or list of matchers to plot
#     strings : str or list of str
#         a string or list of strings to plot (all connected strings will also be plotted)
#     matcher_names : list of str, optional
#         a list of strings to label matchers in the plot legend
#     ax : matplotlib.axes._subplots.AxesSubplot, optional
#         a matplotlib axis object to draw the plot on.

#     Returns
#     -------
#     matplotlib.axes._subplots.AxesSubplot
#         The matplotlib axis object with the plot.
#     """

#     if isinstance(matchers, Matcher):
#         matchers = [matchers]

#     if isinstance(strings, str):
#         strings = [strings]

#     if not matcher_names:
#         matcher_names = [f'matcher{i}' for i in range(len(matchers))]
#     elif not (len(matcher_names) == len(matchers)):
#         raise ValueError('matcher_names must be the same length as matchers')

#     def varname(x): return f'{x=}'.split('=')[0]

#     # First build graph representation of the parent groups
#     G = nx.Graph()
#     for i, matcher in enumerate(matchers):
#         m_groups = set(matcher[strings])
#         for g in m_groups:
#             group_node = f'{matcher_names[i]}: {g}'
#             string_nodes = matcher.groups[g]
#             G.add_nodes_from(string_nodes, type='string', color='w')
#             if len(string_nodes) > 1:
#                 G.add_nodes_from(
#                     [group_node],
#                     type='group',
#                     color=f'C{i}',
#                     label=group_node)
#                 nx.add_star(G, [group_node] + string_nodes, color=f'C{i}')

#     # Now plot graph components in a grid
#     components = sorted(nx.connected_components(G), key=len, reverse=True)

#     n_grid = int(np.ceil(np.sqrt(len(components))))
#     grid_xy = [(x, -y) for y in range(n_grid) for x in range(n_grid)]

#     if ax is None:
#         fig, ax = plt.subplots()

#     for i, component in enumerate(components):
#         G_sub = G.subgraph(component)

#         x0, y0 = grid_xy[i]

#         # Position nodes
#         if len(component) > 1:
#             pos = nx.random_layout(G_sub)
#             pos = nx.kamada_kawai_layout(G_sub, pos=pos, scale=0.25)
#             pos = {n: (x0 + x, y0 + y) for n, (x, y) in pos.items()}
#         else:
#             pos = {list(component)[0]: (x0, y0)}

#         edges = list(G_sub.edges(data=True))

#         edge_coord = [[pos[n0], pos[n1]] for n0, n1, d in edges]
#         edge_colors = [mplt.colors.to_rgba(d['color']) for n0, n1, d in edges]

#         lc = mplt.collections.LineCollection(
#             edge_coord, color=edge_colors, zorder=0)

#         ax.add_collection(lc)

#         for node, d in G_sub.nodes(data=True):
#             x, y = pos[node]
#             if d['type'] == 'group':
#                 ax.scatter(
#                     x,
#                     y,
#                     color=mplt.colors.to_rgb(
#                         d['color']),
#                     label=d['label'],
#                     s=50,
#                     zorder=2)
#             else:
#                 ax.scatter(x, y, color='w', s=200, zorder=1)
#                 ax.text(x, y, node, ha='center', va='center', zorder=3)

#     ax.axis('off')

#     legend_handles = [
#         mplt.lines.Line2D(
#             [],
#             [],
#             color=mplt.colors.to_rgb(f'C{i}'),
#             markersize=15,
#             marker='.',
#             label=f'{name}') for i,
#         name in enumerate(matcher_names)]
#     plt.legend(handles=legend_handles,
#                bbox_to_anchor=(0.5, -0.2), loc='lower center',
#                ncol=3, borderaxespad=0)

#     return ax


def from_df(
        df,
        match_format='detect',
        pair_columns=[
            'string0',
            'string1'],
    string_column='string',
    group_column='group',
        count_column='count'):
    """
    Construct a new matcher from a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    match_format : str, optional
        The format of the dataframe, by default "detect".
        It can be one of ['unmatched', 'groups', 'pairs', 'detect'].
    pair_columns : list of str, optional
        The columns names containing the string pairs, by default ['string0','string1'].
    string_column : str, optional
        The column name containing the strings, by default 'string'.
    group_column : str, optional
        The column name containing the groups, by default 'group'.
    count_column : str, optional
        The column name containing the counts, by default 'count'.

    Returns
    -------
    Matcher
        The constructed Matcher object.

    Raises
    ------
    ValueError
        If the input `match_format` is not one of ['unmatched', 'groups', 'pairs', 'detect'].
    ValueError
        If the `match_format` is 'detect' and the input dataframe format could not be inferred.

    Notes
    -----
    The function accepts two formats of the input dataframe:

        - "groups": The standard format for a matcher dataframe. It includes a
          string column, and a "group" column that contains group labels, and an
          optional "count" column. These three columns completely describe a
          matcher object, allowing lossless matcher -> dataframe -> matcher
          conversion (though the specific group labels in the dataframe will be
          ignored and rebuilt in the new matcher).

        - "pairs": The dataframe includes two string columns, and each row indicates
          a link between a pair of strings. A new matcher will be constructed by
          uniting each pair of strings.
    """

    if match_format not in ['unmatched', 'groups', 'pairs', 'detect']:
        raise ValueError(
            'match_format must be one of "unmatched", "groups", "pairs", or "detect"')

    # Create an empty matcher
    matcher = Matcher()

    if match_format == 'detect':
        if (string_column in df.columns):
            if group_column is None:
                match_format = 'unmatched'
            elif (group_column in df.columns):
                match_format = 'groups'
        elif set(df.columns) == set(pair_columns):
            match_format = 'pairs'

    if match_format == 'detect':
        raise ValueError('Could not infer valid dataframe format from input')

    if count_column in df.columns:
        counts = df[count_column].values
    else:
        counts = np.ones(len(df))

    if match_format == 'unmatched':
        strings = df[string_column].values

        # Build the matcher
        matcher.counts = Counter({s: int(c) for s, c in zip(strings, counts)})
        matcher.labels = {s: s for s in strings}
        matcher.groups = {s: [s] for s in strings}

    elif match_format == 'groups':

        strings = df[string_column].values
        group_ids = df[group_column].values

        # Sort by group and string count
        g_sort = np.lexsort((counts, group_ids))
        group_ids = group_ids[g_sort]
        strings = strings[g_sort]
        counts = counts[g_sort]

        # Identify group boundaries and split locations
        split_locs = np.nonzero(group_ids[1:] != group_ids[:-1])[0] + 1

        # Get grouped strings as separate arrays
        groups = np.split(strings, split_locs)

        # Build the matcher
        matcher.counts = Counter({s: int(c) for s, c in zip(strings, counts)})
        matcher.labels = {s: g[-1] for g in groups for s in g}
        matcher.groups = {g[-1]: list(g) for g in groups}

    elif match_format == 'pairs':
        # TODO: Allow pairs data to use counts
        for pair_column in pair_columns:
            matcher.add_strings(df[pair_column].values, inplace=True)

        # There are several ways to unite pairs
        # Guessing it is most efficient to "group by" one of the string columns
        groups = {s: pair[1] for pair in df[pair_columns].values for s in pair}

        matcher.unite(groups, inplace=True)

    return matcher


def read_csv(
        filename,
        match_format='detect',
        pair_columns=[
            'string0',
            'string1'],
    string_column='string',
    group_column='group',
    count_column='count',
        **pandas_args):
    """
    Read a csv file and construct a new matcher.

    Parameters
    ----------
    filename : str
        The path to the csv file.
    match_format : str, optional (default='detect')
        One of "unmatched", "groups", "pairs", or "detect".
    pair_columns : list of str, optional (default=['string0', 'string1'])
        Two string columns to use if match_format='pairs'.
    string_column : str, optional (default='string')
        Column name for string values in match_format='unmatched' or 'groups'.
    group_column : str, optional (default='group')
        Column name for group values in match_format='groups'.
    count_column : str, optional (default='count')
        Column name for count values in match_format='unmatched' or 'groups'.
    **pandas_args : optional
        Optional arguments to pass to `pandas.read_csv`.

    Returns
    -------
    Matcher
        A new matcher built from the csv file.
    """
    df = pd.read_csv(filename, **pandas_args, na_filter=False)
    df = df.astype(str)

    return from_df(df, match_format=match_format, pair_columns=pair_columns,
                   string_column=string_column, group_column=group_column,
                   count_column=count_column)
