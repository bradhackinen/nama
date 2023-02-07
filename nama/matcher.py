import os
from pathlib import Path
from collections import Counter, defaultdict
from itertools import islice
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mplt

MAX_STR = 50


class Matcher():
    """A class for grouping strings based on set membership.  Supports splitting and uniting of groups."""

    def __init__(self, strings=None):
        """
        Initialize Matcher object.

        Parameters
        ----------
        strings : list, optional
            List of strings to add to the matcher, by default None
        """
        self.counts = Counter()
        self.labels = {}
        self.groups = {}

        if strings is not None:
            self.add_strings(strings, inplace=True)

    def __len__(self):
        """Return the number of strings in the matcher."""
        return len(self.labels)

    def __repr__(self):
        """Return a string representation of the Matcher object."""
        return f'<nama.Matcher containing {len(self)} strings in {len(self.groups)} groups>'

    def __str__(self):
        """Return a string representation of the groups of a Matcher object."""
        output = self.__repr__()
        remaining = MAX_STR
        for group in self.groups.values():
            for s in group:
                if remaining:
                    output += '\n' + s
                    remaining -= 1
                else:
                    output += f'...\n(Output truncated at {MAX_STR} strings)'
                    return output

            output += '\n'

        return output

    def __contains__(self, s):
        """Return True if string is in the matcher, False otherwise."""
        return s in self.labels

    def __getitem__(self, strings):
        """Return the group label for a single string or a list of strings."""
        if isinstance(strings, str):
            return self.labels[strings]
        else:
            return [self.labels[s] for s in strings]

    def __add__(self, matcher):
        """Add two matchers together and return the result."""
        result = self.add_strings(matcher)
        result.unite(matcher, inplace=True)

        return result

    def items(self):
        """Return an iterator of strings and their group labels."""
        for i, g in self.labels.items():
            yield i, g

    def copy(self):
        """Return a copy of the Matcher object."""
        new_matcher = Matcher()
        new_matcher.counts = self.counts.copy()
        new_matcher.labels = self.labels.copy()
        new_matcher.groups = self.groups.copy()

        return new_matcher

    def strings(self):
        """Return a list of strings in the matcher. Order is not guaranteed."""
        return list(self.labels.keys())

    def matches(self, string):
        """Return the group of strings that match the given string."""
        return self.groups[self.labels[string]]

    def add_strings(self, arg, inplace=False):
        """Add new strings to the matcher.

        Parameters
        ----------
        arg : str, Counter, Matcher, Iterable
            String or group of strings to add to the matcher
        inplace : bool, optional
            If True, add strings to the existing Matcher object, by default False

        Returns
        -------
        Matcher
            The updated Matcher object
        """
        if isinstance(arg, str):
            counts = {arg: 1}

        elif isinstance(arg, Counter):
            counts = arg

        elif isinstance(arg, Matcher):
            counts = arg.counts

        elif hasattr(arg, '__next__') or hasattr(arg, '__iter__'):
            counts = Counter(arg)

        if not inplace:
            self = self.copy()

        for s in counts.keys():
            if s not in self.labels:
                self.labels[s] = s
                self.groups[s] = [s]

        self.counts += counts

        return self

    def drop(self, strings, inplace=False):
        """Remove strings from the matcher.

        Parameters
        ----------
        strings : list or str
            String or list of strings to remove from the matcher
        inplace : bool, optional
            If True, remove strings from the existing Matcher object, by default False

        Returns
        -------
        Matcher
            The updated Matcher object
        """
        if isinstance(strings, str):
            strings = [strings]

        strings = set(strings)

        if not inplace:
            self = self.copy()

        # Remove strings from their groups
        affected_group_labels = {self[s] for s in strings}
        for old_label in affected_group_labels:
            old_group = self.groups[old_label]
            new_group = [s for s in old_group if s not in strings]

            if new_group:
                counts = self.counts
                new_label = min((-counts[s], s) for s in new_group)[1]

                if new_label != old_label:
                    del self.groups[old_label]

                self.groups[new_label] = new_group

                for s in new_group:
                    self.labels[s] = new_label
            else:
                del self.groups[old_label]

        # Remove strings from counts and labels
        for s in strings:
            del self.counts[s]
            del self.labels[s]

        return self

    def keep(self, strings, inplace=False):
        """Drop all strings from the matcher except the passed strings.

        Parameters
        ----------
        strings : list
            List of strings to keep in the matcher
        inplace : bool, optional
            If True, drop strings from the existing Matcher object, by default False

        Returns
        -------
        Matcher
            The updated Matcher object
        """
        strings = set(strings)

        to_drop = [s for s in self.strings() if s not in strings]

        return self.drop(to_drop, inplace=inplace)

    def _unite_strings(self, strings):
        """
        Unite strings in the matcher without checking argument type.
        Intended as a low-level function called by self.unite()

        Parameters
        ----------
        strings : list
            List of strings to unite in the matcher

        Returns
        -------
        None
        """
        strings = {s for s in strings if s in self.labels}

        if len(strings) > 1:

            # Identify groups that will be united
            old_labels = set(self[strings])

            # Only need to do the merge if the strings span multiple groups
            if len(old_labels) > 1:

                # Identify the new group label
                counts = self.counts
                new_label = min((-counts[s], s) for s in old_labels)[1]

                # Identify the groups which need to be modified
                old_labels.remove(new_label)

                for old_label in old_labels:
                    # Update the string group labels
                    for s in self.groups[old_label]:
                        self.labels[s] = new_label

                    # Update group dict
                    self.groups[new_label] = self.groups[new_label] + \
                        self.groups[old_label]
                    del self.groups[old_label]

    def unite(self, arg, inplace=False, **kwargs):
        """
        Merge groups containing the passed strings. Groups can be passed as:
        - A list of strings to unite
        - A nested list to unite each set of strings
        - A dictionary mapping strings to labels to unite by label
        - A function mapping strings to labels to unite by label
        - A Matcher instance to unite by Matcher groups

        Parameters
        ----------
        arg : list, dict, function or Matcher instance
            Argument representing the strings or labels to merge.
        inplace : bool, optional
            Whether to perform the operation in place or return a new Matcher.
        kwargs : dict, optional
            Additional arguments to be passed to predict_matcher method if arg
            is a similarity model with a predict_matcher method.

        Returns
        -------
        Matcher
            The updated Matcher object. If `inplace` is True, the updated object
            is returned, else a new Matcher object with the updates is returned.
        """

        if not inplace:
            self = self.copy()

        if isinstance(arg, str):
            raise ValueError('Cannot unite a single string')

        elif isinstance(arg, Matcher):
            self.unite(arg.groups.values(), inplace=True)

        elif hasattr(arg, 'predict_matcher'):
            # Unite can accept a similarity model if it has a predict_matcher
            # method
            self.unite(arg.predict_matcher(self, **kwargs))

        elif callable(arg):
            # Assume arg is a mapping from strings to labels and unite by label
            groups = {s: arg(s) for s in self.strings()}
            self.unite(groups, inplace=True)

        elif isinstance(arg, dict):
            # Assume arg is a mapping from strings to labels and unite by label
            # groups = {label:[] for label in arg.values()}
            groups = defaultdict(list)
            for string, label in arg.items():
                groups[label].append(string)

            for group in groups.values():
                self._unite_strings(group)

        elif hasattr(arg, '__next__'):
            # Assume arg is an iterator of groups to unite
            # (This needs to be checked early to avoid consuming the first group)
            for group in arg:
                self._unite_strings(group)

        elif all(isinstance(s, str) for s in arg):
            # Main case: Unite group of strings
            self._unite_strings(arg)

        elif hasattr(arg, '__iter__'):
            # Assume arg is an iterable of groups to unite
            for group in arg:
                self._unite_strings(group)

        else:
            raise ValueError('Unknown input type')

        if not inplace:
            return self

    def split(self, strings, inplace=False):
        """
        Split strings into singleton groups. Strings can be passed as:
        - A single string to isolate into a singleton group
        - A list or iterator of strings to split

        Parameters
        ----------
        strings : str or list of str
            The string(s) to split into singleton groups.
        inplace : bool, optional
            Whether to perform the operation in place or return a new Matcher.

        Returns
        -------
        Matcher
            The updated Matcher object. If `inplace` is True, the updated object
            is returned, else a new Matcher object with the updates is returned.
        """
        if not inplace:
            self = self.copy()

        if isinstance(strings, str):
            strings = [strings]

        strings = set(strings)

        # Remove strings from their groups
        affected_group_labels = {self[s] for s in strings}
        for old_label in affected_group_labels:
            old_group = self.groups[old_label]
            if len(old_group) > 1:
                new_group = [s for s in old_group if s not in strings]
                if new_group:
                    counts = self.counts
                    new_label = min((-counts[s], s) for s in new_group)[1]

                    if new_label != old_label:
                        del self.groups[old_label]

                    self.groups[new_label] = new_group

                    for s in new_group:
                        self.labels[s] = new_label

        # Update labels and add singleton groups
        for s in strings:
            self.labels[s] = s
            self.groups[s] = [s]

        return self

    def split_all(self, inplace=False):
        """
        Split all strings into singleton groups.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation in place or return a new Matcher.

        Returns
        -------
        Matcher
            The updated Matcher object. If `inplace` is True, the updated object
            is returned, else a new Matcher object with the updates is returned.
        """
        if not inplace:
            self = self.copy()

        self.labels = {s: s for s in self.strings()}
        self.groups = {s: [s] for s in self.strings()}

        return self

    def separate(
    self,
    strings,
    similarity_model,
    inplace=False,
    threshold=0,
     **kwargs):
        """
        Separate the strings in according to the prediction of the similarity_model.

        Parameters
        ----------
        strings: list
            List of strings to be separated.
        similarity_model: Model
            Model used to predict similarity between strings.
        inplace: bool, optional
            If True, the separation operation is performed in-place. Otherwise, a copy is created.
        threshold: float, optional
            Threshold value for prediction.
        kwargs: dict, optional
            Additional keyword arguments passed to the prediction function.

        Returns
        -------
        self: Matcher
            Returns the Matcher object after the separation operation.

        """
        if not inplace:
            self = self.copy()

        # Identify which groups contain the strings to separate
        group_map = defaultdict(list)
        for s in set(strings):
            group_map[self[s]].append(s)

        for g, g_sep in group_map.items():

            # If group contains strings to separate...
            if len(g_sep) > 1:
                group_strings = self.groups[g]

                # Split the group strings
                self.split(group_strings, inplace=True)

                # Re-unite with new prediction that enforces separation
                try:
                    embeddings = similarity_model[group_strings]
                except Exception as e:
                    print(f'{g=} {g_sep} {group_strings}')
                    raise e
                predicted = embeddings.predict(
                                threshold=threshold,
                                separate_strings=strings,
                                **kwargs)

                self.unite(predicted, inplace=True)

        return self

    # def refine(self,similarity_model)

    def top_scored_pairs_df(self, similarity_model,
                                n=10000, buffer_n=100000,
                                by_group=True,
                                sort_by=['impact', 'score'], ascending=False,
                                skip_pairs=None, **kwargs):
        """
        Return the DataFrame containing the n most important pairs of strings, according to the score generated by the `similarity_model`.

        Parameters
        ----------
        similarity_model: Model
            Model used to predict similarity between strings.
        n: int, optional
            Number of most important pairs to return. Default is 10000.
        buffer_n: int, optional
            Size of buffer to iterate through the scored pairs. Default is 100000.
        by_group: bool, optional
            If True, only the most important pair will be returned for each unique group combination.
        sort_by: list, optional
            A list of column names by which to sort the dataframe. Default is ['impact','score'].
        ascending: bool, optional
            Whether the sort order should be ascending or descending. Default is False.
        skip_pairs: list, optional
            List of string pairs to ignore when constructing the ranking.
            If by_group=True, any group combination represented in the skip_pairs list will be ignored
        kwargs: dict, optional
            Additional keyword arguments passed to the `iter_scored_pairs` function.

        Returns
        -------
        top_df: pandas.DataFrame
            The DataFrame containing the n most important pairs of strings.

        """

        top_df = pd.DataFrame(
    columns=[
        'string0',
        'string1',
        'group0',
        'group1',
        'impact',
        'score',
         'loss'])
        pair_iterator = similarity_model.iter_scored_pairs(self, **kwargs)

        def group_size(g):
            return len(self.groups[g])

        if skip_pairs is not None:
            if by_group:
                skip_pairs = {tuple(sorted([self[s0], self[s1]]))
                                    for s0, s1 in skip_pairs}
            else:
                skip_pairs = {tuple(sorted([s0, s1])) for s0, s1 in skip_pairs}

        while True:
            df = pd.DataFrame(islice(pair_iterator, buffer_n))

            if len(df):
                for i in 0, 1:
                    df[f'group{i}'] = [self[s] for s in df[f'string{i}']]
                df['impact'] = df['group0'].apply(
                    group_size) * df['group1'].apply(group_size)

                if by_group:
                    df['group_pair'] = [tuple(sorted([g0, g1])) for g0, g1 in df[[
                                              'group0', 'group1']].values]

                if skip_pairs:
                    if by_group:
                        df = df[~df['group_pair'].isin(skip_pairs)]
                    else:
                        string_pairs = [tuple(sorted([s0, s1])) for s0, s1 in df[[
                                              'string0', 'string1']].values]
                        df = df[~string_pairs.isin(skip_pairs)]

                if len(df):
                    top_df = pd.concat([top_df, df]) \
                                        .sort_values(sort_by, ascending=ascending)

                    if by_group:
                        top_df = top_df \
                                        .groupby('group_pair') \
                                        .first() \
                                        .reset_index()

                    top_df = top_df \
                                        .sort_values(sort_by, ascending=ascending) \
                                        .head(n)
            else:
                break

        if len(top_df) and by_group:
            top_df = top_df \
                        .drop('group_pair', axis=1) \
                        .reset_index()

        return top_df

    def reset_counts(self, inplace=False):
        """
        Reset the counts of strings in the Matcher object.

        Parameters
        ----------
        inplace: bool, optional
            If True, the operation is performed in-place. Otherwise, a copy is created.

        Returns
        -------
        self: Matcher
            Returns the Matcher object after the reset operation.

        """
        if not inplace:
            self = self.copy()

        self.counts = Counter(self.strings())

        return self

    def to_df(self, singletons=True, sort_groups=True):
        """
        Convert the matcher to a dataframe with string, count and group columns.

        Parameters
        ----------
        singletons: bool, optional
            If True, the resulting DataFrame will include singleton groups. Default is True.
        ...

        Returns
        -------
        df: pandas.DataFrame
            The resulting DataFrame.
            """
        strings = self.strings()

        if singletons:
            df = pd.DataFrame([(s, self.counts[s], self.labels[s]) for s in strings],
                                columns=['string', 'count', 'group'])
        else:
            df = pd.DataFrame([(s, self.counts[s], self.labels[s]) for s in strings
                                if len(self.groups[self[s]]) > 1],
                                columns=['string', 'count', 'group'])
        if sort_groups:
            df['group_count'] = df.groupby('group')['count'].transform('sum')
            df = df.sort_values(['group_count', 'group', 'count', 'string'], ascending=[
                                False, True, False, True])
            df = df.drop('group_count', axis=1)
            df = df.reset_index(drop=True)

        return df

    def to_csv(self, filename, singletons=True, **pandas_args):
        """
        Save the matcher as a csv file with string, count and group columns.

        Parameters
        ----------
        filename : str
            Path to file to save the data.
        singletons : bool, optional
            If True, include singleton groups in the saved file, by default True.
        pandas_args : dict
            Additional keyword arguments to pass to the pandas.DataFrame.to_csv method.
        """
        df = self.to_df(singletons=singletons)
        df.to_csv(filename, index=False, **pandas_args)

    def merge_dfs(self, left_df, right_df, how='inner',
                    on=None, left_on=None, right_on=None,
                    group_column_name='match_group', suffixes=('_x', '_y'),
                    **merge_args):
        """
        Replicated pandas.merge() functionality, except that dataframes are merged by match group instead of directly on the strings in the "on" columns.

        Parameters
        ----------
        left_df : pandas.DataFrame
            The left dataframe to merge.
        right_df : pandas.DataFrame
            The right dataframe to merge.
        how : str, optional
            How to merge the dataframes. Possible values are 'left', 'right', 'outer', 'inner', by default 'inner'.
        on : str, optional
            Columns in both left and right dataframes to merge on.
        left_on : str, optional
            Columns in the left dataframe to merge on.
        right_on : str, optional
            Columns in the right dataframe to merge on.
        group_column_name : str, optional
            Column name for the merged match group, by default 'match_group'.
        suffixes : tuple of str, optional
            Suffix to apply to overlapping column names in the left and right dataframes, by default ('_x','_y').
        **merge_args : dict
            Additional keyword arguments to pass to the pandas.DataFrame.merge method.

        Returns
        -------
        pandas.DataFrame
            The merged dataframe.

        Raises
        ------
        ValueError
            If 'on', 'left_on', and 'right_on' are all None.
        ValueError
            If `group_column_name` already exists in one of the dataframes.
        """

        if ((left_on is None) or (right_on is None)) and (on is None):
            raise ValueError('Must provide column(s) to merge on')

        left_df = left_df.copy()
        right_df = right_df.copy()

        if on is not None:
            left_on = on + suffixes[0]
            right_on = on + suffixes[1]

            left_df = left_df.rename(columns={on:left_on})
            right_df = right_df.rename(columns={on:right_on})

        group_map = lambda s: self[s] if s in self.labels else np.nan

        left_group = left_df[left_on].apply(group_map)
        right_group = right_df[right_on].apply(group_map)

        if group_column_name:
            if group_column_name in list(left_df.columns) + list(right_df.columns):
                raise ValueError('f{group_column_name=} already exists in one of the dataframes.')
            else:
                left_df[group_column_name] = left_group

        merged_df = pd.merge(left_df,right_df,left_on=left_group,right_on=right_group,how=how,suffixes=suffixes,**merge_args)

        merged_df = merged_df[[c for c in merged_df.columns if c in list(left_df.columns) + list(right_df.columns)]]

        return merged_df