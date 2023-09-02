from collections import Counter, defaultdict
from itertools import islice
import pandas as pd
import numpy as np

MAX_STR = 50


class MatchData():
    """A class for grouping strings based on set membership.  Supports splitting and uniting of groups."""

    def __init__(self, strings=None):
        """
        Initialize MatchData object.

        Parameters
        ----------
        strings : list, optional
            List of strings to add to the match groups object, by default None
        """
        self.counts = Counter()
        self.labels = {}
        self.groups = {}

        if strings is not None:
            self.add_strings(strings, inplace=True)

    def __len__(self):
        """Return the number of strings in the match groups object."""
        return len(self.labels)

    def __repr__(self):
        """Return a string representation of the MatchData object."""
        return f'<nama.MatchData containing {len(self)} strings in {len(self.groups)} groups>'

    def __str__(self):
        """Return a string representation of the groups of a MatchData object."""
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
        """Return True if string is in the match groups object, False otherwise."""
        return s in self.labels

    def __getitem__(self, strings):
        """Return the group label for a single string or a list of strings."""
        if isinstance(strings, str):
            return self.labels[strings]
        else:
            return [self.labels[s] for s in strings]

    def __add__(self, matches):
        """Add two match groups objects together and return the result."""
        result = self.add_strings(matches)
        result.unite(matches, inplace=True)

        return result

    def items(self):
        """Return an iterator of strings and their group labels."""
        for i, g in self.labels.items():
            yield i, g

    def copy(self):
        """Return a copy of the MatchData object."""
        new_matches = MatchData()
        new_matches.counts = self.counts.copy()
        new_matches.labels = self.labels.copy()
        new_matches.groups = self.groups.copy()

        return new_matches

    def strings(self):
        """Return a list of strings in the match groups object. Order is not guaranteed."""
        return list(self.labels.keys())

    def matches(self, string):
        """Return the group of strings that match the given string."""
        return self.groups[self.labels[string]]

    def add_strings(self, arg, inplace=False):
        """Add new strings to the match groups object.

        Parameters
        ----------
        arg : str, Counter, MatchData, Iterable
            String or group of strings to add to the match groups object
        inplace : bool, optional
            If True, add strings to the existing MatchData object, by default False

        Returns
        -------
        MatchData
            The updated MatchData object
        """
        if isinstance(arg, str):
            counts = {arg: 1}

        elif isinstance(arg, Counter):
            counts = arg

        elif isinstance(arg, MatchData):
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
        """Remove strings from the match groups object.

        Parameters
        ----------
        strings : list or str
            String or list of strings to remove from the match groups object
        inplace : bool, optional
            If True, remove strings from the existing MatchData object, by default False

        Returns
        -------
        MatchData
            The updated MatchData object
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
        """Drop all strings from the match groups object except the passed strings.

        Parameters
        ----------
        strings : list
            List of strings to keep in the match groups object
        inplace : bool, optional
            If True, drop strings from the existing MatchData object, by default False

        Returns
        -------
        MatchData
            The updated MatchData object
        """
        strings = set(strings)

        to_drop = [s for s in self.strings() if s not in strings]

        return self.drop(to_drop, inplace=inplace)

    def _unite_strings(self, strings):
        """
        Unite strings in the match groups object without checking argument type.
        Intended as a low-level function called by self.unite()

        Parameters
        ----------
        strings : list
            List of strings to unite in the match groups object

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
        - A MatchData instance to unite by MatchData groups

        Parameters
        ----------
        arg : list, dict, function or MatchData instance
            Argument representing the strings or labels to merge.
        inplace : bool, optional
            Whether to perform the operation in place or return a new MatchData.
        kwargs : dict, optional
            Additional arguments to be passed to unite_similar method if arg
            is a similarity model with a unite_similar method.

        Returns
        -------
        MatchData
            The updated MatchData object. If `inplace` is True, the updated object
            is returned, else a new MatchData object with the updates is returned.
        """

        if not inplace:
            self = self.copy()

        if isinstance(arg, str):
            raise ValueError('Cannot unite a single string')

        elif isinstance(arg, MatchData):
            self.unite(arg.groups.values(), inplace=True)

        elif hasattr(arg, 'unite_similar'):
            # Unite can accept a similarity model if it has a unite_similar
            # method
            self.unite(arg.unite_similar(self, **kwargs))

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
            Whether to perform the operation in place or return a new MatchData.

        Returns
        -------
        MatchData
            The updated MatchData object. If `inplace` is True, the updated object
            is returned, else a new MatchData object with the updates is returned.
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
            Whether to perform the operation in place or return a new MatchData.

        Returns
        -------
        MatchData
            The updated MatchData object. If `inplace` is True, the updated object
            is returned, else a new MatchData object with the updates is returned.
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
        self: MatchData
            Returns the MatchData object after the separation operation.

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
        Reset the counts of strings in the MatchData object.

        Parameters
        ----------
        inplace: bool, optional
            If True, the operation is performed in-place. Otherwise, a copy is created.

        Returns
        -------
        self: MatchData
            Returns the MatchData object after the reset operation.

        """
        if not inplace:
            self = self.copy()

        self.counts = Counter(self.strings())

        return self

    def to_df(self, singletons=True, sort_groups=True):
        """
        Convert the match groups object to a dataframe with string, count and group columns.

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
        Save the match groups object as a csv file with string, count and group columns.

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
    Construct a new match groups object from a pandas DataFrame.

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
    MatchData
        The constructed MatchData object.

    Raises
    ------
    ValueError
        If the input `match_format` is not one of ['unmatched', 'groups', 'pairs', 'detect'].
    ValueError
        If the `match_format` is 'detect' and the input dataframe format could not be inferred.

    Notes
    -----
    The function accepts two formats of the input dataframe:

        - "groups": The standard format for a match groups object dataframe. It includes a
          string column, and a "group" column that contains group labels, and an
          optional "count" column. These three columns completely describe a
          match groups object, allowing lossless match groups object -> dataframe -> match groups object
          conversion (though the specific group labels in the dataframe will be
          ignored and rebuilt in the new match groups object).

        - "pairs": The dataframe includes two string columns, and each row indicates
          a link between a pair of strings. A new match groups object will be constructed by
          uniting each pair of strings.
    """

    if match_format not in ['unmatched', 'groups', 'pairs', 'detect']:
        raise ValueError(
            'match_format must be one of "unmatched", "groups", "pairs", or "detect"')

    # Create an empty match groups object
    matches = MatchData()

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

        # Build the match groups object
        matches.counts = Counter({s: int(c) for s, c in zip(strings, counts)})
        matches.labels = {s: s for s in strings}
        matches.groups = {s: [s] for s in strings}

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

        # Build the match groups object
        matches.counts = Counter({s: int(c) for s, c in zip(strings, counts)})
        matches.labels = {s: g[-1] for g in groups for s in g}
        matches.groups = {g[-1]: list(g) for g in groups}

    elif match_format == 'pairs':
        # TODO: Allow pairs data to use counts
        for pair_column in pair_columns:
            matches.add_strings(df[pair_column].values, inplace=True)

        # There are several ways to unite pairs
        # Guessing it is most efficient to "group by" one of the string columns
        groups = {s: pair[1] for pair in df[pair_columns].values for s in pair}

        matches.unite(groups, inplace=True)

    return matches


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
    Read a csv file and construct a new match groups object.

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
    MatchData
        A new match groups object built from the csv file.
    """
    df = pd.read_csv(filename, **pandas_args, na_filter=False)
    df = df.astype(str)

    return from_df(df, match_format=match_format, pair_columns=pair_columns,
                   string_column=string_column, group_column=group_column,
                   count_column=count_column)