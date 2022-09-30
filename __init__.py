import os
from pathlib import Path
from collections import Counter, defaultdict
from itertools import islice
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mplt

root_dir = Path(os.path.dirname(os.path.abspath(__file__)))

MAX_STR = 50


class Matcher():
    def __init__(self,strings=None):

        self.counts = Counter()
        self.labels = {}
        self.groups = {}

        if strings is not None:
            self.add_strings(strings,inplace=True)

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f'<nama.Matcher containing {len(self)} strings in {len(self.groups)} groups>'

    def __str__(self):
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

    def __contains__(self,s):
        return s in self.labels

    def __getitem__(self,strings):
        if isinstance(strings,str):
            return self.labels[strings]
        else:
            return [self.labels[s] for s in strings]

    def __add__(self,matcher):
        result = self.add_strings(matcher)
        result.unite(matcher,inplace=True)

        return result

    def items(self):
        for i,g in self.labels.items():
            yield i,g

    def copy(self):
        new_matcher = Matcher()
        new_matcher.counts = self.counts.copy()
        new_matcher.labels = self.labels.copy()
        new_matcher.groups = self.groups.copy()

        return new_matcher

    def strings(self):
        """List strings in the matcher. Order is not guarunteed."""
        return list(self.labels.keys())

    def matches(self,string):
        return self.groups[self.labels[string]]

    def add_strings(self,arg,inplace=False):
        """Add new strings to the matcher"""
        if isinstance(arg,str):
            counts = {arg:1}

        elif isinstance(arg,Counter):
            counts = arg

        elif isinstance(arg,Matcher):
            counts = arg.counts

        elif hasattr(arg,'__next__') or hasattr(arg,'__iter__'):
            counts = Counter(arg)

        if not inplace:
            self = self.copy()

        for s in counts.keys():
            if s not in self.labels:
                self.labels[s] = s
                self.groups[s] = [s]

        self.counts += counts

        return self

    def drop(self,strings,inplace=False):
        """Remove strings from the matcher"""
        if isinstance(strings,str):
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
                new_label = min((-counts[s],s) for s in new_group)[1]

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

    def keep(self,strings,inplace=False):
        """Drop all strings from the matcher except the passed strings"""
        strings = set(strings)

        to_drop = [s for s in self.strings() if s not in strings]

        return self.drop(to_drop,inplace=inplace)

    def _unite_strings(self,strings):
        """
        Unite strings in the matcher without checking argument type
        (Intended as a low-level function called by self.unite())
        """
        strings = {s for s in strings if s in self.labels}

        if len(strings) > 1:

            # Identify groups that will be united
            old_labels = set(self[strings])

            # Only need to do the merge if the strings span multiple groups
            if len(old_labels) > 1:

                # Identify the new group label
                counts = self.counts
                new_label = min((-counts[s],s) for s in old_labels)[1]

                # Identify the groups which need to be modified
                old_labels.remove(new_label)

                for old_label in old_labels:
                    # Update the string group labels
                    for s in self.groups[old_label]:
                        self.labels[s] = new_label

                    # Update group dict
                    self.groups[new_label] = self.groups[new_label] + self.groups[old_label]
                    del self.groups[old_label]

    def unite(self,arg,inplace=False,**kwargs):
        """
        Merge groups containing the passed strings. Can pass:
         - A list of strings to unite
         - A nested list to unite each set of strings
         - A dictionary mapping strings to labels to unite by label
         - A function mapping strings to labels to unite by label
         - A matcher instance to unite by matcher groups
        """
        if not inplace:
            self = self.copy()

        if isinstance(arg,str):
            raise ValueError('Cannot unite a single string')

        elif isinstance(arg,Matcher):
            self.unite(arg.groups.values(),inplace=True)

        elif hasattr(arg,'predict_matcher'):
            # Unite can accept a similarity model if it has a predict_matcher method
            self.unite(arg.predict_matcher(self,**kwargs))

        elif callable(arg):
            # Assume arg is a mapping from strings to labels and unite by label
            groups = {s:arg(s) for s in self.strings()}
            self.unite(groups,inplace=True)

        elif isinstance(arg,dict):
            # Assume arg is a mapping from strings to labels and unite by label
            # groups = {label:[] for label in arg.values()}
            groups = defaultdict(list)
            for string,label in arg.items():
                groups[label].append(string)

            for group in groups.values():
                self._unite_strings(group)

        elif hasattr(arg,'__next__'):
            # Assume arg is an iterator of groups to unite
            # (This needs to be checked early to avoid consuming the first group)
            for group in arg:
                self._unite_strings(group)

        elif all(isinstance(s,str) for s in arg):
            # Main case: Unite group of strings
            self._unite_strings(arg)

        elif hasattr(arg,'__iter__'):
            # Assume arg is an iterable of groups to unite
            for group in arg:
                self._unite_strings(group)

        else:
            raise ValueError('Unknown input type')

        if not inplace:
            return self

    def split(self,strings,inplace=False):
        """
        Split strings into singleton groups.  Can pass:
         - A single string to isolate into a singleton group
         - A list or iterator of strings to split
        """
        if not inplace:
            self = self.copy()

        if isinstance(strings,str):
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
                    new_label = min((-counts[s],s) for s in new_group)[1]

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

    def split_all(self,inplace=False):
        if not inplace:
            self = self.copy()

        self.labels = {s:s for s in self.strings()}
        self.groups = {s:[s] for s in self.strings()}

        return self

    def separate(self,strings,similarity_model,inplace=False,threshold=0,**kwargs):
        if not inplace:
            self = self.copy()

        # Identify which groups contain the strings to separate
        group_map = defaultdict(list)
        for s in set(strings):
            group_map[self[s]].append(s)

        for g,g_sep in group_map.items():

            # If group contains strings to separate...
            if len(g_sep) > 1:
                group_strings = self.groups[g]

                # Split the group strings
                self.split(group_strings,inplace=True)

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

                self.unite(predicted,inplace=True)

        return self

    # def refine(self,similarity_model)

    def top_scored_pairs_df(self,similarity_model,
                                n=10000,buffer_n=100000,
                                by_group=True,
                                sort_by=['impact','score'],ascending=False,
                                skip_pairs=None,**kwargs):
        """
        Use the similarity model to identify the n most important pairs,
        according to the is_match, sort_by, and ascending arguments.

        If by_group=True, only the most important pair will be returned for each
        unique group combination.

        skip_pairs should be a list of string pairs to ignore when constructing
        the ranking. If by_group=True, any group combination represented in the
        skip_pairs list will be ignored.
        """

        top_df = pd.DataFrame(columns=['string0','string1','group0','group1','impact','score','loss'])
        pair_iterator = similarity_model.iter_scored_pairs(self,**kwargs)

        def group_size(g):
            return len(self.groups[g])

        if skip_pairs is not None:
            if by_group:
                skip_pairs = {tuple(sorted([self[s0],self[s1]])) for s0,s1 in skip_pairs}
            else:
                skip_pairs = {tuple(sorted([s0,s1])) for s0,s1 in skip_pairs}

        while True:
            df = pd.DataFrame(islice(pair_iterator,buffer_n))

            if len(df):
                for i in 0,1:
                    df[f'group{i}'] = [self[s] for s in df[f'string{i}']]
                df['impact'] = df['group0'].apply(group_size) * df['group1'].apply(group_size)

                if by_group:
                    df['group_pair'] = [tuple(sorted([g0,g1])) for g0,g1 in df[['group0','group1']].values]

                if skip_pairs:
                    if by_group:
                        df = df[~df['group_pair'].isin(skip_pairs)]
                    else:
                        string_pairs = [tuple(sorted([s0,s1])) for s0,s1 in df[['string0','string1']].values]
                        df = df[~string_pairs.isin(skip_pairs)]

                if len(df):
                    top_df = pd.concat([top_df,df]) \
                                        .sort_values(sort_by,ascending=ascending)

                    if by_group:
                        top_df = top_df \
                                        .groupby('group_pair') \
                                        .first() \
                                        .reset_index()

                    top_df = top_df \
                                        .sort_values(sort_by,ascending=ascending) \
                                        .head(n)
            else:
                break

        if len(top_df) and by_group:
            top_df = top_df \
                        .drop('group_pair',axis=1) \
                        .reset_index()

        return top_df

    def reset_counts(self,inplace=False):
        if not inplace:
            self = self.copy()

        self.counts = Counter(self.strings())

        return self

    def to_df(self,singletons=True,sort_groups=True):
        """
        Convert the matcher to a dataframe with string,count and group columns.
        """
        strings = self.strings()

        if singletons:
            df = pd.DataFrame([(s,self.counts[s],self.labels[s]) for s in strings],
                                columns=['string','count','group'])
        else:
            df = pd.DataFrame([(s,self.counts[s],self.labels[s]) for s in strings
                                if len(self.groups[self[s]]) > 1],
                                columns=['string','count','group'])
        if sort_groups:
            df['group_count'] = df.groupby('group')['count'].transform('sum')
            df = df.sort_values(['group_count','group','count','string'],ascending=[False,True,False,True])
            df = df.drop('group_count',axis=1)
            df = df.reset_index(drop=True)

        return df

    def to_csv(self,filename,singletons=True,**pandas_args):
        """
        Save the matcher as a csv file with string, count and group columns.
        """
        df = self.to_df(singletons=singletons)
        df.to_csv(filename,index=False,**pandas_args)

    def merge_dfs(self,left_df,right_df,how='inner',
                    on=None,left_on=None,right_on=None,
                    group_column_name='match_group',suffixes=('_x','_y'),
                    **merge_args):
        """
        Replicated pandas.merge() functionality, except that dataframes are
        merged by match group instead of directly on the strings in the "on"
        columns.

        "on" columns are assumed to contain strings which appear in the matcher.
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


def from_df(df,match_format='detect',pair_columns=['string0','string1'],
            string_column='string',group_column='group',count_column='count'):
    """
    Construct a new matcher from a dataframe.

    Accepts two formats:
        - "groups": The standard format for a matcher dataframe. Includes a
          string column, and a "group" column that contains group labels, and an
          optional "count" column. These three columns completely describe a
          matcher object, allowing lossless matcher --> dataframe --> matcher
          conversion (though the specific group labels in the dataframe will be
          ignored and rebuilt in the new matcher).

        - "pairs": Dataframe includes two string columns, and each row indicates
          a link between a pair of strings. A new matcher will be constructed by
          uniting each pair of strings.
    """

    if match_format not in ['unmatched','groups','pairs','detect']:
        raise ValueError('match_format must be one of "unmatched", "groups", "pairs", or "detect"')

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
        matcher.counts = Counter({s:int(c) for s,c in zip(strings,counts)})
        matcher.labels = {s:s for s in strings}
        matcher.groups = {s:[s] for s in strings}

    elif match_format == 'groups':

        strings = df[string_column].values
        group_ids = df[group_column].values

        # Sort by group and string count
        g_sort = np.lexsort((counts,group_ids))
        group_ids = group_ids[g_sort]
        strings = strings[g_sort]
        counts = counts[g_sort]

        # Identify group boundaries and split locations
        split_locs = np.nonzero(group_ids[1:] != group_ids[:-1])[0] + 1

        # Get grouped strings as separate arrays
        groups = np.split(strings,split_locs)

        # Build the matcher
        matcher.counts = Counter({s:int(c) for s,c in zip(strings,counts)})
        matcher.labels = {s:g[-1] for g in groups for s in g}
        matcher.groups = {g[-1]:list(g) for g in groups}

    elif match_format == 'pairs':
        # TODO: Allow pairs data to use counts
        for pair_column in pair_columns:
            matcher.add_strings(df[pair_column].values,inplace=True)

        # There are several ways to unite pairs
        # Guessing it is most efficient to "group by" one of the string columns
        groups = {s:pair[1] for pair in df[pair_columns].values for s in pair}

        matcher.unite(groups,inplace=True)

    return matcher


def read_csv(filename,match_format='detect',pair_columns=['string0','string1'],
                string_column='string',group_column='group',count_column='count',
                **pandas_args):

    df = pd.read_csv(filename,**pandas_args,na_filter=False)

    return from_df(df,match_format=match_format,pair_columns=pair_columns,
                    string_column=string_column,group_column=group_column,
                    count_column=count_column)


def plot(matchers,strings,matcher_names=None,ax=None):
    """
    Plots strings and their parent groups for multiple matchers as a graph, with
    groups represented as nodes that connect strings

    Arguments:
    matchers -- a matcher or list of matchers to plot
    strings  -- a string or list of strings to plot (all connected strings will
                also be plotted)
    matcher_names -- (optional) a list of strings to label matchers in the plot
                     legend
    ax -- (optional) a matplotlib axis object to draw the plot on.
    """
    if isinstance(matchers,Matcher):
        matchers = [matchers]

    if isinstance(strings,str):
        strings = [strings]

    if not matcher_names:
        matcher_names = [f'matcher{i}' for i in range(len(matchers))]
    elif not (len(matcher_names) == len(matchers)):
        raise ValueError('matcher_names must be the same length as matchers')

    varname = lambda x: f'{x=}'.split('=')[0]

    # First build graph representation of the parent groups
    G = nx.Graph()
    for i,matcher in enumerate(matchers):
        m_groups = set(matcher[strings])
        for g in m_groups:
            group_node = f'{matcher_names[i]}: {g}'
            string_nodes = matcher.groups[g]
            G.add_nodes_from(string_nodes,type='string',color='w')
            if len(string_nodes) > 1:
                G.add_nodes_from([group_node],type='group',color=f'C{i}',label=group_node)
                nx.add_star(G,[group_node] + string_nodes,color=f'C{i}')

    # Now plot graph components in a grid
    components = sorted(nx.connected_components(G),key=len,reverse=True)

    n_grid = int(np.ceil(np.sqrt(len(components))))
    grid_xy = [(x,-y) for y in range(n_grid) for x in range(n_grid)]

    if ax is None:
        fig, ax = plt.subplots()

    for i,component in enumerate(components):
        G_sub = G.subgraph(component)

        x0,y0 = grid_xy[i]

        # Position nodes
        if len(component) > 1:
            pos = nx.random_layout(G_sub)
            pos = nx.kamada_kawai_layout(G_sub,pos=pos,scale=0.25)
            pos = {n:(x0+x,y0+y) for n,(x,y) in pos.items()}
        else:
            pos = {list(component)[0]:(x0,y0)}

        edges = list(G_sub.edges(data=True))

        edge_coord = [[pos[n0],pos[n1]] for n0,n1,d in edges]
        edge_colors = [mplt.colors.to_rgba(d['color']) for n0,n1,d in edges]

        lc = mplt.collections.LineCollection(edge_coord,color=edge_colors,zorder=0)

        ax.add_collection(lc)

        for node,d in G_sub.nodes(data=True):
            x,y = pos[node]
            if d['type'] == 'group':
                ax.scatter(x,y,color=mplt.colors.to_rgb(d['color']),label=d['label'],s=50,zorder=2)
            else:
                ax.scatter(x,y,color='w',s=200,zorder=1)
                ax.text(x,y,node,ha='center',va='center',zorder=3)

    ax.axis('off')

    legend_handles = [mplt.lines.Line2D([],[],color=mplt.colors.to_rgb(f'C{i}'),markersize=15,marker='.',label=f'{name}')
                        for i,name in enumerate(matcher_names)]
    plt.legend(handles=legend_handles,
                bbox_to_anchor=(0.5,-0.2),loc='lower center',
                ncol=3,borderaxespad=0)

    return ax


# df1 = pd.DataFrame(['ABC Inc.','abc inc','A.B.C. INCORPORATED','The XYZ Company','X Y Z CO'],columns=['name'])
# df2 = pd.DataFrame(['ABC Inc.','XYZ Co.'],columns=['name'])
#
# print(f'Toy data:\ndf1=\n{df1}\ndf2=\n{df2}')
#
#
# matcher = Matcher()
#
# matcher = matcher.add_strings(df1['name'])
# matcher = matcher.add_strings(df2['name'])
#
# matcher = matcher.unite(['X Y Z CO','XYZ Co.'])
#
# df = matcher.to_df()
#
# from_df(df)
