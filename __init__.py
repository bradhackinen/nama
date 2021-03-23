import os
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mplt


root_dir = Path(os.path.dirname(os.path.abspath(__file__)))


class Matcher():
    def __init__(self,strings=None):

        self.counts = Counter()
        self.labels = {}
        self.groups = {}

        if strings:
            self.add(strings,inplace=True)

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f'<nama.Matcher containing {len(self)} strings in {len(self.groups)} groups>'

    def __contains__(self,s):
        return s in self.labels

    def __getitem__(self,strings):
        if isinstance(strings,str):
            return self.labels[strings]
        else:
            return [self.labels[s] for s in strings]

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
        """List strings in the matcher"""
        return list(self.labels.keys())

    def info(self,strings):
        """ Return all information about strings"""
        return [{'string':s,'count':self.counts[s],
                'group':self.labels[s],'matched':self.groups[i]}
                for s in strings]

    def add(self,strings,inplace=False):
        """Add new strings to the matcher"""
        if isinstance(strings,str):
            strings = [strings]
        else:
            counts = Counter(strings)

        if not inplace:
            self = self.copy()

        for s in counts.keys():
            if s not in self.counts:
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

        for s in strings:
            del self.counts[s]
            del self.labels[s]

        # Remove strings from their groups
        self.groups = {label:[s for s in strings if s in self.labels] \
                        for label,strings in self.groups.items()}

        # Remove empty groups
        self.groups = {label:strings for label,strings in self.groups.items if strings}


    def keep(self,strings,inplace=False):
        """Drop all strings from the matcher except the passed strings"""
        if isinstance(strings,str):
            strings = [strings]

        strings = set(strings)

        to_drop = [s for s in self.strings() if s not in strings]

        return self.drop(to_drop,inplace=inplace)


    def unite(self,arg,inplace=False):
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

        elif callable(arg):
            # Assume arg is a mapping from strings to labels and unite by label
            groups = {s:arg(s) for s in self.strings()}
            self.unite(groups,inplace=True)

        elif isinstance(arg,dict):
            # Assume arg is a mapping from strings to labels and unite by label
            groups = {label:[] for label in arg.values()}
            for string,label in arg.items():
                groups[label].append(string)
            for group in groups.values():
                self.unite(group,inplace=True)

        elif all(isinstance(s,str) for s in arg):

            # Main case: Unite group of strings

            # Identify old groups that will be replaced
            old_labels = set(self[arg])

            # Only need to do the merge if the strings span multiple groups
            if len(old_labels) > 1:

                # Build the new group
                new_group = {s for label in old_labels for s in self.groups[label]}
                new_group = sorted(new_group,key=lambda s: (-self.counts[s],s))
                new_label = new_group[0]

                # Update string labels
                for s in new_group:
                    self.labels[s] = new_label

                # Update group dict
                for label in old_labels:
                    del self.groups[label]

                self.groups[new_label] = new_group

        else:
            # Assume arg is a iterable of groups to unite
            for group in arg:
                self.unite(group,inplace=True)

        return self

    def split(self,strings,inplace=False):
        """
        Separate strings into singleton groups.  Can pass:
         - A nested list to split each set of strings.
         - A single string to move that string to a singleton group
        """
        if not inplace:
            self = self.copy()

        if isinstance(strings,str):
            self.split([strings],inplace=True)

        if isinstance(strings[0],str):
            strings = set(strings)

            # Remove strings from their groups
            self.groups = {label:[s for s in group_strings if s not in strings] \
                            for label,group_strings in self.groups.items()}

            # Remove empty groups
            self.groups = {label:strings for label,strings in self.groups.items if strings}

            # Update labels and add singleton groups
            for s in strings:
                self.labels[s] = s
                self.groups[s] = [s]
        else:
            for group in strings:
                self.split(group,inplace=True)

    def print_groups(self,strings=None,singletons=True,max_lines=1000):
        if strings is None:
            strings = self.strings()
        elif isinstance(strings,str):
            strings = [strings]

        labels = sorted(set(self[strings]), key=lambda g:(-len(self.groups[g]),g))

        lines_remaining = max_lines
        for label in labels:
            group = self.groups[label]
            if singletons or len(group) > 1:
                if lines_remaining:
                    print('\n'+'\n'.join(group[:lines_remaining]))

                    lines_remaining -= len(group)
                else:
                    print(f'...\n(Output truncated at {max_lines=})')
                    break

    def to_df(self,singletons=True,sort_groups=True):
        """
        Convert the matcher to a dataframe with string,count and group columns.
        """
        strings = self.strings()

        if singletons:
            df = pd.DataFrame([(s,self.counts[s],self.labels[s]) for s in strings],
                            columns=['string','count','group'])
        else:
            df = pd.DataFrame([(s,self.counts[s],self.labels[s]) for s in strings if len(self.groups[self[s]])>1],
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

    if not match_format in ['groups','pairs','detect']:
        raise ValueError('match_format must be one of "groups", "pairs", or "detect"')

    # Create an empty matcher
    matcher = Matcher()

    if match_format == 'detect':
        if (string_column in df.columns) & (group_column in df.columns):
            match_format = 'groups'
        elif set(df.columns) == set(pair_columns):
            match_format = 'pairs'
        else:
            raise

    if (string_column in df.columns) & (group_column in df.columns):
        matcher.add(df[string_column],inplace=True)

        groups = {s:g for s,g in df[[string_column,group_column]].values}

        matcher.unite(groups,inplace=True)

        if count_column in df.columns:
            for s,c in df[[string_column,count_column]].values:
                matcher.counts[s] = c

        return matcher
    # Detect whether the dataframe contains pairs or labeled strings
    if set(df.columns) == set(pair_columns):

        for pair_column in pair_columns:
            matcher.add(df[pair_column],inplace=True)

        # There are several ways to unite pairs
        # Guessing it is most efficient to "group by" one of the string columns
        groups = {s:pair[1] for pair in df[pair_columns].values for s in pair}

        matcher.unite(groups,inplace=True)

        return matcher

def read_csv(filename,match_format='detect',**pandas_args):

    df = pd.read_csv(filename,**pandas_args)

    return from_df(df,match_format=match_format)

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
# matcher = matcher.add(df1['name'])
# matcher = matcher.add(df2['name'])
#
# matcher = matcher.unite(['X Y Z CO','XYZ Co.'])
#
# df = matcher.to_df()
#
# from_df(df)
