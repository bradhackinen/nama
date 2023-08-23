import re


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
    s = re.sub(r'[\s\.,!@#$%^&*:;/\'"`´‘’“”\(\)_—\-]+', ' ', s)
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


# MatchData Tools

# def plot(matches, strings, matches_names=None, ax=None):
#     """
#     Plots strings and their parent groups for multiple matches as a graph, with
#     groups represented as nodes that connect strings.

#     Parameters
#     ----------
#     matches : MatchData or list of MatchData
#         a matches or list of matches to plot
#     strings : str or list of str
#         a string or list of strings to plot (all connected strings will also be plotted)
#     matches_names : list of str, optional
#         a list of strings to label matches in the plot legend
#     ax : matplotlib.axes._subplots.AxesSubplot, optional
#         a matplotlib axis object to draw the plot on.

#     Returns
#     -------
#     matplotlib.axes._subplots.AxesSubplot
#         The matplotlib axis object with the plot.
#     """

#     if isinstance(matches, MatchData):
#         matches = [matches]

#     if isinstance(strings, str):
#         strings = [strings]

#     if not matches_names:
#         matches_names = [f'matches{i}' for i in range(len(matches))]
#     elif not (len(matches_names) == len(matches)):
#         raise ValueError('matches_names must be the same length as matches')

#     def varname(x): return f'{x=}'.split('=')[0]

#     # First build graph representation of the parent groups
#     G = nx.Graph()
#     for i, matches in enumerate(matches):
#         m_groups = set(matches[strings])
#         for g in m_groups:
#             group_node = f'{matches_names[i]}: {g}'
#             string_nodes = matches.groups[g]
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
#         name in enumerate(matches_names)]
#     plt.legend(handles=legend_handles,
#                bbox_to_anchor=(0.5, -0.2), loc='lower center',
#                ncol=3, borderaxespad=0)

#     return ax