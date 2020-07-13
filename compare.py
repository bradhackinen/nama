import numpy as np
import math
import networkx as nx
from itertools import product


def edge_compare(matcherA, matcherB):
    n = len(matcherA.strings(min_count=0))

    edges_A = {tuple(sorted((i, j))) for i, j in matcherA.G.edges()}
    edges_B = {tuple(sorted((i, j))) for i, j in matcherB.G.edges()}

    n_misses = len(edges_A ^ edges_B)

    return 1 - 2*n_misses/(n**2 - n)


def component_compare(matcherA, matcherB):

    n = len(matcherA.strings(min_count=0))

    components_A = [set(c) for c in matcherA.components()]
    components_B = [set(c) for c in matcherB.components()]

    n_misses = 0
    for c_A, c_B in list(product(components_A, components_B)):
        n_misses += len(c_A & c_B) * len(c_A ^ c_B)

    return round(abs(1 - 2*n_misses/(n**2 - n)), 3)


def bp_compare(matcherA, matcherB):
    g1 = matcherA.G
    g2 = matcherB.G

    g1A = nx.to_numpy_matrix(g1)
    g2A = nx.to_numpy_matrix(g2)

    g1L = nx.laplacian_matrix(g1).todense()
    g2L = nx.laplacian_matrix(g2).todense()

    g1D = np.add(g1L, g1A)
    g2D = np.add(g2L, g2A)

    g1I = np.identity(len(g1))
    g2I = np.identity(len(g2))

    g1_HF = _compute_homophily_factor(g1D)
    g2_HF = _compute_homophily_factor(g2D)

    (g1a, g1c_prime) = _compute_FaBP_constants(g1_HF)
    (g2a, g2c_prime) = _compute_FaBP_constants(g2_HF)

    g1_epsilon = 1 / (1 + np.where(g1D == np.amax(g1D))[0][0])
    g2_epsilon = 1 / (1 + np.where(g2D == np.amax(g2D))[0][0])

    # g1_to_invert = np.add(g1I, g1a * g1D, -g1c_prime * g1A)
    # g2_to_invert = np.add(g2I, g2a * g2D, -g2c_prime * g2A)
    g1_to_invert = np.add(g1I, g1_epsilon**2 * g1D, -g1_epsilon * g1A)
    g2_to_invert = np.add(g2I, g2_epsilon**2 * g2D, -g2_epsilon * g2A)

    g1_inverse = np.linalg.inv(g1_to_invert)
    g2_inverse = np.linalg.inv(g2_to_invert)

    dist = np.linalg.norm(g1_inverse - g2_inverse)

    s = 1 / (1 + dist)

    return s


def _compute_homophily_factor(D):
    one_norm = 1 / (2 + 2 * np.where(D == np.amax(D))[0][0])
    c1 = 2 + np.sum(D)
    c2 = np.sum(np.square(D)) - 1
    frobenius_norm = math.sqrt(
        (-c1 + math.sqrt(c1**2 + 4 * c2)) / (8 * c2))
    return max(one_norm, frobenius_norm)


def _compute_FaBP_constants(HF):
    about_half_HF = HF - 0.5
    four_HF_squared = 4 * about_half_HF**2
    a = (four_HF_squared) / (1 - four_HF_squared)
    c_prime = (2 * about_half_HF) / (1 - four_HF_squared)
    return (a, c_prime)
