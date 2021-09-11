# modifed from: https://github.com/wengong-jin/hgraph2graph/blob/master/props/properties.py

import math
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
import networkx as nx

from . import sa_scorer


def get_score(obj, mol):
    try:
        if obj == 'qed': 
            return QED.qed(mol)
        elif obj == 'sa': 
            x = sa_scorer.calculateScore(mol)
            return (10. - x) / 9. # normalized to [0, 1]
        elif obj == 'mw': # molecular weight
            return mw(mol)
        elif obj == 'logp': # real number
            return Descriptors.MolLogP(mol)
        elif obj == 'penalized_logp':
            return penalized_logp(mol)
        elif 'rand' in obj:
            raise NotImplementedError
            # return rand_scorer.get_score(obj, mol)
        else: raise NotImplementedError
    except ValueError:
        return 0.

def mw(mol):
    '''
    molecular weight estimation from qed
    '''
    x = Descriptors.MolWt(mol)
    a, b, c, d, e, f = 2.817, 392.575, 290.749, 2.420, 49.223, 65.371
    g = math.exp(-(x - c + d/2) / e)
    h = math.exp(-(x - c - d/2) / f)
    x = a + b / (1 + g) * (1 - 1 / (1 + h))
    return x / 104.981
    
def penalized_logp(mol):
    # Modified from https://github.com/bowenliu16/rl_graph_generation
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sa_scorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle
