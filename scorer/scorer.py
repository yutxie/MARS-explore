from rdkit import DataStructs

from .scorers import sa_scorer
from .scorers import kinase_scorer
from .scorers import other_scorer
from ..utils.chem import standardize_smiles, fingerprint

class Scorer():
    def __init__(self, config):
        '''
        @params:
            config (dict): configurations
        '''
        self.objectives = config['objectives']

    def get_scores(self, mols, fps_uniq=[]):
        '''
        @params:
            mols: molecules to estimate score
        @return:
            dicts (list): list of score dictionaries
        '''
        mols = [standardize_smiles(mol) for mol in mols]
        mols_valid = [mol for mol in mols if mol is not None]

        dicts = [{} for _ in mols]
        for obj in self.objectives:
            if obj.startswith('nov'):
                N = len(fps_uniq)
                if N == 0:
                    scores = [1. for _ in mols_valid]
                else:
                    scores = []
                    fps = [fingerprint(mol) for mol in mols_valid]
                    for fp in fps:
                        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_uniq)
                        if   obj.endswith('nn'): scores.append(1. - max(sims))
                        elif obj.endswith('ad'): scores.append(1. - sum(sims) * 1. / N)
                        else: raise NotImplementedError
                
            elif obj == 'jnk3' or \
                 obj == 'gsk3b':
                scores = kinase_scorer.get_scores(obj, mols_valid)
            # elif obj == 'drd2':
            #     scores = drd2_scorer.get_scores(mols_valid)
            # elif obj.startswith('chemprop'):
            #     scores = chemprop_scorer.get_scores(obj, mols_valid)
            else: scores = [other_scorer.get_score(obj, mol) for mol in mols_valid]

            scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
            for i, mol in enumerate(mols):
                dicts[i][obj] = scores[i]
        return dicts

