from rdkit import DataStructs

from .scorers import sa_scorer
from .scorers import kinase_scorer
from .scorers import other_scorer
from ..utils.chem import standardize_mol, mol_to_fp

class Scorer():
    def __init__(self, config, fps_uniq):
        '''
        @params:
            config (dict): configurations
        '''
        self.objectives = config['objectives']
        self.score_wght = config['score_wght']
        self.score_clip = config['score_clip']
        self.fps_uniq = fps_uniq

    def weighted_scores(self, dicts):
        '''
        @params:
            dicts (list): list of score dictionaries
        @return:
            avg_scores (list): sum of property scores of each molecule after clipping
        '''
        avg_scores = []
        score_norm = sum(self.score_wght.values())
        for score_dict in dicts:
            avg_score = 0.
            for k in self.objectives:
                v = score_dict[k]
                if self.score_clip[k] > 0.:
                    v = min(v, self.score_clip[k])
                avg_score += self.score_wght[k] * v
            avg_score /= score_norm
            avg_score = max(avg_score, 0.)
            avg_scores.append(avg_score)
        return avg_scores

    def get_scores(self, mols, old_dicts=None):
        '''
        @params:
            mols: molecules to estimate score
        @return:
            dicts (list): list of score dictionaries
        '''
        mols = [standardize_mol(mol) for mol in mols]
        mols_valid = [mol for mol in mols if mol is not None]
        dicts = old_dicts if old_dicts is not None \
            else [{} for _ in mols]

        objectives = set(self.objectives)
        objectives.update(['nov_nn', 'nov_ad'])
        for obj in objectives:
            if old_dicts is not None and \
                not obj.startswith('nov'): continue

            if obj.startswith('nov'):
                N = len(self.fps_uniq)
                if N == 0:
                    scores = [1. for _ in mols_valid]
                else:
                    scores = []
                    fps = [mol_to_fp(mol) for mol in mols_valid]
                    for fp in fps:
                        sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_uniq)
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
        scores = self.weighted_scores(dicts)
        return scores, dicts

