from rdkit import Chem

from ..utils.chem import standardize_smiles
from .scorers import sa_scorer
from .scorers import kinase_scorer
from .scorers import other_scorer

class Scorer():
    def __init__(self, config):
        '''
        @params:
            config (dict): configurations
        '''
        self.objectives = config['objectives']

    def get_scores(self, mols):
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
            # scores = get_scores(obj, mols)
            if objective == 'drd2':
                scores = drd2_scorer.get_scores(mols_valid)
            elif objective == 'jnk3' or objective == 'gsk3b':
                scores = kinase_scorer.get_scores(objective, mols_valid)
            elif objective.startswith('chemprop'):
                scores = chemprop_scorer.get_scores(objective, mols_valid)
            else: scores = [other_scorer(objective, mol) for mol in mols_valid]

            scores = [scores.pop(0) if mol is not None else 0. for mol in mols]
            for i, mol in enumerate(mols):
                dicts[i][obj] = scores[i]
        return dicts

