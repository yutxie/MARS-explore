import os
import math
import torch
import random
import logging as log
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from ..utils.chem import standardize_mol, mol_to_fp, mol_to_smiles
from ..datasets.utils import load_mols
from ..datasets.sampler import StreamSampler


class Evaluator():
    def __init__(self, config, scorer, measure, mols_refe=[]):
        self.scorer = scorer
        self.measure = measure

        ### model evaluation
        # self.fps_refe = [mol_to_fp(mol) for mol in mols_refe]
        # self.fps_uniq = StreamSampler(S=5000) # for novelty optimization
        # self.fps_succ = StreamSampler(S=5000) # for novelty and diversity evaluation
        # self.fps_circ = []                    # for coverage
        self.smiles_uniq = set()
        self.score_succ = config['score_succ']

        self.N = 0
        self.n_succ = 0
        # self.n_novl = 0
        # self.similarity = 0. # unnormalized
        self.n_succ_dict = {k: 0 for k in config['objectives']} 

    def get_scores(self, *args, **kargs):
        return self.scorer.get_scores(*args, **kargs)

    def novelty(self, mols):
        return self.measure.novelty(mols)

    def update(self, mols, dicts):
        self.N += len(mols)
        self.measure.update(mols)

        ### uniqueness
        # fps_uniq   = [] # len() < n
        dicts_uniq = [] # len() < n
        for mol, score_dict in zip(mols, dicts):
            mol = standardize_mol(mol)
            smiles = mol_to_smiles(mol)
            if smiles is None: continue
            if smiles not in self.smiles_uniq:
                self.smiles_uniq.add(smiles)

                # fp = mol_to_fp(mol)
                # fps_uniq.append(fp)
                dicts_uniq.append(score_dict)
        # self.fps_uniq.update(fps_uniq)
            
        ### success rate, novelty, and diversity
        # for fp, score_dict in zip(fps_uniq, dicts_uniq):
        for score_dict in dicts_uniq:
            all_success = True
            # for k, v in score_dict.items():
            for k in self.scorer.objectives:
                v = score_dict[k]
                if v >= self.score_succ[k]:
                    self.n_succ_dict[k] += 1.
                else: all_success = False
            
            # success
            if all_success:
                # novelty
                # sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_refe)
                # if len(sims) == 0: sims = [0]
                # if max(sims) < 0.4: self.n_novl += 1

                # diversity
                # sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_succ)
                # self.similarity += sum(sims) / len(self.fps_succ) * self.n_succ \
                #     if len(sims) > 0 else 0.

                # coverage
                # sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_circ)
                # if len(sims) == 0: sims = [0]
                # if max(sims) < 0.4: self.fps_circ.append(fp)

                self.n_succ += 1
                # self.fps_succ.update([fp])

    def get_results(self):
        n_uniq = len(self.smiles_uniq)
        metrics = {
            # 'unique'   : 1. *      n_uniq     / self.N,
            # 'success'  : 1. * self.n_succ     /      n_uniq if n_uniq      > 0 else 0.,
            # 'novelty'  : 1. * self.n_novl     / self.n_succ if self.n_succ > 0 else 0.,
            # 'diversity': 1. - self.similarity / self.n_succ / (self.n_succ-1) * 2 if self.n_succ > 1 else 0.
            # 'n_div'  : self.n_succ * evaluation['diversity'],
            # 'n_ess'  : self.n_succ / (1. + (self.n_succ-1.) * (1. - evaluation['diversity'])) \
            #             if self.n_succ > 0 else 0.,
            'n_uniq' : n_uniq,
            'n_succ' : self.n_succ,
            # 'n_circ' : len(self.fps_circ)
            'measure' : self.measure.report(),
        }
        # succ_dict = {k : 1. * v / n_uniq for k, v in self.n_succ_dict.items()}
        # return metrics, succ_dict
        return metrics, self.n_succ_dict
