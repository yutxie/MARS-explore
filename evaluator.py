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

from .utils.train import train
from .utils.chem import mol_to_dgl
from .utils.utils import print_mols
from .datasets.utils import load_mols
from .datasets.datasets import ImitationDataset, \
                               GraphClassificationDataset

from .utils.chem import standardize_smiles, fingerprint


class Evaluator():
    def __init__(self, config, mols_refe):
        self.fps_refe = [fingerprint(mol) for mol in mols_refe]

        self.fps_uniq = [] # for novelty optimization
        self.smiles_uniq = set()

        self.fps_succ = [] # for novelty and diversity evaluation
        self.score_succ = {k: v for k, v in zip(config['objectives'], config['score_succ'])}

        self.N = 0
        self.n_succ = 0
        self.n_novl = 0
        self.similarity = 0. # unnormalized
        self.n_succ_dict = {k: 0 for k in config['objectives']} 

    def update(self, mols, dicts):
        self.N += len(mols)

        ### uniqueness
        fps_uniq   = [] # len() < n
        dicts_uniq = [] # len() < n
        for mol, score_dict in zip(mols, dicts):
            smiles = standardize_smiles(mol)
            if smiles not in self.smiles_uniq:
                self.smiles_uniq.add(smiles)

                fp = fingerprint(mol)
                fps_uniq.append(fp)
                dicts_uniq.append(score_dict)
        self.fps_uniq += fps_uniq
            
        ### success rate, novelty, and diversity
        for fp, score_dict in zip(fps_uniq, dicts_uniq):
            all_success = True
            for k, v in score_dict.items():
                if v >= self.score_succ[k]:
                    self.n_succ_dict[k] += 1
                else: all_success = False
            
            if all_success:
                self.n_succ += 1
                self.fps_succ.append(fp)

                sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_refe)
                if len(sims) == 0: sims = [0]
                if max(sims) < 0.4: self.n_novl += 1

                sims = DataStructs.BulkTanimotoSimilarity(fp, self.fps_succ[:-1])
                self.similarity += sum(sims)

    def get_results(self):
        n_uniq = len(self.smiles_uniq)
        scalars = {
            # 'N': self.N,
            'unique'   : 1. *      n_uniq     / self.N,
            'success'  : 1. * self.n_succ     /      n_uniq if n_uniq      > 0 else 0.,
            'novelty'  : 1. * self.n_novl     / self.n_succ if self.n_succ > 0 else 0.,
            'diversity': 1. - self.similarity / self.n_succ / (self.n_succ-1) * 2 if self.n_succ > 1 else 0.
        }
        succ_dict = {k : 1. * v / self.N for k, v in self.n_succ_dict.items()}
        return scalars, succ_dict



