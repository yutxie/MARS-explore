import os
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem, RDLogger

from .utils import load_mols, Vocab, load_vocab
from ..utils.chem import break_bond, Arm, Skeleton

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   default='MARS/data')
    parser.add_argument('--mols_file',  type=str,   default='chembl.txt')
    parser.add_argument('--vocab_name', type=str,   default='chembl',)
    parser.add_argument('--vocab_size', type=int,   default=1000)
    parser.add_argument('--max_size',   type=int,   default=10, help='max size of arm')
    parser.add_argument('--func',       action='store_true')
    args = parser.parse_args()

    ### load data
    print('loading molecules from the database...')
    mols = load_mols(args.data_dir, args.mols_file)
    print('loaded %i mols' % len(mols))

    ### drop arms
    arms, cnts, smiles2idx = [], [], {}
    for mol in tqdm(mols):
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if not bond.GetBondType() == \
                Chem.rdchem.BondType.SINGLE: continue
            try: skeleton, arm = break_bond(mol, u, v)
            except ValueError: continue

            for reverse in [False, True]:
                if reverse is True:
                    tmp = arm
                    arm = Arm(skeleton.mol, skeleton.u, skeleton.bond_type)
                    skeleton = Skeleton(tmp.mol, tmp.v, tmp.bond_type)
                if arm.mol.GetNumAtoms() > args.max_size: continue

                # functional group check
                if args.func:
                    mark = False
                    if not skeleton.mol.GetAtomWithIdx( # connected with C
                        skeleton.u).GetAtomicNum() == 6: continue
                    for atom in arm.mol.GetAtoms(): # contain non-C atoms
                        if not atom.GetAtomicNum() == 6:
                            mark = True
                            break
                    for bond in arm.mol.GetBonds(): # contain non-single bonds
                        if mark: break
                        if bond.GetBondType() == \
                            Chem.rdchem.BondType.DOUBLE or \
                            bond.GetBondType() == \
                            Chem.rdchem.BondType.TRIPLE:
                            mark = True
                            break
                    if not mark: continue

                smiles = Chem.MolToSmiles(arm.mol, rootedAtAtom=arm.v)
                if args.func and smiles.startswith('CC'): continue
                if smiles2idx.get(smiles) is None:
                    smiles2idx[smiles] = len(arms)
                    arms.append(arm)
                    cnts.append(1)
                else: cnts[smiles2idx[smiles]] += 1

    ### save arms as a vocab
    topk = sorted(range(len(cnts)), key=lambda k: cnts[k], reverse=True)[:args.vocab_size]
    arms = [arms[i] for i in topk]
    cnts = [cnts[i] for i in topk]
    topk_set = set(topk)
    smiles = [smi for smi, idx in smiles2idx.items() if idx in topk_set]
    vocab = Vocab(arms, cnts, smiles)
    vocab.save(args.data_dir, args.vocab_name)
    vocab = load_vocab(args.data_dir, args.vocab_name, args.vocab_size)
    
    