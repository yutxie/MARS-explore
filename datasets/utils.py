import os
import csv
from rdkit import Chem

from ..utils.chem import BOND_TYPES, Arm

def load_mols(data_dir, mols_file):
    path = os.path.join(data_dir, mols_file)
    if mols_file.endswith('.smiles'): # processed zinc data
        mols = []
        with open(path, 'r') as f:
            lines = f.readlines(int(1e8))
            lines = [line.strip('\n').split('\t') for line in lines]
            smiles = [line[1] for line in lines]
    elif mols_file.endswith('.csv'): # zinc_250k.csv
        reader = csv.reader(open(path, 'r'))
        smiles = [line[0].strip() for line in reader]
        smiles = smiles[1:]
    elif mols_file == 'chembl.txt':
        with open(path, 'r') as f:
            lines = f.readlines()
            smiles = [line.strip() for line in lines]
    elif mols_file == 'kinase.tsv': # kinase data
        with open(path, 'r') as f:
            lines = f.readlines()[2:]
            lines = [line.strip().split('\t') for line in lines]
            lines = [line for line in lines if line[1] == '1']
            smiles = [line[-1] for line in lines]
    elif mols_file.startswith('actives'): # refenrence active mols
        with open(path, 'r') as f:
            lines = f.readlines()[1:]
            lines = [line.strip().split(',') for line in lines]
            smiles = [line[0] for line in lines]
    else: raise NotImplementedError
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [mol for mol in mols if mol]
    print('loaded %i molecules' % len(mols))
    return mols


class Vocab():
    def __init__(self, arms, cnts, smiles):
        self.arms = arms
        self.cnts = cnts
        self.smiles = smiles
        self.smiles2idx = {smi : idx for idx, smi in enumerate(smiles)}

    def __len__(self):
        return len(self.arms)

    def save(self, data_dir, vocab_name):
        print('saving vocab...')
        vocab_dir = os.path.join(data_dir, 'vocab_%s' % vocab_name)
        os.makedirs(vocab_dir, exist_ok=True)

        sd_writer = Chem.SDWriter(os.path.join(vocab_dir, 'arms.sdf'))
        for arm in self.arms:
            sd_writer.write(arm.mol)

        with open(os.path.join(vocab_dir, 'arms.smiles'), 'w') as f:
            for i in range(len(self)):
                arm = self.arms[i]
                cnt = self.cnts[i]
                smi = self.smiles[i]
                v = arm.v
                bond = int(arm.bond_type)
                cnt, v, bond = map(str, (cnt, v, bond))
                f.write('\t'.join([cnt, smi, v, bond]) + '\n')
        print('saved vocab of size %i' % len(self))

def load_vocab(data_dir, vocab_name, vocab_size=1000):
    vocab_dir = os.path.join(data_dir, 'vocab_%s' % vocab_name)
    
    sd_supplier = Chem.SDMolSupplier(os.path.join(vocab_dir, 'arms.sdf'))
    arm_mols = [mol for mol in sd_supplier]

    with open(os.path.join(vocab_dir, 'arms.smiles'), 'r') as f:
        lines = f.readlines()
        lines = lines[:vocab_size]
        lines = [line.strip('\n').split('\t') for line in lines]
        cnts, smiles, vs, bonds = zip(*lines)
        cnts, vs, bonds = map(lambda lst: map(int, lst), (cnts, vs, bonds))

    arms = []
    for mol, v, bond in zip(arm_mols, vs, bonds):
        arms.append(Arm(mol, v, BOND_TYPES[bond]))

    vocab = Vocab(arms, cnts, smiles)
    print('loaded vocab of size %i' % len(vocab))
    return vocab
