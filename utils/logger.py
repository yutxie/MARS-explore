import os

from .chem import mol_to_smiles


class Recorder():
    def __init__(self, metrics):
        self.metrics = metrics
        self.metric2sum = {}
        self.n_records = 0
        self.reset()

    def reset(self):
        self.n_records = 0
        for metric in self.metrics:
            self.metric2sum[metric] = 0.

    def record(self, n_records, values):
        self.n_records += n_records
        for k, v in zip(self.metrics, values):
            self.metric2sum[k] += v * n_records

    def report_avg(self):
        '''
        @return:
            metric2avg : dictionary from metric names to average values
        '''
        metric2avg = {}
        for metric in self.metrics:
            summ = self.metric2sum[metric]
            avg = summ / self.n_records
            metric2avg[metric] = avg
        return metric2avg

class MolsPrinter():
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'mols.csv')
        self.smiles_uniq = set()

    def print_head(self, dicts):
        names = list(dicts[0].keys())
        with open(self.path, 'a') as f:
            f.write('Step,Score,%s,SMILES\n' % ','.join(names))

    def print_mols(self, step, mols, scores, dicts):
        if step == -1: self.print_head(dicts)
        names = list(dicts[0].keys())
        with open(self.path, 'a') as f:
            for i, mol in enumerate(mols):
                smiles = mol_to_smiles(mol)
                if smiles in self.smiles_uniq: continue
                self.smiles_uniq.add(smiles)
                if smiles is not None:
                    score = scores[i]
                    prop_scores = [dicts[i][name] for name in names]
                else: assert False
                    # score = 0. 
                    # smiles = '[INVALID]'
                    # prop_scores = [0. for _ in names]
                prop_scores = ['%f' % _ for _ in prop_scores]
                f.write('%i,%f,%s,%s\n' % (
                    step, score, ','.join(prop_scores), smiles))
