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
        self.run_dir = run_dir
        self.smiles_uniq = set()

    def print_mols(self, step, mols, scores, dicts):
        path = os.path.join(self.run_dir, 'mols.txt')
        with open(path, 'a') as f:
            f.write('molecules obtained at step %i\n' % step)
            names = list(dicts[0].keys())
            f.write('#\tscore\t%s\tsmiles\n' % '\t'.join(names))
            for i, mol in enumerate(mols):
                smiles = mol_to_smiles(mol)
                if smiles in self.smiles_uniq: continue
                self.smiles_uniq.add(smiles)
                if smiles is not None:
                    score = scores[i]
                    target_scores = [dicts[i][name] for name in names]
                else:
                    score = 0. 
                    smiles = '[INVALID]'
                    target_scores = [0. for _ in names]
                    assert False
                target_scores = ['%f' % _ for _ in target_scores]
                f.write('%i\t%f\t%s\t%s\n' % (
                    i, score, '\t'.join(target_scores), smiles))
