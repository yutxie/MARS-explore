import rdkit
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from ..datasets.sampler import StreamSampler


def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

def fingerprints(mols):
    return [fingerprint(mol) for mol in mols]

def similarities_tanimoto(fp, fps):
    return DataStructs.BulkTanimotoSimilarity(fp, fps)

def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)


class Measure():
    def __init__(self, 
            vectorizer=fingerprints, 
            sim_mat_func=similarity_matrix_tanimoto
        ):
        self.vectorizer = vectorizer
        self.sim_mat_func = sim_mat_func
        self.mols = None
        self.vecs = None

    def update(self, mols=[]):
        pass

    def novelty(self, mols=[]):
        '''
        novelty of molecules
        @return:
            res : list of length len(mols)
        '''
        return [0] * len(mols)

    def report(self):
        raise NotImplementedError
    
    
class DissimilarityBasedMeasure(Measure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        # self.dis_mat = np.zeros((0, 0))
        self.vecs = StreamSampler()

    def update(self, mols=[]):
        vecs = self.vectorizer(mols)
        self.vecs.update(vecs)
        # self.dis_mat = 1. - \
        #     self.sim_mat_func(self.vecs, self.vecs)


class Diversity(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def novelty(self, mols=[], normalize=False):
        if len(self.vecs) < 1: return [0] * len(mols)
        vecs = self.vectorizer(mols)
        dis_mat = 1. - self.sim_mat_func(vecs, self.vecs) # (n_new, n_old)
        if normalize:
            dis_sum = dis_mat.sum(axis=1) # (n_new,)
            old_div = self.report()
            m = self.dis_mat
            n = m.shape[0]
            new_div = (dis_sum * 2 + m.sum()) / n / (n+1)
            return new_div - old_div
        else: return dis_mat.mean(axis=1) # (n_new,)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        return (m.sum(axis=-1) / (n - 1)).mean().item()
    
class SumBottleneck(DissimilarityBasedMeasure):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def novelty(self, mols=[], approx=True):
        if len(self.vecs) < 1: return [0] * len(mols)
        vecs = self.vectorizer(mols)
        dis_mat = 1. - self.sim_mat_func(vecs, self.vecs) # (n_new, n_old)
        if not approx:
            raise NotImplementedError
        else: return dis_mat.min(axis=1) # (n_new,)

    def report(self):
        if len(self.vecs) < 2: return 0
        m = self.dis_mat
        n = m.shape[0]
        m = m + np.eye(n) * 1e9
        return m.min(axis=-1).sum().item()


class ReferenceBasedMeasure(Measure):
    def __init__(self):
        super().__init__()

from .cal_ifg_atom import CollectFG
class NFG(ReferenceBasedMeasure):
    def __init__(self):
        super().__init__()
        self.frags = set()
        
    def update(self, mols=[]):
        frags = CollectFG(mols)
        self.frags.update(frags)

    def novelty(self, mols=[]):
        res = []
        for mol in mols:
            frags = CollectFG([mol])
            res.append(len([f for f in frags if f not in self.frags]))
        return res
        
    def report(self):
        return len(self.frags)


class NCircles(Measure):
    def __init__(self, t=0.10, *args, **kargs):
        super().__init__(*args, **kargs)
        self.vecs = StreamSampler()
        self.t = t # t (similarity threshold) = 1-t in the paper

    def update(self, mols=[]):
        vecs = self.vectorizer(mols)
        for vec in vecs:
            if len(self.vecs) > 0:
                sims = self.sim_mat_func([vec], self.vecs)
                if sims.max() >= self.t:
                    continue
            self.vecs.update([vec])

    def novelty(self, mols=[]):
        if len(self.vecs) < 1: return [0] * len(mols)
        vecs = self.vectorizer(mols)
        sim_mat = self.sim_mat_func(vecs, self.vecs)
        sim_max = sim_mat.max(axis=1)
        return sim_max < self.t

    def report(self):
        return self.vecs.N