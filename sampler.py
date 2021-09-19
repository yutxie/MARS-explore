import os
import math
import torch
import random
import logging as log
from tqdm import tqdm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from .utils.train import train
from .utils.utils import print_mols
from .datasets.datasets import GraphEditingDataset

class Sampler():
    def __init__(self, config, scorer, proposal, evaluator):
        self.proposal = proposal
        self.scorer = scorer
        self.evaluator = evaluator
        
        self.writer = None
        self.run_dir = None
        
        ### for sampling
        self.step = None
        self.num_path = config['num_path']
        self.num_step = config['num_step']
        self.log_every = config['log_every']
        self.score_wght = config['score_wght']
        self.score_clip = config['score_clip']

        ### for editing and training
        self.train = config['train']
        self.last_avg_size = None
        self.batch_size = config['batch_size']
        if self.train:
            self.dataset = GraphEditingDataset()
            self.max_dataset_size = config['dataset_size']
            self.optimizer = torch.optim.Adam(self.proposal.editor.parameters(), lr=config['lr'])

    def average_scores(self, dicts):
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
            for k, v in score_dict.items():
                if self.score_clip[k] > 0.:
                    v = min(v, self.score_clip[k])
                avg_score += self.score_wght[k] * v
            avg_score /= score_norm
            avg_score = max(avg_score, 0.)
            avg_scores.append(avg_score)
        return avg_scores

    def record(self, step, mols, dicts, acc_rates):
        ### average score and size
        avg_scores = self.average_scores(dicts)
        mean_avg_score = 1. * sum(avg_scores) / len(avg_scores)
        sizes = [mol.GetNumAtoms() for mol in mols]
        avg_size = sum(sizes) / len(mols)
        self.last_avg_size = avg_size
        
        ### evaluation results
        evaluation, coverage, succ_dict = self.evaluator.get_results()

        ### logging and writing tensorboard
        log.info('Step: {:02d},\tMean Avg Score: {:.7f}'.format(step, mean_avg_score))
        log.info('\t%s' % str(evaluation))
        log.info('\t%s' % str(coverage))
        log.info('\t%s' % str(succ_dict))
        self.writer.add_scalar('mean_avg_score', mean_avg_score, step)
        self.writer.add_scalar('avg_size', avg_size, step)
        self.writer.add_scalars('evaluation', evaluation, step)
        self.writer.add_scalars('coverage', coverage, step)
        self.writer.add_scalars('succ_dict', succ_dict, step)
        self.writer.add_histogram('acc_rates', torch.tensor(acc_rates), step)
        self.writer.add_histogram('avg_scores', torch.tensor(avg_scores), step)
        for k in dicts[0].keys():
            scores = [score_dict[k] for score_dict in dicts]
            self.writer.add_histogram(k, torch.tensor(scores), step)
        print_mols(self.run_dir, step, mols, avg_scores, dicts)
        
        ### early stop
        # TODO
        
    def acc_rates(self, new_scores, old_scores, pops):
        '''
        compute sampling acceptance rates
        @params:
            new_scores : scores of new proposed molecules
            old_scores : scores of old molcules
            pops       : transition probability p over p for each proposal
        '''
        raise NotImplementedError

    def sample(self, run_dir, mols_init):
        '''
        sample molecules from initial ones
        @params:
            mols_init : initial molecules
        '''
        self.run_dir = run_dir
        self.writer = SummaryWriter(log_dir=run_dir)
        
        ### sample
        old_mols = [mol for mol in mols_init]
        old_dicts = self.scorer.get_scores(old_mols)
        old_scores = self.average_scores(old_dicts)
        acc_rates = [0. for _ in old_mols]
        self.evaluator.update(old_mols, old_dicts)
        self.record(-1, old_mols, old_dicts, acc_rates)

        for step in range(self.num_step):
            self.step = step
            new_mols, pops = self.proposal.propose(old_mols) 
            new_dicts = self.scorer.get_scores(new_mols)
            new_scores = self.average_scores(new_dicts)
            
            indices = [i for i in range(len(old_mols)) if new_scores[i] > old_scores[i]]
            with open(os.path.join(self.run_dir, 'edits.txt'), 'a') as f:
                f.write('edits at step %i\n' % step)
                f.write('improve\tact\tarm\n')
                for i, item in enumerate(self.proposal.dataset):
                    _, edit = item
                    improve = new_scores[i] > old_scores[i]
                    f.write('%i\t%i\t%i\n' % (improve, edit['act'], edit['arm']))
            
            acc_rates = self.acc_rates(new_scores, old_scores, pops)
            acc_rates = [min(1., max(0., A)) for A in acc_rates]
            updated_mols, updated_dicts = [], []
            for i in range(self.num_path):
                A = acc_rates[i] # A = p(x') * g(x|x') / p(x) / g(x'|x)
                if random.random() > A: continue
                old_mols[i] = new_mols[i]
                old_scores[i] = new_scores[i]
                old_dicts[i] = new_dicts[i]
                updated_mols.append(new_mols[i])
                updated_dicts.append(new_dicts[i])

            self.evaluator.update(updated_mols, updated_dicts)
            if step % self.log_every == 0:
                self.record(step, old_mols, old_dicts, acc_rates)

            ### train editor
            if self.train:
                dataset = self.proposal.dataset
                dataset = dataset.subset(indices)
                self.dataset.merge_(dataset)
                
                n_sample = len(self.dataset)
                if n_sample > 2 * self.max_dataset_size:
                    indices = [i for i in range(n_sample)]
                    random.shuffle(indices)
                    indices = indices[:self.max_dataset_size]
                    self.dataset = self.dataset.subset(indices)
                batch_size = int(self.batch_size * 20 / self.last_avg_size)
                log.info('formed a graph editing dataset of size %i' % len(self.dataset))
                loader = data.DataLoader(self.dataset,
                    batch_size=batch_size, shuffle=True,
                    collate_fn=GraphEditingDataset.collate_fn
                )
                
                train(
                    model=self.proposal.editor, 
                    loaders={'dev': loader}, 
                    optimizer=self.optimizer,
                    n_epoch=1,
                    log_every=10,
                    max_step=25,
                    metrics=[
                        'loss', 
                        'loss_del', 'prob_del',
                        'loss_add', 'prob_add',
                        'loss_arm', 'prob_arm'
                    ]
                )
                
                if not self.proposal.editor.device == \
                    torch.device('cpu'):
                    torch.cuda.empty_cache()


class Sampler_Recursive(Sampler):
    def __init__(self, *args):
        super().__init__(*args)
        
    def acc_rates(self, *args):
        acc_rates = []
        for i in range(self.num_path):
            A = 1.
            acc_rates.append(A)
        return acc_rates


class Sampler_Improve(Sampler):
    def __init__(self, *args):
        super().__init__(*args)

    def acc_rates(self, new_scores, old_scores, *args):
        acc_rates = []
        for i in range(self.num_path):
            A = new_scores[i] > old_scores[i]
            acc_rates.append(A)
        return acc_rates 


class Sampler_SA(Sampler):
    def __init__(self, *args):
        super().__init__(*args)
        self.k = 0
        self.step_cur_T = 0
        self.T = Sampler_SA.T_k(self.k)

    @staticmethod
    def T_k(k):
        T_0 = 1. #.1
        BETA = .05
        ALPHA = .95
        
        # return 1. * T_0 / (math.log(k + 1) + 1e-6)
        # return max(1e-6, T_0 - k * BETA)
        return ALPHA ** k * T_0

    def update_T(self):
        STEP_PER_T = 5
        if self.step_cur_T == STEP_PER_T:
            self.k += 1
            self.step_cur_T = 0
            self.T = Sampler_SA.T_k(self.k)
        else: self.step_cur_T += 1
        self.T = max(self.T, 1e-2)
        return self.T
        
    def acc_rates(self, new_scores, old_scores, *args):
        acc_rates = []
        T = self.update_T()
        # T = 1. / (4. * math.log(self.step + 8.))
        for i in range(self.num_path):
            # A = min(1., math.exp(1. * (new_scores[i] - old_scores[i]) / T))
            A = min(1., 1. * new_scores[i] / max(old_scores[i], 1e-6))
            A = min(1., A ** (1. / T))
            acc_rates.append(A)
        return acc_rates


class Sampler_MH(Sampler):
    def __init__(self, *args):
        super().__init__(*args)
        self.power = 30.
        
    def acc_rates(self, new_scores, old_scores, pops):
        acc_rates = []
        for i in range(self.num_path):
            old_score = max(old_scores[i], 1e-5)
            A = ((new_scores[i] / old_score) ** self.power) * pops[i]
            acc_rates.append(A)
        return acc_rates

