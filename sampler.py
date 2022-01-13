import os
import math
import torch
import random
import logging as log
from tqdm import tqdm
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from .utils.train import train
from .utils.logger import MolsPrinter
from .datasets.datasets import GraphEditingDataset


class Sampler():
    def __init__(self, config, run_dir, proposal, evaluator):
        self.run_dir = run_dir
        self.proposal = proposal
        self.evaluator = evaluator
        self.writer = SummaryWriter(log_dir=run_dir)
        self.mols_printer = MolsPrinter(run_dir)
        
        ### for sampling
        self.step = None
        self.num_path = config['num_path']
        self.num_step = config['num_step']
        self.log_every = config['log_every']
        self.nov_coef = config['nov_coef']

        ### for editing and training
        self.train = config['train']
        self.last_avg_size = None
        self.batch_size = config['batch_size']
        if self.train:
            self.dataset = GraphEditingDataset()
            self.max_dataset_size = config['dataset_size']
            self.optimizer = torch.optim.Adam(self.proposal.editor.parameters(), lr=config['lr'])

    def record(self, step, mols, scores, dicts, acc_rates):
        ### average size
        sizes = [mol.GetNumAtoms() for mol in mols]
        avg_size = sum(sizes) / len(mols)
        self.last_avg_size = avg_size

        ### average score
        avg_score = 1. * sum(scores) / len(scores)      
        avg_score_dict = {}
        for k in dicts[0].keys():
            v = [dictt[k] for dictt in dicts]
            avg_score_dict[k] = sum(v) / len(v)
        
        ### evaluation results
        metrics, succ_dict = self.evaluator.get_results()

        ### logging and writing tensorboard
        log.info('=' * 20 + 'Step: {:02d}:'.format(step))
        log.info('\tAvg weighted score: ' + str(avg_score))
        log.info('\tAvg molecule size: ' + str(avg_size))
        log.info('\tMetrics: ' + str(metrics))
        log.info('\tSuccess dict: ' + str(succ_dict))
        log.info('\tAvg score: ' + str(avg_score_dict))
        self.writer.add_scalar('avg_score', avg_score, step)
        self.writer.add_scalar('avg_size', avg_size, step)
        self.writer.add_scalars('metrics', metrics, step)
        self.writer.add_scalars('succ_dict', succ_dict, step)
        self.writer.add_scalars('avg_score_dict', avg_score_dict, step)
        self.writer.add_histogram('acc_rates', torch.tensor(acc_rates), step)
        self.writer.add_histogram('scores', torch.tensor(scores), step)
        for k in dicts[0].keys():
            prop_scores = [score_dict[k] for score_dict in dicts]
            self.writer.add_histogram(k, torch.tensor(prop_scores), step)
        self.mols_printer.print_mols(step, mols, scores, dicts)
        
        ### early stop
        # TODO
        
    def acc_rates(self, new_scores, old_scores, nov_scores):
        '''
        compute sampling acceptance rates
        @params:
            new_scores : scores of new proposed molecules
            old_scores : scores of old molcules
            nov_scores : novelty scores of new proposed molecules
        '''
        raise NotImplementedError

    def sample(self, mols_init):
        '''
        sample molecules from initial ones
        @params:
            mols_init : initial molecules
        '''
        
        ### sample
        old_mols = [mol for mol in mols_init]
        old_scores, old_dicts = self.evaluator.get_scores(old_mols)
        acc_rates = [0. for _ in old_mols]
        self.evaluator.update(old_mols, old_dicts)
        self.record(-1, old_mols, old_scores, old_dicts, acc_rates)

        for step in range(self.num_step):
            self.step = step
            new_mols = self.proposal.propose(old_mols)
            new_indices = [i for i in range(self.num_path) if new_mols[i] is not None]
            new_mols   = [new_mols[i]   for i in new_indices]
            cmp_scores = [old_scores[i] for i in new_indices]
            cmp_dicts  = [old_dicts[i]  for i in new_indices]
            new_scores, new_dicts = self.evaluator.get_scores(new_mols)
            nov_scores = self.evaluator.novelty(new_mols)
            
            train_indices = [i for j, i in enumerate(new_indices) if new_scores[j] > cmp_scores[j]]
            # with open(os.path.join(self.run_dir, 'edits.txt'), 'a') as f:
            #     f.write('edits at step %i\n' % step)
            #     f.write('improve\tact\tarm\n')
            #     for i, item in enumerate(self.proposal.dataset):
            #         _, edit = item
            #         improve = new_scores[i] > old_scores[i]
            #         f.write('%i\t%i\t%i\n' % (improve, edit['act'], edit['arm']))
            
            acc_rates = self.acc_rates(new_scores, cmp_scores, nov_scores)
            acc_rates = [min(1., max(0., A)) for A in acc_rates]
            updated_mols, updated_dicts = [], []
            for j, i in enumerate(new_indices):
                A = acc_rates[j] # A = p(x') * g(x|x') / p(x) / g(x'|x)
                if random.random() > A: continue
                old_mols[i] = new_mols[j]
                old_scores[i] = new_scores[j]
                old_dicts[i] = new_dicts[j]
                updated_mols.append(new_mols[j])
                updated_dicts.append(new_dicts[j])

            self.evaluator.update(updated_mols, updated_dicts)
            if step % self.log_every == 0:
                self.record(step, old_mols, old_scores, old_dicts, acc_rates)
            if self.train: self.train_editor(train_indices)

    def train_editor(self, indices):
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


class Sampler_SA(Sampler):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.k = 0
        self.step_cur_T = 0
        self.T = Sampler_SA.T_k(self.k)

    @staticmethod
    def T_k(k):
        T_0 = 1. 
        ALPHA = .95
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
        
    def acc_rates(self, new_scores, old_scores, nov_scores, *args):
        acc_rates = []
        T = self.update_T()
        for i in range(len(new_scores)):
            new_score = new_scores[i] + self.nov_coef * nov_scores[i]
            A = new_score / max(old_scores[i], 1e-6)
            A = min(1, max(A, 0))
            A = min(1., A ** (1. / T))
            acc_rates.append(A)
        return acc_rates
