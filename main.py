import os
import rdkit
import torch
import random
import pathlib
import argparse
import numpy as np
import logging as log
from tqdm import tqdm
from rdkit import Chem, RDLogger

from .datasets.utils import load_mols
from .evaluator.scorer import Scorer
from .evaluator.evaluator import Evaluator
from .evaluator.measures import Measure, Diversity, SumBottleneck, NFG, NCircles
from .proposal.models.editor_basic import BasicEditor
from .proposal.proposal import Proposal_Editor
from .sampler import Sampler_SA

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument('--debug',      action='store_true')
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--run_exist',  action='store_true')
    parser.add_argument('--root_dir',   type=str,   default='MARS')
    parser.add_argument('--data_dir',   type=str,   default='data')
    parser.add_argument('--run_dir',    type=str,   default='runs/debug')
    parser.add_argument('--editor_dir', type=str,   default=None)
    parser.add_argument('--mols_init',  type=str,   default=None)
    # parser.add_argument('--mols_refe',  type=str,   default='actives_gsk3b,jnk3.txt')
    parser.add_argument('--vocab',      type=str,   default='chembl_func')
    parser.add_argument('--vocab_size', type=int,   default=1000)
    parser.add_argument('--max_size',   type=int,   default=40)
    parser.add_argument('--num_path',   type=int,   default=5000)
    parser.add_argument('--num_step',   type=int,   default=5000)
    parser.add_argument('--num_runs',   type=int,   default=10)
    parser.add_argument('--log_every',  type=int,   default=1)

    parser.add_argument('--sampler',    type=str,   default='sa')
    parser.add_argument('--proposal',   type=str,   default='editor')
    parser.add_argument('--objectives', type=str,   default='gsk3b,jnk3,qed,sa')
    parser.add_argument('--nov_term',   type=str,   default='Measure') # default as a dummy measure
    parser.add_argument('--nov_coef',   type=float, default=1.)
    
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--dataset_size',   type=int,   default=50000)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--n_atom_feat',    type=int,   default=17)
    parser.add_argument('--n_bond_feat',    type=int,   default=5)
    parser.add_argument('--n_node_hidden',  type=int,   default=64)
    parser.add_argument('--n_edge_hidden',  type=int,   default=128)
    parser.add_argument('--n_layers',       type=int,   default=6)
    args = parser.parse_args()
    if args.debug: args.num_runs, args.num_path, args.num_step = 1, 100, 10
    if args.run_dir == 'runs/debug': args.run_exist = True

    config = vars(args)
    config['device'] = torch.device(config['device'])
    config['run_dir'] = os.path.join(config['root_dir'], config['run_dir'])
    config['data_dir'] = os.path.join(config['root_dir'], config['data_dir'])
    config['objectives'] = config['objectives'].split(',')
    config['score_wght'], config['score_succ'], config['score_clip'] = {}, {}, {}
    for obj in config['objectives']:
        if   obj == 'gsk3b' or obj == 'jnk3': wght, succ, clip = 1.0, 0.5, 0.6
        elif obj == 'qed'                   : wght, succ, clip = 1.0, 0.6, 0.7
        elif obj == 'sa'                    : wght, succ, clip = 1.0, .67, 0.7
        config['score_wght'][obj] = wght
        config['score_succ'][obj] = succ
        config['score_clip'][obj] = clip
    os.makedirs(config['run_dir'], exist_ok=config['run_exist'])
    log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
    log.getLogger().addHandler(log.FileHandler(os.path.join(config['run_dir'], 'log.txt'), mode='w'))
    log.info(str(config))

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)


    for run in range(config['num_runs']):
        run_dir = os.path.join(config['run_dir'], 'run_%02d' % run)
        log.info('Run %02d: ======================' % run)
        
        ### proposal
        editor = BasicEditor(config).to(config['device']) \
            if not config['proposal'] == 'random' else None
        if config['editor_dir'] is not None: # load pre-trained editor
            path = os.path.join(config['root_dir'], config['editor_dir'], 'model_best.pt')
            editor.load_state_dict(torch.load(path, map_location=torch.device(config['device'])))
            print('successfully loaded editor model from %s' % path)
        proposal = Proposal_Editor(config, editor)

        ### evaluator
        scorer = Scorer(config)
        measure = eval(config['nov_term'])()
        evaluator = Evaluator(config, scorer, measure)

        ### sampler
        if config['mols_init']:
            mols = load_mols(config['data_dir'], config['mols_init'])
            mols = random.choices(mols, k=config['num_path'])
            mols_init = mols[:config['num_path']]
        else: mols_init = [
            Chem.MolFromSmiles('CC') for _ in range(config['num_path'])]
        sampler = Sampler_SA(config, run_dir, proposal, evaluator)
        sampler.sample(mols_init)
