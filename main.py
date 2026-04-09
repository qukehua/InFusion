import argparse
import sys
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *
import numpy as np
sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats

from data_loader.dataset_amass import DatasetAMASS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run 'pip install wandb' to enable wandb logging.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='h36m', help='h36m or humaneva or amass or HARPER')
    parser.add_argument('--exp_name', type=str, default=None, help='custom experiment folder name')
    parser.add_argument('--mode', default='train', help='train / eval / pred')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--milestone', type=list, default=None)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--save_model_interval', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/h36m_ckpt.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_col', type=int, default=10)
    parser.add_argument('--vis_row', type=int, default=3)
    # wandb arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='TransFusion', help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)
    # seed_set(6) 

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    if cfg.dataset == 'amass':
        # Avoid loading the large AMASS train split for eval/pred to save time & RAM.
        if args.mode == 'train':
            dataset = {'train': DatasetAMASS('train'), 'test': DatasetAMASS('test')}
        else:
            dataset = {'test': DatasetAMASS('test')}
    else:
        dataset, dataset_multi_test = dataset_split(cfg)


    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    
    # Initialize wandb
    wandb_logger = None
    if args.use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            'dataset': cfg.dataset,
            't_his': cfg.t_his,
            't_pred': cfg.t_pred,
            'batch_size': cfg.batch_size,
            'num_epoch': cfg.num_epoch,
            'lr': cfg.lr,
            'n_pre': cfg.n_pre,
            'noise_steps': cfg.noise_steps,
            'ddim_timesteps': cfg.ddim_timesteps,
            'scheduler': cfg.scheduler,
            'num_layers': cfg.num_layers,
            'num_heads': cfg.num_heads,
            'latent_dims': cfg.latent_dims,
            'dropout': cfg.dropout,
            'seed': args.seed,
        }
        run_name = args.wandb_name if args.wandb_name else (cfg.exp_name if cfg.exp_name else f"{cfg.dataset}_{args.seed}")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            dir=cfg.cfg_dir,
        )
        wandb_logger = wandb
        logger.info(f"wandb initialized: {args.wandb_project}/{run_name}")
    
    """model"""
    model, diffusion = create_model_and_diffusion(cfg)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            logger=logger,
            tb_logger=tb_logger,
            wandb_logger=wandb_logger)
        trainer.loop()
        
        # Finish wandb run
        if wandb_logger is not None:
            wandb.finish()

    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        if cfg.dataset == 'amass':
            multimodal_dict = get_multimodal_gt_full(logger, dataset['test'], args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(diffusion, multimodal_dict, model, logger, cfg)

    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
