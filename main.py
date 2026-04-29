import argparse
import sys
import os
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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run 'pip install wandb' to enable wandb logging.")


def _to_wandb_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_wandb_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_wandb_safe(v) for k, v in value.items()}
    return str(value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='chico', help='config id, e.g. chico or harper3d_30hz')
    parser.add_argument('--exp_name', type=str, default=None, help='custom experiment folder name')
    parser.add_argument('--mode', default='train', help='train / eval / pred')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
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
    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))
    seed_set(cfg.seed)

    dataset, dataset_multi_test = dataset_split(cfg)


    """logger"""
    tb_logger = None if cfg.disable_tensorboard else SummaryWriter(cfg.tb_dir)
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
            'seed': cfg.seed,
        }
        run_name = args.wandb_name if args.wandb_name else (cfg.exp_name if cfg.exp_name else f"{cfg.dataset}_{cfg.seed}")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            dir=cfg.cfg_dir,
        )
        cfg_for_wandb = {}
        for k, v in cfg.__dict__.items():
            if 'dct_m' in k or 'idct_m' in k:
                continue
            cfg_for_wandb[k] = _to_wandb_safe(v)
        wandb.config.update(cfg_for_wandb, allow_val_change=True)
        cfg_file = os.path.join('cfg', f'{args.cfg}.yml')
        if os.path.exists(cfg_file):
            wandb.save(cfg_file, policy='now')
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
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(diffusion, multimodal_dict, model, logger, cfg, wandb_logger=wandb_logger)
        if wandb_logger is not None:
            wandb.finish()

    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
        if wandb_logger is not None:
            wandb.finish()
