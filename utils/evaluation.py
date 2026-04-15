import csv
import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def compute_stats(diffusion, multimodal_dict, model, logger, cfg, wandb_logger=None):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    def get_prediction(data, model_select):
        traj_np, traj_cond_np = split_motion_inputs(data, cfg)
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj_cond = tensor(traj_cond_np, device=cfg.device, dtype=torch.float32)

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics', traj_cond=traj_cond)
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'ADE-m', 'FDE-m', 'MMADE-m', 'MMFDE-m', 'ADE-w', 'FDE-w', 'MMADE-w', 'MMFDE-w']
    stats_meter = {x: {y: AverageMeter() for y in ['TransFusion']} for x in stats_names}

    def _to_float(v):
        if hasattr(v, 'item'):
            return float(v.item())
        return float(v)

    K = 50
    eval_batch_size = max(1, min(int(getattr(cfg, 'eval_batch_size', cfg.batch_size)), num_samples))
    logger.info(f'Chunked eval enabled: eval_batch_size={eval_batch_size}, K={K}')

    for start in tqdm(range(0, num_samples, eval_batch_size), desc='Eval: sample chunks', unit='chunk'):
        end = min(start + eval_batch_size, num_samples)
        data_chunk = data_group[start:end]
        pred_chunk = []

        for _ in tqdm(range(0, K), desc='Eval: DDIM samples (K)', position=1, unit='round', leave=False):
            pred_i_nd = get_prediction(data_chunk, model)
            pred_chunk.append(pred_i_nd)

        pred_chunk = np.concatenate(pred_chunk, axis=0)
        pred_chunk = pred_chunk[:, :, cfg.t_his:, :]

        for local_idx, sample_idx in enumerate(
            tqdm(
                range(start, end),
                desc='Eval: metrics per sample',
                unit='sample',
                leave=False,
                position=1,
            )
        ):
            pred_sample = torch.from_numpy(pred_chunk[:, local_idx, :, :]).to(cfg.device)
            gt_sample = torch.from_numpy(gt_group[sample_idx][np.newaxis, ...]).to(cfg.device)
            apd, ade, fde, mmade, mmfde, ade_m, fde_m, mmade_m, mmfde_m, ade_w, fde_w, mmade_w, mmfde_w = compute_all_metrics(
                pred_sample,
                gt_sample,
                traj_gt_arr[sample_idx]
            )
            stats_meter['APD']['TransFusion'].update(apd)
            stats_meter['ADE']['TransFusion'].update(ade)
            stats_meter['FDE']['TransFusion'].update(fde)
            stats_meter['MMADE']['TransFusion'].update(mmade)
            stats_meter['MMFDE']['TransFusion'].update(mmfde)
            stats_meter['ADE-m']['TransFusion'].update(ade_m)
            stats_meter['FDE-m']['TransFusion'].update(fde_m)
            stats_meter['MMADE-m']['TransFusion'].update(mmade_m)
            stats_meter['MMFDE-m']['TransFusion'].update(mmfde_m)
            stats_meter['ADE-w']['TransFusion'].update(ade_w)
            stats_meter['FDE-w']['TransFusion'].update(fde_w)
            stats_meter['MMADE-w']['TransFusion'].update(mmade_w)
            stats_meter['MMFDE-w']['TransFusion'].update(mmfde_w)

        del pred_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for stats in stats_names:
        str_stats = f'{stats}: ' + ' '.join(
            [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
        )
        logger.info(str_stats)
    if wandb_logger is not None:
        eval_metrics = {
            f'eval/{stats}': _to_float(stats_meter[stats]['TransFusion'].avg)
            for stats in stats_names
        }
        wandb_logger.log(eval_metrics)
        wandb_logger.summary.update(eval_metrics)

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['TransFusion'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['TransFusion'] = new_meter['TransFusion'].cpu().numpy()
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['TransFusion']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)

    if wandb_logger is not None:
        wandb_logger.save(file_latest % cfg.result_dir)
        wandb_logger.save(file_stat % cfg.result_dir)