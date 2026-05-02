import os
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def demo_visualize(mode, cfg, model, diffusion, dataset):
    """
    script for drawing gifs in different modes
    """
    if mode == 'pred':
        action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=cfg.vis_row)
            suffix = action_list[i]
            vis_azim = getattr(cfg, 'vis_azim', 0.0)
            vis_elev = getattr(cfg, 'vis_elev', 15.0)
            vis_axis_radius = getattr(cfg, 'vis_axis_radius', 2.5)
            vis_size = getattr(cfg, 'vis_size', 2.4)
            vis_dpi = getattr(cfg, 'vis_dpi', 160)
            vis_auto_axis = getattr(cfg, 'vis_auto_axis', True)
            vis_axis_padding = getattr(cfg, 'vis_axis_padding', 0.2)
            vis_line_width = getattr(cfg, 'vis_line_width', 2.0)
            vis_title_fontsize = getattr(cfg, 'vis_title_fontsize', 18)
            coord_order = (0, 2, 1) if cfg.dataset == 'harper3d' else (0, 1, 2)
            # Human-only models still draw GT robot for CHICO/HARPER; axis limits must use only
            # human joints or the arm spans the whole subplot and the person shrinks to a dot.
            axis_bbox_num_joints = (
                cfg.output_total_joints if getattr(cfg, 'predict_human_only', False) else None
            )
            render_animation(dataset['test'].skeleton, pose_gen, ['TransFusion'], cfg.t_his, ncol=cfg.vis_col + 2,
                             output=os.path.join(cfg.gif_dir, f'pred_{suffix}.gif'), mode=mode,
                             azim=vis_azim, elev=vis_elev, axis_radius=vis_axis_radius,
                             size=vis_size, dpi=vis_dpi, coord_order=coord_order,
                             auto_axis=vis_auto_axis, axis_padding=vis_axis_padding,
                             line_width=vis_line_width, title_fontsize=vis_title_fontsize,
                             axis_bbox_num_joints=axis_bbox_num_joints)

    else:
        raise NotImplementedError(f"sorry, {mode} is not only available.")  
