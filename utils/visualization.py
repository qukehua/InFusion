import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from vpython import *
import time
from datetime import datetime


def render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0,
                     elev=15.0, axis_radius=1.5, output=None, mode='pred',
                     size=2, ncol=5, bitrate=3000, dpi=80, coord_order=(0, 1, 2),
                     auto_axis=True, axis_padding=0.2, line_width=2.0,
                     title_fontsize=18, camera_dist=5.0, axis_bbox_num_joints=None,
                     use_legacy_visualization=False):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).

    use_legacy_visualization:
      True  -> Harper3D/original: global axis_params per panel, double continue (n==0 and n%ncol==0),
               fixed axis limits from compute_axis_params (full sequence).
      False -> CHICO-tuned: optional axis_bbox_num_joints, per-frame axis limits, freeze only n==0 context.
    """
    all_poses = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    axis_radius = float(axis_radius)
    coord_order = tuple(coord_order)

    def project_pose(pos):
        return pos[..., coord_order]

    def compute_axis_params(pose_values):
        params = []
        for pose in pose_values:
            projected = project_pose(pose)
            flat = projected.reshape(-1, 3)
            mins = flat.min(axis=0)
            maxs = flat.max(axis=0)
            center = (mins + maxs) / 2.0
            radius = max((maxs - mins).max() / 2.0 + axis_padding, 1e-3)
            params.append((center, radius))
        return params

    def axis_params_one_frame(single_frame_joints):
        """Per-frame bbox (CHICO); optional clip to first K joints for human-only axis scale."""
        jnts = single_frame_joints
        if axis_bbox_num_joints is not None:
            jnts = single_frame_joints[:axis_bbox_num_joints]
        projected = project_pose(jnts)
        flat = projected.reshape(-1, 3)
        mins = flat.min(axis=0)
        maxs = flat.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = max((maxs - mins).max() / 2.0 + axis_padding, 1e-3)
        return center, radius

    pose_values = list(poses.values())
    projected_poses = [project_pose(pose) for pose in pose_values]
    axis_params = compute_axis_params(pose_values)

    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index+1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        if auto_axis:
            if use_legacy_visualization:
                center, radius = axis_params[index]
            else:
                center, radius = axis_params_one_frame(projected_poses[index][0])
            ax.set_xlim3d([center[0] - radius, center[0] + radius])
            ax.set_ylim3d([center[1] - radius, center[1] + radius])
            ax.set_zlim3d([center[2] - radius, center[2] + radius])
        else:
            ax.set_xlim3d([-axis_radius, axis_radius])
            ax.set_ylim3d([-axis_radius, axis_radius])
            ax.set_zlim3d([-axis_radius, axis_radius])
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect([1, 1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = camera_dist
        if index == 0 or index == 1:
            ax.set_title(title, y=1.0, fontsize=title_fontsize)
        elif index > 1 and index <= 11:
            ax.set_title(f'pred #{index-1}', y=1.0, fontsize=title_fontsize)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(projected_poses[index][:, 0])
    fig.tight_layout(h_pad=15,w_pad=15)
    fig.subplots_adjust(wspace=-0.4, hspace=0.4)
    poses = pose_values

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_mcol, hist_rcol = 'gray', 'black', 'red'
    pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'

    base_bone_pairs = skeleton.links() if hasattr(skeleton, 'links') else [
        (j, j_parent) for j, j_parent in enumerate(skeleton.parents()) if j_parent != -1
    ]
    bone_pairs_per_pose = []
    for pose in poses:
        joint_num = pose.shape[1]
        bone_pairs_per_pose.append(
            [(j, j_parent) for j, j_parent in base_bone_pairs if j < joint_num and j_parent < joint_num]
        )

    def update_video(i):
        nonlocal initialized
        if i < t_hist:
            lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
        else:
            lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if use_legacy_visualization:
                # Original Harper/studio logic: two independent skips (see legacy repo).
                if fix_0 and n == 0 and i >= t_hist:
                    continue
                if fix_0 and n % ncol == 0 and i >= t_hist:
                    continue
                trajectories[n] = projected_poses[n][:, 0]
                if auto_axis:
                    center, radius = axis_params[n]
                    ax.set_xlim3d([center[0] - radius, center[0] + radius])
                    ax.set_ylim3d([center[1] - radius, center[1] + radius])
                    ax.set_zlim3d([center[2] - radius, center[2] + radius])
                else:
                    ax.set_xlim3d([-axis_radius + trajectories[n][i, 0], axis_radius + trajectories[n][i, 0]])
                    ax.set_ylim3d([-axis_radius + trajectories[n][i, 1], axis_radius + trajectories[n][i, 1]])
                    ax.set_zlim3d([-axis_radius + trajectories[n][i, 2], axis_radius + trajectories[n][i, 2]])
            else:
                frame_for_axis = i
                if fix_0 and n == 0 and i >= t_hist:
                    frame_for_axis = t_hist - 1
                trajectories[n] = projected_poses[n][:, 0]
                if auto_axis:
                    center, radius = axis_params_one_frame(projected_poses[n][frame_for_axis])
                    ax.set_xlim3d([center[0] - radius, center[0] + radius])
                    ax.set_ylim3d([center[1] - radius, center[1] + radius])
                    ax.set_zlim3d([center[2] - radius, center[2] + radius])
                else:
                    ax.set_xlim3d([-axis_radius + trajectories[n][i, 0], axis_radius + trajectories[n][i, 0]])
                    ax.set_ylim3d([-axis_radius + trajectories[n][i, 1], axis_radius + trajectories[n][i, 1]])
                    ax.set_zlim3d([-axis_radius + trajectories[n][i, 2], axis_radius + trajectories[n][i, 2]])

        if not initialized:

            for n, ax in enumerate(ax_3d):
                pos = projected_poses[n][i]
                for j, j_parent in bone_pairs_per_pose[n]:
                    if j in skeleton.joints_right():
                        col = rcol
                    elif j in skeleton.joints_left():
                        col = lcol
                    else:
                        col = mcol

                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=line_width))
            initialized = True
        else:

            for n, ax in enumerate(ax_3d):
                if use_legacy_visualization:
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    if fix_0 and n % ncol == 0 and i >= t_hist:
                        continue
                else:
                    if fix_0 and n == 0 and i >= t_hist:
                        continue

                pos = projected_poses[n][i]
                for bone_idx, (j, j_parent) in enumerate(bone_pairs_per_pose[n]):
                    if j in skeleton.joints_right():
                        col = rcol
                    elif j in skeleton.joints_left():
                        col = lcol
                    else:
                        col = mcol
                    x_array = np.array([pos[j, 0], pos[j_parent, 0]])
                    y_array = np.array([pos[j, 1], pos[j_parent, 1]])
                    z_array = np.array([pos[j, 2], pos[j_parent, 2]])
                    lines_3d[n][bone_idx][0].set_data_3d(x_array, y_array, z_array)
                    lines_3d[n][bone_idx][0].set_color(col)


    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses, projected_poses, axis_params, bone_pairs_per_pose
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        if x[0] in {'gt', 'context'}:
            for ax, title in zip(ax_3d, poses.keys()):
                ax.set_title(title, y=1.0, fontsize=title_fontsize)
        if mode == 'switch':
            if x[0] in {algo + '_0'}:
                for ax, title in zip(ax_3d, poses.keys()):
                    ax.set_title('target', y=1.0, fontsize=12)
        
        poses = list(poses.values())
        projected_poses = [project_pose(pose) for pose in poses]
        axis_params = compute_axis_params(poses)
        bone_pairs_per_pose = []
        for pose in poses:
            joint_num = pose.shape[1]
            bone_pairs_per_pose.append(
                [(j, j_parent) for j, j_parent in base_bone_pairs if j < joint_num and j_parent < joint_num]
            )

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        os.makedirs('out_svg', exist_ok=True)
        suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        os.makedirs('out_svg_' + suffix, exist_ok=True)
        for algo in algos:
            reload_poses()
            for i in range(0, t_total + 1, 10):
                if i == 0:
                    update_video(0)
                else:
                    update_video(i - 1)
                fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 50
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=dpi, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    save()
    show_animation()
    plt.show()
    plt.close()
