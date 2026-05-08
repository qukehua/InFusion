import yaml
import os
from utils import util, torch, generate_pad


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        if v is None and hasattr(cfg, k):
            continue
        setattr(cfg, k, v)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    exp_base_name = getattr(cfg, 'exp_name', None) or args_dict['cfg']
    if args_dict['mode'] == ('train' or 'pred' or 'eval'):
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, exp_base_name)
    else:
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode'])
    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    # Keep visualization outputs in a fixed project folder for easier browsing.
    if args_dict['mode'] == 'pred':
        cli_gif = args_dict.get('gif_dir')
        if cli_gif:
            cfg.gif_dir = os.path.abspath(cli_gif)
        elif getattr(cfg, 'gif_dir', None):
            cfg.gif_dir = os.path.abspath(cfg.gif_dir)
        else:
            cfg.gif_dir = os.path.join(os.getcwd(), 'images')
    else:
        cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    return cfg


def parse_comad_test_interactions(raw):
    """
    CoMad path: <split>/<action>/<HH|HR>/<id>/data.json
    Returns None (load all) or a set of interaction folder names, e.g. {'HR'}.
    """
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return None
    if isinstance(raw, (list, tuple)):
        out = {str(x).strip().upper() for x in raw if str(x).strip()}
    else:
        s = str(raw).strip()
        if not s or s.lower() == "all":
            return None
        if " " in s or "," in s:
            out = {p.upper() for p in s.replace(",", " ").split() if p.strip()}
        else:
            out = {s.upper()}
    allowed = {"HH", "HR"}
    unknown = out - allowed
    if unknown:
        raise ValueError(
            f"comad_test_interactions must be a subset of {allowed}, got unknown labels {unknown}."
        )
    return out or None


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'inference' if test else 'results'
        os.makedirs(self.base_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'chico')
        self.exp_name = cfg.get('exp_name', None)
        self.seed = cfg.get('seed', 0)
        self.batch_size = cfg['batch_size']
        self.eval_batch_size = cfg.get('eval_batch_size', self.batch_size)
        self.normalize_data = cfg.get('normalize_data', False)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']

        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']
        self.milestone = cfg.get('milestone', [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
        self.gamma = cfg.get('gamma', 0.8)
        self.save_model_interval = cfg.get('save_model_interval', 100)
        self.disable_tensorboard = cfg.get('disable_tensorboard', False)
        self.use_data_aug = cfg.get('use_data_aug', False)
        self.aug_rotate_prob = cfg.get('aug_rotate_prob', 0.5)
        self.aug_reverse_prob = cfg.get('aug_reverse_prob', 0.3)
        self.vis_azim = cfg.get('vis_azim', 0.0)
        self.vis_elev = cfg.get('vis_elev', 15.0)
        self.vis_axis_radius = cfg.get('vis_axis_radius', 2.5)
        self.vis_size = cfg.get('vis_size', 2.4)
        self.vis_dpi = cfg.get('vis_dpi', 160)
        self.vis_auto_axis = cfg.get('vis_auto_axis', True)
        self.vis_axis_padding = cfg.get('vis_axis_padding', 0.2)
        self.vis_line_width = cfg.get('vis_line_width', 2.0)
        self.vis_title_fontsize = cfg.get('vis_title_fontsize', 18)
        self.vis_sample_strategy = cfg.get('vis_sample_strategy', 'random')
        self.vis_candidate_k = cfg.get('vis_candidate_k', 10)
        self.vis_skip_attach_robot = cfg.get('vis_skip_attach_robot', False)
        self.vis_output_only = cfg.get('vis_output_only', False)
        self.use_velocity_input = cfg.get('use_velocity_input', False)
        self.velocity_loss_weight = cfg.get('velocity_loss_weight', 0.0)

        # pred mode: optional default GIF output dir (CLI --gif_dir overrides)
        self.gif_dir = cfg.get('gif_dir')

        self.n_pre = cfg['n_pre']
        self.multimodal_path = cfg.get('multimodal_path', None)
        self.data_candi_path = cfg.get('data_candi_path', None)

        self.padding = cfg['padding']
        self.Complete = cfg['Complete']
        self.noise_steps = cfg['noise_steps']
        self.ddim_timesteps = cfg['ddim_timesteps']
        self.scheduler = cfg['scheduler']
        self.model_variant = cfg.get('model_variant', 'default')
        self.stage1_num_layers = cfg.get('stage1_num_layers', None)
        self.dit_attn_mode = cfg.get('dit_attn_mode', 'spatio_temporal')

        self.num_layers = cfg['num_layers']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']

        self.dct_norm_enable = cfg['dct_norm_enable']

        # dataset-specific config
        self.data_path = cfg.get('data_path', './data/harper3d')
        self.include_spot = cfg.get('include_spot', True)
        self.include_robot = cfg.get('include_robot', True)
        self.include_person2 = cfg.get('include_person2', True)
        self.predict_human_only = cfg.get('predict_human_only', False)
        self.use_spot_condition = cfg.get('use_spot_condition', self.include_spot)
        self.use_robot_condition = cfg.get('use_robot_condition', self.include_robot)
        self.use_hr_robot_condition = cfg.get('use_hr_robot_condition', self.use_robot_condition)
        self.use_hh_human_condition = cfg.get('use_hh_human_condition', self.include_person2)
        self.fps = cfg.get('fps', '30hz')
        self.harper3d_multimodal_dir = cfg.get('harper3d_multimodal_dir', '/data3/user/qkh/DATASET/TransFusion/HARPER')
        self.scene_filter_train = cfg.get('scene_filter_train', None)
        self.scene_filter_test = cfg.get('scene_filter_test', None)
        self.require_two_person = cfg.get('require_two_person', True)
        self.cmu_scene_filter = cfg.get('cmu_scene_filter', None)
        self.cmu_file_filter = cfg.get('cmu_file_filter', None)

        # indirect variable
        if self.dataset != "comad":
            self.comad_test_interactions = None

        if self.dataset == 'harper3d':
            # Harper3D can use robot joints as conditioning while predicting only
            # human motion. We therefore track output and conditioning sizes separately.
            self.total_joint_num = 44 if self.include_spot else 21
            self.output_total_joints = 21 if self.predict_human_only else self.total_joint_num
            self.joint_num = self.output_total_joints - 1
            self.cond_joint_num = self.total_joint_num - 1 if self.use_spot_condition else self.joint_num
            
            # Auto-construct multimodal paths based on fps if not explicitly set
            if self.multimodal_path is None or self.multimodal_path == 'auto':
                self.multimodal_path = os.path.join(
                    self.harper3d_multimodal_dir,
                    f't_his{self.t_his}_{self.fps}_thre0.500_t_pred{self.t_pred}_thre0.100_filtered.npz'
                )
            if self.data_candi_path is None or self.data_candi_path == 'auto':
                self.data_candi_path = os.path.join(
                    self.harper3d_multimodal_dir,
                    f'data_candi_{self.fps}_t_his{self.t_his}_t_pred{self.t_pred}_skiprate20.npz'
                )
        elif self.dataset == 'chico':
            # Match CHICO-PoseForecasting: train/eval on normal actions only (no *_CRASH.pkl).
            self.chico_exclude_crash = cfg.get('chico_exclude_crash', True)
            # CHICO: 15 human joints + 9 robot joints.
            self.total_joint_num = 24 if self.include_robot else 15
            self.output_total_joints = 15 if self.predict_human_only else self.total_joint_num
            self.joint_num = self.output_total_joints - 1
            self.cond_joint_num = self.total_joint_num - 1 if self.use_robot_condition else self.joint_num
        elif self.dataset == 'comad':
            self.comad_p1_joints = 25
            self.comad_p2_joints = 25 if self.include_person2 else 0
            self.comad_robot_joints = 12 if self.include_robot else 0
            self.total_joint_num = self.comad_p1_joints + self.comad_p2_joints + self.comad_robot_joints
            self.output_total_joints = self.comad_p1_joints if self.predict_human_only else self.total_joint_num
            self.joint_num = self.output_total_joints - 1
            use_scene_condition = self.use_hr_robot_condition or self.use_hh_human_condition
            self.cond_joint_num = self.total_joint_num - 1 if use_scene_condition else self.joint_num
            # Test / val / pred / multimodal eval: restrict to path segment HH or HR (train always loads all).
            self.comad_test_interactions = parse_comad_test_interactions(cfg.get("comad_test_interactions", "all"))
            # Pred GIF: apply InteRACT comad_hr.py Motive coordinate fix (-x, z, y). HR scenes match paper;
            # set false if HH-only viz looks better in raw Motive axes.
            self.comad_motive_to_interact_axes = cfg.get("comad_motive_to_interact_axes", False)
        elif self.dataset == '3dpw':
            # 3DPW: two persons, each 24 SMPL joints (total 48).
            self.total_joint_num = 48
            self.output_total_joints = self.total_joint_num
            self.joint_num = self.output_total_joints - 1
            self.cond_joint_num = self.joint_num
        elif self.dataset == 'cmu_mocap':
            # CMU loader synthesizes Person_2 from Person_1; default single person has 39 joints.
            # We keep this configurable in case txt layout differs.
            self.total_joint_num = cfg.get('cmu_total_joint_num', 78)
            self.output_total_joints = self.total_joint_num
            self.joint_num = self.output_total_joints - 1
            self.cond_joint_num = self.joint_num
        else:
            raise ValueError(
                f"Unsupported dataset '{self.dataset}'. Supported datasets are 'harper3d', 'chico', 'comad', '3dpw', and 'cmu_mocap'."
            )
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
