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
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    return cfg

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
        self.dataset = cfg.get('dataset', 'h36m')
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
        self.use_velocity_input = cfg.get('use_velocity_input', False)
        self.velocity_loss_weight = cfg.get('velocity_loss_weight', 0.0)

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

        self.num_layers = cfg['num_layers']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']

        self.dct_norm_enable = cfg['dct_norm_enable']

        # harper3d specific config
        self.data_path = cfg.get('data_path', './data/harper3d')
        self.include_spot = cfg.get('include_spot', True)
        self.predict_human_only = cfg.get('predict_human_only', False)
        self.use_spot_condition = cfg.get('use_spot_condition', self.include_spot)
        self.fps = cfg.get('fps', '30hz')
        self.harper3d_multimodal_dir = cfg.get('harper3d_multimodal_dir', '/data3/user/qkh/DATASET/TransFusion/HARPER')

        # indirect variable
        if self.dataset == 'h36m':
            self.joint_num = 16
            self.cond_joint_num = self.joint_num
            self.output_total_joints = self.joint_num + 1
        elif self.dataset == 'amass':
            self.joint_num = 21
            self.cond_joint_num = self.joint_num
            self.output_total_joints = self.joint_num + 1
        elif self.dataset == 'harper3d':
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
        else:
            self.joint_num = 14
            self.cond_joint_num = self.joint_num
            self.output_total_joints = self.joint_num + 1
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
