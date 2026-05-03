import copy
import time
from torch import optim, nn
from tqdm import tqdm
from utils.visualization import render_animation
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
                 dataset,
                 cfg,
                 logger,
                 tb_logger,
                 wandb_logger=None,
                 resume_ckpt=None,
                 start_epoch=0):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.val_noise_losses = None
        self.val_velocity_losses = None
        self.t_s = None
        self.train_losses = None
        self.train_noise_losses = None
        self.train_velocity_losses = None
        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger
        self.wandb_logger = wandb_logger

        self.iter = 0
        self.start_epoch = max(0, int(start_epoch))
        self.lrs = []
        self.best_val_loss = float('inf')

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

        if resume_ckpt:
            sd = torch.load(resume_ckpt, map_location=self.cfg.device)
            if self.cfg.ema is True:
                self.ema_model.load_state_dict(sd)
                self.model.load_state_dict(sd)
                # Past EMA warm-up so step_ema updates the average instead of resetting.
                self.ema.step = max(self.ema.step, 2000)
            else:
                self.model.load_state_dict(sd)
            self.logger.info('Loaded weights from {} (resume from epoch {})'.format(
                resume_ckpt, self.start_epoch))

    def compute_velocity_aux_loss(self, x_t, t, predicted_noise, traj):
        if getattr(self.cfg, 'velocity_loss_weight', 0.0) <= 0:
            return None

        alpha_hat = self.diffusion.alpha_hat[t][:, None, None]
        predicted_x0 = (x_t - torch.sqrt(1. - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
        traj_est = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], predicted_x0)
        pred_vel = motion_to_velocity(traj_est)
        gt_vel = motion_to_velocity(traj)
        return self.criterion(pred_vel, gt_vel)

    def loop(self):
        self.before_train()
        # Epoch progress bar
        epoch_pbar = tqdm(
            range(self.start_epoch, self.cfg.num_epoch),
            desc="Training",
            unit="epoch",
        )
        for self.iter in epoch_pbar:
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{self.train_losses.avg:.4f}',
                'val_loss': f'{self.val_losses.avg:.4f}',
                'lr': f'{self.lrs[-1]:.2e}'
            })
            
            # Log progress to wandb
            if self.wandb_logger is not None:
                progress = (self.iter + 1) / self.cfg.num_epoch * 100
                self.wandb_logger.log({'progress': progress}, step=self.iter)

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        # Align LR with completed epochs when resuming (avoids calling step() before any
        # optimizer.step(), which triggers PyTorch's UserWarning).
        sched_last = self.start_epoch - 1 if self.start_epoch > 0 else -1
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.milestone,
            gamma=self.cfg.gamma,
            last_epoch=sched_last,
        )
        self.criterion = nn.MSELoss()

    def before_train_step(self):
        self.model.train()
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.train_noise_losses = AverageMeter()
        self.train_velocity_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        num_batches = self.cfg.num_data_sample // self.cfg.batch_size
        pbar = tqdm(self.generator_train, total=num_batches, 
                    desc=f"Epoch {self.iter}", leave=False, unit="batch")
        for traj_np in pbar:
            with torch.no_grad():
                traj_np, traj_cond_np = split_motion_inputs(traj_np, self.cfg)
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_cond = tensor(traj_cond_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_cond_pad = padding_traj(traj_cond, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_cond_pad)

                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

            # train
            t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
            x_t, noise = self.diffusion.noise_motion(traj_dct, t)
            predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
            noise_loss = self.criterion(predicted_noise, noise)
            velocity_loss = self.compute_velocity_aux_loss(x_t, t, predicted_noise, traj)
            loss = noise_loss
            if velocity_loss is not None:
                loss = loss + self.cfg.velocity_loss_weight * velocity_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.train_noise_losses.update(noise_loss.item())
            self.train_velocity_losses.update(0.0 if velocity_loss is None else velocity_loss.item())
            if self.tb_logger is not None:
                self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)
                self.tb_logger.add_scalar('Loss/train_noise', noise_loss.item(), self.iter)
                if velocity_loss is not None:
                    self.tb_logger.add_scalar('Loss/train_velocity', velocity_loss.item(), self.iter)
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f'{self.train_losses.avg:.4f}',
                'noise': f'{self.train_noise_losses.avg:.4f}',
                'vel': f'{self.train_velocity_losses.avg:.4f}'
            })

            del loss, noise_loss, velocity_loss, traj, traj_cond, traj_dct, traj_dct_mod, traj_cond_pad, traj_np, traj_cond_np

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} Noise Loss: {} Velocity Loss: {} lr: {:.5f}'.format(
                self.iter,
                time.time() - self.t_s,
                self.train_losses.avg,
                self.train_noise_losses.avg,
                self.train_velocity_losses.avg,
                self.lrs[-1]
            ))
        # Log to wandb
        if self.wandb_logger is not None:
            self.wandb_logger.log({
                'epoch': self.iter,
                'train/loss': self.train_losses.avg,
                'train/noise_loss': self.train_noise_losses.avg,
                'train/velocity_loss': self.train_velocity_losses.avg,
                'train/lr': self.lrs[-1],
                'train/time': time.time() - self.t_s,
            }, step=self.iter)

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.val_noise_losses = AverageMeter()
        self.val_velocity_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size,
                                                                     aug=False)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        val_seed = self.cfg.seed + 12345
        np.random.seed(val_seed)
        torch.manual_seed(val_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(val_seed)

        try:
            for traj_np in self.generator_val:
                with torch.no_grad():
                    traj_np, traj_cond_np = split_motion_inputs(traj_np, self.cfg)
                    traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                    traj_cond = tensor(traj_cond_np, device=self.cfg.device, dtype=self.cfg.dtype)
                    traj_cond_pad = padding_traj(traj_cond, self.cfg.padding, self.cfg.idx_pad,
                                                 self.cfg.zero_index)
                    traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                    traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_cond_pad)

                    if np.random.random() > self.cfg.mod_test:
                        traj_dct_mod = None

                    t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                    x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                    predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
                    noise_loss = self.criterion(predicted_noise, noise)
                    velocity_loss = self.compute_velocity_aux_loss(x_t, t, predicted_noise, traj)
                    loss = noise_loss
                    if velocity_loss is not None:
                        loss = loss + self.cfg.velocity_loss_weight * velocity_loss

                    self.val_losses.update(loss.item())
                    self.val_noise_losses.update(noise_loss.item())
                    self.val_velocity_losses.update(0.0 if velocity_loss is None else velocity_loss.item())
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)
                        self.tb_logger.add_scalar('Loss/val_noise', noise_loss.item(), self.iter)
                        if velocity_loss is not None:
                            self.tb_logger.add_scalar('Loss/val_velocity', velocity_loss.item(), self.iter)

                del loss, noise_loss, velocity_loss, traj, traj_cond, traj_dct, traj_dct_mod, traj_cond_pad, traj_np, traj_cond_np
        finally:
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def after_val_step(self):
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Val Loss: {} Noise Loss: {} Velocity Loss: {}'.format(
                self.iter,
                time.time() - self.t_s,
                self.val_losses.avg,
                self.val_noise_losses.avg,
                self.val_velocity_losses.avg
            ))
        # Log to wandb
        if self.wandb_logger is not None:
            self.wandb_logger.log({
                'val/loss': self.val_losses.avg,
                'val/noise_loss': self.val_noise_losses.avg,
                'val/velocity_loss': self.val_velocity_losses.avg,
                'val/time': time.time() - self.t_s,
            }, step=self.iter)
        
        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                model_path = os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt")
                torch.save(self.ema_model.state_dict(), model_path)
            else:
                model_path = os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt")
                torch.save(self.model.state_dict(), model_path)
            
            # Log model artifact to wandb
            if self.wandb_logger is not None:
                self.wandb_logger.save(model_path)

        # Save the best model according to validation loss.
        if self.val_losses.avg < self.best_val_loss:
            self.best_val_loss = self.val_losses.avg
            if self.cfg.ema is True:
                best_model_path = os.path.join(self.cfg.model_path, "best_ema.pt")
                torch.save(self.ema_model.state_dict(), best_model_path)
            else:
                best_model_path = os.path.join(self.cfg.model_path, "best.pt")
                torch.save(self.model.state_dict(), best_model_path)

            self.logger.info(
                '====> New best model at epoch {}: val_loss={:.6f}, saved to {}'.format(
                    self.iter, self.best_val_loss, best_model_path
                )
            )
            if self.wandb_logger is not None:
                self.wandb_logger.log({'val/best_loss': self.best_val_loss}, step=self.iter)
                self.wandb_logger.save(best_model_path)
