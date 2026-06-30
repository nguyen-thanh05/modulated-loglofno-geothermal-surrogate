import copy
import math
import os

import numpy as np
import torch

from models.aux_head import AuxHead
from training.checkpointing import CheckpointManager
from training.config import build_run_config
from training.constants import WELL_COORDS
from training.data_setup import build_data_loaders
from training.logging_utils import TrainingLogger
from training.loss_computation import LossComputer
from training.model_adapters import create_adapter
from training.model_factory import create_model, get_hidden_dim, get_out_channels
from training.physics import add_adaptive_noise
from training.utils import get_device, set_seed, update_ema


class Trainer:
    def __init__(self, run_config, *, resume_path=None):
        self.cfg = run_config
        self.data = build_data_loaders(run_config)
        self.device = get_device()
        self.checkpoints = CheckpointManager(
            run_config.checkpoints,
            resume_path=resume_path,
            device=self.device,
        )
        self.logger = TrainingLogger(run_config)

        model_cfg = run_config.model.raw
        model_type = run_config.model.type
        self.model = create_model(model_cfg, model_type).to(self.device)

        self.aux_head_model = AuxHead(
            state_channels=get_out_channels(model_cfg),
            depth=16,
            aux_channels=model_cfg.get('aux_channels', 16),
            hidden_dim=get_hidden_dim(model_cfg),
        ).to(self.device)

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.ema_aux = copy.deepcopy(self.aux_head_model)
        self.ema_aux.requires_grad_(False)
        self.ema_aux.eval()

        self.adapter = create_adapter(model_type, run_config.data.heterogeneous)
        self.loss_computer = LossComputer(
            run_config.loss,
            heterogeneous=run_config.data.heterogeneous,
            pres_min=self.data.dataset.pres_min,
            pres_max=self.data.dataset.pres_max,
        )

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.aux_head_model.parameters()),
            lr=run_config.training.lr,
            weight_decay=run_config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._build_lr_lambda(),
        )

        self.global_step = 0
        self.wandb_run_id = None

    def _build_lr_lambda(self):
        train_cfg = self.cfg.training
        total_steps = len(self.data.train_loader) * train_cfg.num_epochs

        def lr_lambda(step):
            if train_cfg.warmup_steps > 0 and step < train_cfg.warmup_steps:
                return float(step + 1) / float(train_cfg.warmup_steps)
            progress = float(step - train_cfg.warmup_steps) / float(
                max(1, total_steps - train_cfg.warmup_steps))
            min_ratio = train_cfg.min_lr / train_cfg.lr
            return min_ratio + (1.0 - min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress))

        return lr_lambda

    def run(self):
        resume_state = self.checkpoints.detect_resume()
        start_epoch = resume_state.start_epoch
        self.global_step = resume_state.global_step
        self.wandb_run_id = resume_state.wandb_run_id

        num_epochs = self.cfg.training.num_epochs
        if start_epoch >= num_epochs:
            print(f"[RESUME] Training already complete "
                  f"({start_epoch}/{num_epochs} epochs). Exiting.")
            return

        end_epoch = min(start_epoch + self.cfg.training.epochs_per_run, num_epochs)

        self.wandb_run_id = self.logger.setup(
            model=self.model,
            aux_head_model=self.aux_head_model,
            wandb_run_id=self.wandb_run_id,
        )
        self.checkpoints.apply_resume_state(
            resume_state,
            model=self.model,
            ema_model=self.ema_model,
            aux_head=self.aux_head_model,
            ema_aux=self.ema_aux,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loader_gen=self.data.loader_gen,
        )

        print(f"Training epochs {start_epoch+1} to {end_epoch} "
              f"(of {num_epochs} total)")

        for epoch in range(start_epoch, end_epoch):
            self._run_epoch(epoch)

        self._finish_segment(start_epoch, end_epoch, num_epochs)

    def _run_epoch(self, epoch):
        self.model.train()

        for batch_idx, batch in enumerate(self.data.train_loader):
            one_step_loss, loss_l2_rel, loss_pf, k, k_upper = self._train_batch(batch)
            total_loss = one_step_loss.loss + loss_pf

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.aux_head_model.parameters()),
                max_norm=self.cfg.training.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
            update_ema(
                self.ema_model, self.model,
                self.ema_aux, self.aux_head_model,
                self.cfg.training.ema_decay,
            )

            self._log_batch(
                epoch=epoch,
                batch_idx=batch_idx,
                one_step_loss=one_step_loss,
                loss_l2_rel=loss_l2_rel,
                total_loss=total_loss,
                loss_pf=loss_pf,
                k=k,
            )

            self.global_step += 1

        if (epoch + 1) % self.cfg.checkpoints.save_every == 0:
            self.checkpoints.save_resume(
                model=self.model,
                ema_model=self.ema_model,
                aux_head=self.aux_head_model,
                ema_aux=self.ema_aux,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                loader_gen=self.data.loader_gen,
                wandb_run_id=self.wandb_run_id,
            )
            print(f"Epoch {epoch + 1}, k_upper={k_upper}, "
                  f"LossMSE: {one_step_loss.loss_mse.item():.5f}, "
                  f"LossH1: {one_step_loss.loss_h1.item():.5f}")

    def _train_batch(self, batch):
        (
            y_history, action_history, valid_k, y_t, y_tp1, action_t,
            aux_t, aux_tp1, static,
        ) = self._unpack_batch(batch)

        train_cfg = self.cfg.training
        k_upper = min(
            train_cfg.pushforward_k_max,
            1 + self.global_step // train_cfg.pushforward_k_step_interval,
        )
        k = np.random.randint(1, k_upper + 1)

        self._mask_actions(action_history, action_t)

        y_history = y_history.to(self.device)
        action_history = action_history.to(self.device)
        valid_k = valid_k.to(self.device)
        y_t = y_t.to(self.device)
        y_tp1 = y_tp1.to(self.device)
        action_t = action_t.to(self.device)
        aux_t = aux_t.to(self.device)
        aux_tp1 = aux_tp1.to(self.device)

        if torch.rand(1).item() < 0.8:
            y_noisy = add_adaptive_noise(y_t, alpha=train_cfg.noise_alpha)
        else:
            y_noisy = y_t

        model_input = self.adapter.build_model_input(y_noisy, action_t, static)
        predicted_y = self.adapter.forward(self.model, model_input)
        predicted_aux = self.aux_head_model(y_t, predicted_y)

        one_step_loss = self.loss_computer.compute_one_step_loss(
            predicted_y=predicted_y,
            predicted_aux=predicted_aux,
            y_t=y_t,
            y_tp1=y_tp1,
            action_t=action_t,
            aux_tp1=aux_tp1,
            static=static,
        )
        loss_pf = self._compute_pushforward_loss(
            k=k,
            y_history=y_history,
            action_history=action_history,
            valid_k=valid_k,
            y_tp1=y_tp1,
            action_t=action_t,
            aux_tp1=aux_tp1,
            static=static,
        )

        with torch.no_grad():
            loss_l2_rel = self.loss_computer.l2_relative(predicted_y, y_tp1)

        return one_step_loss, loss_l2_rel, loss_pf, k, k_upper

    def _unpack_batch(self, batch):
        if self.cfg.data.heterogeneous:
            (
                y_history, action_history, valid_k, y_t, y_tp1, action_t,
                aux_t, aux_tp1, static,
            ) = batch
            static = static.to(self.device)
        else:
            (
                y_history, action_history, valid_k, y_t, y_tp1, action_t,
                aux_t, aux_tp1,
            ) = batch
            static = None
        return y_history, action_history, valid_k, y_t, y_tp1, action_t, aux_t, aux_tp1, static

    def _mask_actions(self, action_history, action_t):
        for well in WELL_COORDS:
            action_history[:, :, 0:2, well[0], well[1]] = 0.
            action_t[:, 0:2, well[0], well[1]] = 0.

    def _compute_pushforward_loss(
        self, *, k, y_history, action_history, valid_k, y_tp1,
        action_t, aux_tp1, static
    ):
        device = y_tp1.device
        loss_pf = torch.tensor(0.0, device=device)
        train_cfg = self.cfg.training

        if not train_cfg.use_pushforward or k <= 0:
            return loss_pf

        start_idx = train_cfg.pushforward_k_max - k
        pf_mask = valid_k[:, start_idx].bool()
        if not pf_mask.any():
            return loss_pf

        static_pf = static[pf_mask] if static is not None else None

        with torch.no_grad():
            self.model.eval()
            y_pf = y_history[pf_mask, start_idx]
            for i in range(k):
                pf_input_i = self.adapter.build_model_input(
                    y_pf, action_history[pf_mask, start_idx + i], static_pf)
                y_pf = self.adapter.forward(self.model, pf_input_i)
            self.model.train()

        y_pf = y_pf.detach()
        target_pf = y_tp1[pf_mask]

        pred_pf = self.adapter.forward(
            self.model,
            self.adapter.build_model_input(y_pf, action_t[pf_mask], static_pf),
        )
        pred_pf_aux = self.aux_head_model(y_pf, pred_pf)

        return self.loss_computer.compute_pushforward_loss(
            y_pf=y_pf,
            pred_pf=pred_pf,
            pred_pf_aux=pred_pf_aux,
            target_pf=target_pf,
            action_t=action_t[pf_mask],
            aux_tp1=aux_tp1[pf_mask],
            static=static_pf,
        )

    def _log_batch(self, *, epoch, batch_idx, one_step_loss, loss_l2_rel,
                   total_loss, loss_pf, k):
        if self.logger.writer:
            if self.global_step % self.cfg.logging.log_scalar_every == 0:
                self.logger.log_training_scalars(
                    one_step_loss=one_step_loss,
                    loss_l2_rel=loss_l2_rel,
                    total_loss=total_loss,
                    loss_pf=loss_pf,
                    k=k,
                    lr=self.optimizer.param_groups[0]['lr'],
                    global_step=self.global_step,
                )

            if epoch % self.cfg.training.log_every == 0 and batch_idx == 0:
                self.logger.run_validation(
                    model=self.ema_model,
                    aux_model=self.ema_aux,
                    test_loader=self.data.test_loader,
                    adapter=self.adapter,
                    loss_computer=self.loss_computer,
                    device=self.device,
                    heterogeneous=self.cfg.data.heterogeneous,
                    global_step=self.global_step,
                )
                self.model.train()
        else:
            print(f"Batch {batch_idx}, k={k}, "
                  f"LossMSE: {one_step_loss.loss_mse.item():.5f}, "
                  f"LossH1: {one_step_loss.loss_h1.item():.5f}, "
                  f"LossPF: {loss_pf.item():.5f}, "
                  f"LossSpec: {one_step_loss.loss_spectral.item():.5f}")

    def _finish_segment(self, start_epoch, end_epoch, num_epochs):
        training_complete = (end_epoch >= num_epochs)

        if training_complete:
            self.checkpoints.save_final(
                model=self.model,
                ema_model=self.ema_model,
                aux_head=self.aux_head_model,
                ema_aux=self.ema_aux,
            )
            self.checkpoints.remove_resume_files()
            self.logger.log_final_artifact(self.checkpoints.final_path)
            self.logger.finish()
            print(f"[DONE] All {num_epochs} epochs complete.")
        else:
            self.checkpoints.save_resume(
                model=self.model,
                ema_model=self.ema_model,
                aux_head=self.aux_head_model,
                ema_aux=self.ema_aux,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=end_epoch - 1,
                global_step=self.global_step,
                loader_gen=self.data.loader_gen,
                wandb_run_id=self.wandb_run_id,
            )
            self.logger.finish(exit_code=0)
            print(f"[SEGMENT DONE] Epochs {start_epoch+1}-{end_epoch} "
                  f"of {num_epochs} complete.")


def run_training(cfg, args, resume_path=None):
    run_config = build_run_config(cfg, args)
    set_seed(run_config.seed)
    print(f"Seed set to {run_config.seed}")

    if os.path.isfile(run_config.checkpoints.final_path):
        print(f"[DONE] Final model already exists at "
              f"{run_config.checkpoints.final_path}, skipping.")
        return

    trainer = Trainer(run_config, resume_path=resume_path)
    trainer.run()
