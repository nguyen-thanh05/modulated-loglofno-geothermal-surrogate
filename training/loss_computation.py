from dataclasses import dataclass

import torch
import torch.nn as nn

from losses import H1Loss, LpLoss
from training.physics import (
    compute_mbe_loss, extract_physical_porosity, mean_field_pressure_loss,
    radial_binned_spectral_loss,
)
from training.utils import build_action_for_mbe


@dataclass
class OneStepLoss:
    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_h1: torch.Tensor
    loss_aux: torch.Tensor
    loss_mbe: torch.Tensor
    loss_meanfield: torch.Tensor
    loss_spectral: torch.Tensor
    spectral_bands: torch.Tensor


class LossComputer:
    def __init__(self, loss_config, *, heterogeneous, pres_min, pres_max):
        self.cfg = loss_config
        self.heterogeneous = heterogeneous
        self.pres_min = pres_min
        self.pres_max = pres_max
        self.mse_fn = nn.MSELoss()
        self.h1_loss_fn = H1Loss(
            d=3,
            reduction='none',
            fix_x_bnd=True,
            fix_y_bnd=True,
            fix_z_bnd=True,
            measure=[0.25, 1., 0.5],
        ).abs
        self.l2_rel = LpLoss(d=3, p=2, reduction='mean', measure=[0.25, 1., 0.5])
        self._channel_weights = torch.tensor(loss_config.channel_weights)

    def calculate_weighted_mse_loss(self, pred, target):
        diff = (pred - target) ** 2
        weights = self._channel_weights.to(pred.device)
        return torch.mean(diff * weights.view(1, -1, 1, 1, 1))

    def calculate_weighted_h1_loss(self, pred, target):
        per_channel = self.h1_loss_fn(pred, target)
        weights = self._channel_weights.to(pred.device)
        return (per_channel * weights).mean()

    def get_mbe_porosity(self, static):
        if self.heterogeneous:
            return extract_physical_porosity(static)
        return 0.05, 0.005

    def compute_one_step_loss(
        self, *, predicted_y, predicted_aux, y_t, y_tp1, action_t, aux_tp1, static
    ):
        cfg = self.cfg
        device = predicted_y.device
        loss = torch.tensor(0.0, device=device)

        if cfg.use_mse:
            loss_mse = self.calculate_weighted_mse_loss(predicted_y, y_tp1)
            loss = loss + cfg.mse_weight * loss_mse
        else:
            loss_mse = torch.tensor(0.0, device=device)

        if cfg.use_h1:
            loss_h1 = self.calculate_weighted_h1_loss(predicted_y, y_tp1)
            loss = loss + cfg.h1_weight * loss_h1
        else:
            loss_h1 = torch.tensor(0.0, device=device)

        if cfg.use_aux:
            loss_aux = cfg.aux_weight * self.mse_fn(predicted_aux, aux_tp1)
            loss = loss + loss_aux
        else:
            loss_aux = torch.tensor(0.0, device=device)

        if cfg.use_mbe:
            phi_m, phi_frac = self.get_mbe_porosity(static)
            action_mbe = build_action_for_mbe(action_t)
            loss_mbe = cfg.mbe_weight * compute_mbe_loss(
                y_t, predicted_y, action_mbe, phi_m, phi_frac,
                pres_min=self.pres_min, pres_max=self.pres_max)
            loss = loss + loss_mbe
        else:
            loss_mbe = torch.tensor(0.0, device=device)

        if cfg.use_meanfield:
            loss_meanfield = cfg.meanfield_weight * mean_field_pressure_loss(predicted_y, y_tp1)
            loss = loss + loss_meanfield
        else:
            loss_meanfield = torch.tensor(0.0, device=device)

        if cfg.use_spectral and cfg.spectral_weight > 0:
            loss_spectral, spectral_bands = radial_binned_spectral_loss(
                predicted_y, y_tp1, iLow=cfg.spectral_iLow, iHigh=cfg.spectral_iHigh)
            loss = loss + cfg.spectral_weight * loss_spectral
        else:
            loss_spectral = torch.tensor(0.0, device=device)
            spectral_bands = torch.zeros(3, device=device)

        return OneStepLoss(
            loss=loss,
            loss_mse=loss_mse,
            loss_h1=loss_h1,
            loss_aux=loss_aux,
            loss_mbe=loss_mbe,
            loss_meanfield=loss_meanfield,
            loss_spectral=loss_spectral,
            spectral_bands=spectral_bands,
        )

    def compute_pushforward_loss(
        self, *, y_pf, pred_pf, pred_pf_aux, target_pf, action_t, aux_tp1, static
    ):
        cfg = self.cfg
        device = pred_pf.device
        loss_pf = torch.tensor(0.0, device=device)

        if cfg.use_mse:
            loss_pf = loss_pf + cfg.mse_weight * self.calculate_weighted_mse_loss(
                pred_pf, target_pf)
        if cfg.use_h1:
            loss_pf = loss_pf + cfg.h1_weight * self.calculate_weighted_h1_loss(
                pred_pf, target_pf)
        if cfg.use_aux:
            loss_pf = loss_pf + cfg.aux_weight * self.mse_fn(pred_pf_aux, aux_tp1)
        if cfg.use_mbe:
            phi_m_pf, phi_frac_pf = self.get_mbe_porosity(static)
            action_mbe_pf = build_action_for_mbe(action_t)
            loss_pf = loss_pf + cfg.mbe_weight * compute_mbe_loss(
                y_pf, pred_pf, action_mbe_pf, phi_m_pf, phi_frac_pf,
                pres_min=self.pres_min, pres_max=self.pres_max)
        if cfg.use_meanfield:
            loss_pf = loss_pf + cfg.meanfield_weight * mean_field_pressure_loss(
                pred_pf, target_pf)
        if cfg.use_spectral and cfg.spectral_weight > 0:
            loss_pf_spec, _ = radial_binned_spectral_loss(
                pred_pf, target_pf, iLow=cfg.spectral_iLow, iHigh=cfg.spectral_iHigh)
            loss_pf = loss_pf + cfg.spectral_weight * loss_pf_spec

        return loss_pf

    def l2_relative(self, pred, target):
        return self.l2_rel(pred, target)
