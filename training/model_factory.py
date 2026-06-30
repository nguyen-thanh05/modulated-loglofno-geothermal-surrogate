from models.unet3d import UNet3D
from models.fno_wrapper import FNOWrapper
from models.loglo_fno import ModulatedLOGLO_FNO, VanillaLOGLO_FNO
from models.transolver3d import TransolverWrapper


def create_model(model_cfg, model_type):
    if model_type == 'unet3d':
        return UNet3D(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_channels=model_cfg['hidden_channels'],
            depth=model_cfg.get('depth', 3),
            channel_multipliers=model_cfg.get('channel_multipliers', None),
        )
    elif model_type == 'fno':
        return FNOWrapper(
            n_modes=model_cfg['n_modes'],
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            n_layers=model_cfg['n_layers'],
            hidden_channels=model_cfg['hidden_channels'],
        )
    elif model_type == 'modulated_loglo':
        return ModulatedLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
            action_channels=model_cfg['action_channels'],
        )
    elif model_type == 'vanilla_loglo':
        return VanillaLOGLO_FNO(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            lifting_dim=model_cfg['lifting_dim'],
            projection_dim=model_cfg['projection_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
        )
    elif model_type == 'transolver':
        return TransolverWrapper(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['out_channels'],
            hidden_dim=model_cfg['hidden_dim'],
            n_layers=model_cfg['n_layers'],
            n_head=model_cfg['n_head'],
            slice_num=model_cfg.get('slice_num', 32),
            mlp_ratio=model_cfg.get('mlp_ratio', 2),
            H=model_cfg.get('H', 16),
            W=model_cfg.get('W', 64),
            D=model_cfg.get('D', 32),
            spatial_embed=model_cfg.get('spatial_embed', True),
            num_bands=model_cfg.get('num_bands', 32),
            max_freq=model_cfg.get('max_freq', 64.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_out_channels(model_cfg):
    return model_cfg.get('out_channels', model_cfg.get('out_dim', 4))


def get_hidden_dim(model_cfg):
    return model_cfg.get('hidden_channels', model_cfg.get('hidden_dim', 64))
